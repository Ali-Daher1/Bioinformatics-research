from utils import *
from msa_tools import *
from plm_wrapper import *
from scoring import *
from pairwise import *
from clustal import *


class ARIES:
    def __init__(self, **kwargs):
        self.plm = kwargs['plm']
        self.w = kwargs.get('window', 5)
        self.h = kwargs.get('num_hidden_states', 9)
        self.b = kwargs.get('batch', 1)
        self.r = kwargs.get('reciprocal', 200.)
        self.g = kwargs.get('blur', 3.)
        self.pad_char = kwargs.get('pad_char', 'X')
        self.prof = kwargs.get('profiler') or Profiler()
        # device for similarity, DP fill, traceback and MSA assembly (the PLM has its own)
        self.align_device = torch.device(kwargs.get('align_device', 'cpu'))

        # medoid config
        self.medoid_mode = kwargs.get('medoid_mode', 'edit')
        self.topk = kwargs.get('medoid_topk', 'log')
        self.guide_tree_tool = kwargs.get('guide_tree_tool', 'clustalo')
        
        # sim config
        self.sim_metric = kwargs.get('sim_metric', 'l2-gm')

        aligner_tokens = kwargs.get('aligner', 'aries').split('-')
        dtw = kwargs.get('dtw', 'torch')
        if aligner_tokens[0] == 'aries' and dtw != 'torch':
            from dtw_kernels import CudaKernelDTW, KERNELS  # needs CuPy; imported only when chosen
            self.aligner = CudaKernelDTW(batch=True, profiler=self.prof, fill=KERNELS[dtw])
        elif aligner_tokens[0] == 'aries':
            self.aligner = DTW(batch=True, profiler=self.prof)
        elif aligner_tokens[0] == 'turnpen':
            self.aligner = TurnPenaltyDTW(tau=float(aligner_tokens[1]), batch=True, profiler=self.prof)

    def get_embeddings(self, seqs, return_logits=False):
        padded_seqs = polyX_paddings(seqs, pad_char=self.pad_char, w=self.w)
        with self.prof.section('embedding'):
            embeddings, logits = self.plm(padded_seqs, num_hidden_states=self.h, batch=self.b)
        if return_logits:
            return embeddings, logits
        return embeddings

    def build_msa(self, embeddings, seqs, consensus_embedding, verbose=True, paths=None, pairwise_scores=None):
        template_length = consensus_embedding.shape[0] - 2 * self.w
        if paths is None:
            if verbose:
                print(f'>>>> Aligning with template')
            _, paths, pairwise_scores = self.aligner.align_embeddings(
                consensus_embedding, embeddings,
                w=self.w, reciprocal=self.r, blur=self.g,
                return_scores=True, sim_metric=self.sim_metric
            )
        with self.prof.section('msa_assembly'):
            return self.assemble_columns(seqs, template_length, paths, pairwise_scores, verbose=verbose)

    def assemble_columns(self, seqs, template_length, paths, pairwise_scores, verbose=True):
        num_seqs = len(seqs)
        if self.align_device.type == 'cpu':
            cols = self.infer_columns_loop(seqs, template_length, paths, pairwise_scores, verbose=verbose)
        else:
            cols = self.infer_columns_vectorized(seqs, template_length, paths, pairwise_scores, verbose=verbose)

        if verbose:
            print(f'>>>> Centering columns')
        for t, (col_left, col_right) in enumerate(cols):
            left_width = max([len(s) for s in col_left])
            right_width = max([len(s) for s in col_right])
            col = []
            for i in range(num_seqs):
                pad_left = (left_width - len(col_left[i])) * '-'
                pad_right = (right_width - len(col_right[i])) * '-'
                col.append(pad_left + col_left[i] + col_right[i] + pad_right)
            cols[t] = col
        alns = [
            ''.join([c[i] for c in cols])
            for i in range(num_seqs)
        ]
        del cols
        return alns

    def infer_columns_vectorized(self, seqs, template_length, paths, pairwise_scores, verbose=True):
        # Same result as infer_columns_loop, but each step is one masked argmax on align_device
        if verbose:
            print(f'>>>> Assigning columns')
        lefts, rights = [], []
        for i in range(len(seqs)):
            S = pairwise_scores[i].to(self.align_device)
            L = len(seqs[i])
            path = torch.tensor(paths[i], device=S.device)
            on_path = torch.zeros_like(S, dtype=torch.bool)
            on_path[path[:, 0], path[:, 1]] = True
            # each residue keeps its best-scoring template position on the path
            best_t = S.masked_fill(~on_path, -float('inf')).argmax(dim=0)
            assigned = torch.zeros_like(on_path)
            assigned[best_t, torch.arange(L, device=S.device)] = True
            # anchor = best-scoring residue among those assigned to each template position
            anchor_j = S.masked_fill(~assigned, -float('inf')).argmax(dim=1)
            best_t, anchor_j = best_t.tolist(), anchor_j.tolist()
            left, right = defaultdict(str), defaultdict(str)
            for j, t in enumerate(best_t):
                if j <= anchor_j[t]:
                    left[t] += seqs[i][j]
                else:
                    right[t] += seqs[i][j]
            lefts.append(left)
            rights.append(right)

        if verbose:
            print(f'>>>> Inferring anchors')
        cols = []
        for t in range(template_length):
            col_left = [left.get(t, '') for left in lefts]
            col_right = [right.get(t, '') for right in rights]
            if any(col_left) or any(col_right):
                cols.append((col_left, col_right))
        return cols

    def infer_columns_loop(self, seqs, template_length, paths, pairwise_scores, verbose=True):
        num_seqs = len(seqs)
        edges = []
        for i in range(num_seqs):
            edges.append(torch.zeros(template_length, len(seqs[i]), dtype=int))
            for u, v in paths[i]:
                edges[i][u][v] = 1  # edge from template[u] to si[v]
        if verbose:
            print(f'>>>> Assigning columns')
        for i in range(len(edges)):
            for j in range(len(seqs[i])):
                template_residues = torch.where(edges[i][:, j])[0]
                best_match = template_residues[torch.argmax(pairwise_scores[i][template_residues, j])]
                edges[i][template_residues, j] = 0
                edges[i][best_match, j] = 1

        if verbose:
            print(f'>>>> Inferring anchors')
        cols = []
        for t in range(template_length):
            col_left, col_right = [], []
            for i in range(num_seqs):
                edges_from_t = torch.where(edges[i][t] > 0)[0]
                if edges_from_t.numel() == 0:
                    col_left.append('')
                    col_right.append('')
                    continue
                best_match_idx = torch.argmax(pairwise_scores[i][t][edges_from_t])
                left_idx = edges_from_t[:best_match_idx + 1].tolist()
                right_idx = edges_from_t[best_match_idx + 1:].tolist()
                col_left.append(''.join([seqs[i][j] for j in left_idx]))
                col_right.append(''.join([seqs[i][j] for j in right_idx]))
            left_width = max([len(s) for s in col_left])
            right_width = max([len(s) for s in col_right])
            if (left_width + right_width) == 0:
                continue
            else:
                cols.append((col_left, col_right))
        return cols

    def align(self, seqs, **kwargs):
        print(f'Building template')
        t = time.time()
        msa_name = kwargs['msa_name']
        if self.medoid_mode == 'dnd':
            with self.prof.section('guide_tree'):
                if self.guide_tree_tool == 'clustalo':
                    path = f'{temp_dir}/{msa_name}_clustalo.dnd'
                    if not os.path.exists(path):
                        run_clustalo(seqs, msa_name, guidetree_only=True)
                else:
                    path = f'{temp_dir}/{msa_name}_clustalw.dnd'
                    if not os.path.exists(path):
                        run_clustalw(seqs, msa_name, guidetree_only=True)
        with self.prof.section('medoid_selection'):
            medoids = topk_medoids(seqs, k=self.topk, mode=self.medoid_mode,
                                   guide_tree_tool=self.guide_tree_tool, **kwargs)
        if self.medoid_mode == 'dnd' and any(m >= len(seqs) for m in medoids):
            # Stale cached .dnd from a run with more sequences — regenerate.
            os.remove(path)
            with self.prof.section('guide_tree'):
                if self.guide_tree_tool == 'clustalo':
                    run_clustalo(seqs, msa_name, guidetree_only=True)
                else:
                    run_clustalw(seqs, msa_name, guidetree_only=True)
            with self.prof.section('medoid_selection'):
                medoids = topk_medoids(seqs, k=self.topk, mode=self.medoid_mode,
                                       guide_tree_tool=self.guide_tree_tool, **kwargs)
        medoid_seqs = [seqs[m] for m in medoids]

        print(f'Embed {len(medoid_seqs)} medoid sequences with PLM')
        t_emb = time.time()
        medoid_embs = self.get_embeddings(medoid_seqs)
        emb_time = time.time() - t_emb

        # the template's device decides where similarity and DP run (pairwise.align_embeddings)
        global_medoid_emb = medoid_embs[0].to(self.align_device)
        medoid_aln = self.build_msa(medoid_embs, medoid_seqs, global_medoid_emb)
        consensus = [s.replace('-', 'X') for s in medoid_aln]
        consensus_emb = torch.stack(self.get_embeddings(consensus), dim=0).to(self.align_device)
        mask = torch.tensor([[1 if c != '-' else 0 for c in s] for s in polyX_paddings(medoid_aln, pad_char=self.pad_char, w=self.w)]).to(consensus_emb.device)
        consensus_emb = (consensus_emb * mask[:, :, None]).sum(dim=0) / mask.sum(dim=0)[:, None]

        print(f'Building MSA given template')
        print(f'Embed {len(seqs)} sequences with PLM')
        all_paths, all_scores = [], []
        for i in range(0, len(seqs), self.b):
            batch_embs = self.get_embeddings(seqs[i:i + self.b])
            _, paths, scores = self.aligner.align_embeddings(
                consensus_emb, batch_embs,
                w=self.w, reciprocal=self.r, blur=self.g,
                return_scores=True, sim_metric=self.sim_metric,
            )
            all_paths.extend(paths)
            # keep finished score matrices off the GPU; large families would not fit
            all_scores.extend(s.cpu() for s in scores)

        alns = self.build_msa(None, seqs, consensus_emb, paths=all_paths, pairwise_scores=all_scores)
        aln_time = time.time() - t
        return alns, (emb_time, aln_time)
