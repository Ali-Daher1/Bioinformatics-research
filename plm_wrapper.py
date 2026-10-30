from utils import *
from transformers import AutoModel, AutoTokenizer, EsmModel, AutoModelForMaskedLM, T5Tokenizer, T5ForConditionalGeneration, AutoConfig, T5EncoderModel, EsmForMaskedLM
from typing import List, Optional, Tuple, Union

MODELS = {
    'prottrans-half': 'Rostlab/prot_t5_xl_half_uniref50-enc',
    'prottrans': 'Rostlab/prot_t5_xl_uniref50',
    'protbert': 'Rostlab/prot_bert',
    'esm2-35M': 'facebook/esm2_t12_35M_UR50D',
    'esm2-150M': 'facebook/esm2_t30_150M_UR50D',
    'esm2-650M': 'facebook/esm2_t33_650M_UR50D'
}

class PLMWrapper(nn.Module):
    def __init__(self, plm, **kwargs):
        super(PLMWrapper, self).__init__()
        self.model_name = plm
        if plm in MODELS:
            if 'prottrans' in plm:
                self.plm = T5ForConditionalGeneration.from_pretrained(MODELS[plm])
                self.tokenizer = T5Tokenizer.from_pretrained(MODELS[plm])
            elif plm == 'protbert':
                self.plm = AutoModelForMaskedLM.from_pretrained(MODELS[plm])
                self.tokenizer = AutoTokenizer.from_pretrained(MODELS[plm])
            elif 'esm2' in plm:
                self.plm = EsmForMaskedLM.from_pretrained(MODELS[plm])
                self.tokenizer = AutoTokenizer.from_pretrained(MODELS[plm])
            else:
                raise ValueError(f'PLM model {plm} not supported.')
        else:
            try:
                if 'prot_t5' in plm:
                    self.plm = T5ForConditionalGeneration.from_pretrained(plm)
                    self.tokenizer = T5Tokenizer.from_pretrained(plm)
                elif "esm2" in plm:
                    self.plm = EsmForMaskedLM.from_pretrained(plm)
                    self.tokenizer = AutoTokenizer.from_pretrained(plm)
                else:
                    self.plm = AutoModelForMaskedLM.from_pretrained(plm)
                    self.tokenizer = AutoTokenizer.from_pretrained(plm)
            except:
                raise ValueError(f'PLM model {plm} not supported.')

        self.hidden_size = self.plm.config.hidden_size
        self.prof = Profiler()      # ARIES sets its own profiler here to time the copy to the CPU
        self.is_t5 = getattr(self.plm.config, "model_type", "") == "t5"
        self.plm.gradient_checkpointing_enable()
        if kwargs.get('freeze_backbone', True):
            self.plm.eval()
            freeze_module(self.plm)

    # Take input sequence, return input ids and attention_mask
    def tokenize(self, input_seqs):
        input_seqs = [' '.join(s) for s in input_seqs]
        # Allow using '!' as a stand-in for the tokenizer's pad token.
        pad_tok = self.tokenizer.pad_token
        if pad_tok is None:
            # Fallback: treat '!' as a normal residue (X) when no pad token exists.
            pad_tok = "X"
        input_seqs = [s.replace('!', pad_tok) for s in input_seqs]
        tokenized_seqs = self.tokenizer(
            input_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = tokenized_seqs["input_ids"]
        attn_mask = tokenized_seqs["attention_mask"]
        return input_ids.to(self.plm.device), attn_mask.to(self.plm.device)
    
    def decode(self, hidden_state):
        logits = self.plm.lm_head(hidden_state * (self.plm.model_dim ** -0.5))
        seqs = self.tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
        return [s.replace(' ', '') for s in seqs]

    def forward_small_batch(self, input_ids, attn_mask, **kwargs):
        outputs = self.plm(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids, output_hidden_states=True, return_dict=True)
        hs = outputs.encoder_hidden_states if self.is_t5 else outputs.hidden_states
        retrieve_hs = kwargs.get('num_hidden_states', 1)
        if isinstance(retrieve_hs, list):
            # prottrans dont have CLS so we shave it away from esm embeddings (not required for alignment)
            embeddings = [hs[i][:, 0 if self.is_t5 else 1:] for i in retrieve_hs]
        else:
            # prottrans dont have CLS so we shave it away from esm embeddings (not required for alignment)
            assert isinstance(retrieve_hs, int), 'num_hidden_states must be either list or int'
            embeddings = [hs[i][:, 0 if self.is_t5 else 1:] for i in range(len(hs) - kwargs.get('num_hidden_states', 1), len(hs))]
        embeddings = torch.cat(embeddings, dim=-1)
        # prottrans dont have CLS so we shave it away from esm embeddings (not required for alignment)
        logits = outputs.logits[:, 0 if self.is_t5 else 1:]
        # one GPU -> CPU copy of the batch (concatenated on the PLM device first)
        with self.prof.part('embedding_copy'):
            embeddings, logits = embeddings.to('cpu'), logits.to('cpu')
        del outputs
        torch.cuda.empty_cache()
        return embeddings, logits

    def forward(self, seqs, **kwargs):
        batch = kwargs.get('batch', None)
        maxlen = kwargs.get('maxlen', 1022)
        overlap = kwargs.get('overlap', 750)
        tiles = []
        sid_to_tid = defaultdict(list)
        for i, s in enumerate(seqs):
            if len(s) < maxlen:
                tiles.append(s)
                sid_to_tid[i].append(len(tiles) - 1)
            else:
                start = 0
                while start < len(s):
                    tiles.append(s[start: min(len(s), start + maxlen)])
                    sid_to_tid[i].append(len(tiles) - 1)
                    start += overlap

        input_ids, attn_mask = self.tokenize(tiles) if kwargs.get('tokenize', True) else tiles
        if batch is None:
            enc_embeddings, logits = self.forward_small_batch(input_ids, attn_mask, **kwargs)
        else:
            enc_embeddings, logits = [], []
            for j in trange(0, len(tiles), batch, disable=len(tiles) <= batch):
                batch_end = min(j + batch, len(tiles))
                ij, aj, = input_ids[j: batch_end], attn_mask[j: batch_end]
                ej, lj = self.forward_small_batch(ij, aj, **kwargs)
                enc_embeddings.append(ej)
                logits.append(lj)
            enc_embeddings = torch.cat(enc_embeddings, dim=0)
            logits = torch.cat(logits, dim=0)

        # merge tiles, strip eos
        final_embeddings = []
        final_logits = []
        for i in range(len(seqs)):
            if len(sid_to_tid[i]) == 1:
                tid = sid_to_tid[i][0]
                tile_length = len(tiles[tid])
                final_embeddings.append(enc_embeddings[tid][:tile_length])
                final_logits.append(logits[tid][:tile_length])
            else:
                device = enc_embeddings.device
                logit = torch.zeros(len(seqs[i]), logits.shape[-1], device=device)
                embedding = torch.zeros(len(seqs[i]), enc_embeddings.shape[-1], device=device)
                num_tiles = torch.zeros(len(seqs[i]), device=device)
                for j, tid in enumerate(sid_to_tid[i]):
                    tile_length = len(tiles[tid])
                    start_idx = j * overlap
                    end_idx = start_idx + tile_length
                    logit[start_idx: end_idx] += logits[tid][:tile_length].to(device)
                    embedding[start_idx: end_idx] += enc_embeddings[tid][:tile_length].to(device)
                    num_tiles[start_idx: end_idx] += 1
                embedding = embedding / num_tiles[:, None]
                logit = logit/ num_tiles[:, None]
                final_embeddings.append(embedding.to('cpu'))
                final_logits.append(logit.to('cpu'))
                del embedding, logit, num_tiles
                torch.cuda.empty_cache()
        del enc_embeddings, logits
        torch.cuda.empty_cache()
        return final_embeddings, final_logits

if __name__ == '__main__':
    from msa_tools import *
    set_seed(123)
    plm = PLMWrapper('esm2-35M').to('cuda')
    seqs = [
        random_protein_sequence(length=random.randint(500, 750))
        for i in range(5)
    ]

    seqs = polypad_paddings(seqs, w=4)
    
    embeddings, logits = plm(seqs)

    pprint([len(s) for s in seqs])
    pprint([e.shape for e in embeddings])
