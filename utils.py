# set up directories
import os
temp_dir = os.path.abspath("./tmp")
os.makedirs(temp_dir, exist_ok=True)

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from torch.nn.utils.rnn import pad_sequence
# Analytics
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from itertools import combinations, permutations, product
# System utilities
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from itertools import product, combinations
from tqdm import tqdm, trange
from copy import deepcopy
from pprint import pprint
from functools import partial
from contextlib import contextmanager
from io import StringIO
from datetime import datetime
from collections import Counter
import random, gc, re, sys, requests, glob, time, math, subprocess
# BioPython
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Levenshtein import distance as edit_distance
from Bio.Align import substitution_matrices

# PyTorch utilities
def assess_cuda_memory(header):
    print(f'{header}-------------------------------')
    mem_alloc = torch.cuda.memory_allocated() / 1024 ** 2
    mem_cache = torch.cuda.memory_reserved() / 1024 ** 2
    print(f'Allocated: {mem_alloc:.2f}MB, Cache: {mem_cache:.2f}')

def freeze_module(module, unfreeze=False):
    for param in module.parameters():
        param.requires_grad = unfreeze
def set_seed(seed=2603):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def num_params(net):
    return sum(p.numel() for p in net.parameters())

def spawn_fasta(seqs, fasta_path, maxlen=None):
    maxlen = maxlen or max((len(s) for s in seqs), default=0)
    if maxlen == 0:
        raise ValueError("spawn_fasta received only empty sequences")
    records = [SeqRecord(Seq(s if len(s) > 0 else '-' * maxlen), id=f'seq{i}', description="") for i, s in enumerate(seqs)]
    with open(fasta_path, 'w') as output_handle:
        SeqIO.write(records, output_handle, "fasta")

# Stage profiling
PROFILE_STAGES = ['guide_tree', 'medoid_selection', 'embedding', 'similarity', 'dp_fill', 'traceback', 'msa_assembly']

class Profiler:
    """Accumulates wall-clock time per stage. Disabled by default (no syncs, no timing)."""
    def __init__(self, enabled=False, device='cpu'):
        self.enabled = enabled
        self.device = str(device)
        self.times = defaultdict(float)
        self.totals = defaultdict(float)

    def sync(self):
        # GPU kernels run asynchronously; wait for them so time lands in the right stage
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        elif self.device.startswith('mps'):
            torch.mps.synchronize()

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        self.sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.times[name] += time.perf_counter() - t

    def reset(self):
        self.times = defaultdict(float)

    def report(self, header, total, times=None):
        times = self.times if times is None else times
        stages = PROFILE_STAGES + [s for s in times if s not in PROFILE_STAGES]
        other = total - sum(times.values())
        print(f'Profile ({header}):')
        for s in stages + ['other']:
            t = other if s == 'other' else times.get(s, 0.0)
            pct = 100 * t / total if total > 0 else 0.0
            print(f'  {s:<18}{t:>10.3f}s {pct:>6.1f}%')
        print(f'  {"total":<18}{total:>10.3f}s')

    def end_family(self, header, total):
        """Print this family's breakdown and add it to the run totals."""
        if not self.enabled:
            return
        self.report(header, total)
        for s, t in self.times.items():
            self.totals[s] += t
        self.totals['_total'] += total
        self.reset()

    def report_totals(self, header):
        if not self.enabled:
            return
        times = {s: t for s, t in self.totals.items() if s != '_total'}
        self.report(header, self.totals['_total'], times=times)
