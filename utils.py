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
    """Accumulates wall-clock time per stage. Disabled by default (no syncs, no timing).

    Parts of a stage (sub-rows named '<stage>_<part>', e.g. 'traceback_kernel') are kept apart from the
    stages, so they never count twice in 'other' or the total.
    """
    def __init__(self, enabled=False, devices=('cpu',)):
        self.enabled = enabled
        self.devices = {str(d) for d in ([devices] if isinstance(devices, str) else devices)}
        self.times = defaultdict(float)
        self.totals = defaultdict(float)
        self.parts = defaultdict(float)
        self.part_totals = defaultdict(float)

    def sync(self):
        # GPU kernels run asynchronously; wait for them so time lands in the right stage
        if any(d.startswith('cuda') for d in self.devices):
            torch.cuda.synchronize()
        if any(d.startswith('mps') for d in self.devices):
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

    @contextmanager
    def part(self, name):
        """Wall-clock time of one part of a stage (synced like section)."""
        if not self.enabled:
            yield
            return
        self.sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.parts[name] += time.perf_counter() - t

    @contextmanager
    def gpu_part(self, name):
        """GPU execution time of the work queued inside the block, measured with CUDA events
        (kernel time only: no CPU time, no transfers)."""
        if not self.enabled:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            end.synchronize()
            self.parts[name] += start.elapsed_time(end) / 1e3

    def reset(self):
        self.times = defaultdict(float)
        self.parts = defaultdict(float)

    def report(self, header, total, times=None, parts=None):
        times = self.times if times is None else times
        parts = self.parts if parts is None else parts
        stages = PROFILE_STAGES + [s for s in times if s not in PROFILE_STAGES]
        other = total - sum(times.values())
        print(f'Profile ({header}):')
        for s in stages + ['other']:
            t = other if s == 'other' else times.get(s, 0.0)
            pct = 100 * t / total if total > 0 else 0.0
            print(f'  {s:<18}{t:>10.3f}s {pct:>6.1f}%')
            for p, tp in parts.items():
                if p.startswith(s + '_'):
                    pct = 100 * tp / total if total > 0 else 0.0
                    print(f'    {p:<18}{tp:>8.3f}s {pct:>6.1f}%')
        print(f'  {"total":<18}{total:>10.3f}s')

    def end_family(self, header, total):
        """Print this family's breakdown and add it to the run totals."""
        if not self.enabled:
            return
        self.report(header, total)
        for s, t in self.times.items():
            self.totals[s] += t
        for p, t in self.parts.items():
            self.part_totals[p] += t
        self.totals['_total'] += total
        self.reset()

    def report_totals(self, header):
        if not self.enabled:
            return
        times = {s: t for s, t in self.totals.items() if s != '_total'}
        self.report(header, self.totals['_total'], times=times, parts=self.part_totals)
