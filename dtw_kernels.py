"""CUDA DTW kernels for ARIES (same result as pairwise.DTW with batch=True).

Compiled at run time with CuPy (NVRTC, no C++ compiler needed). Selected with --align-device:

kernel0 (version 1):
  dtw_fill:        one thread block for each sequence; the threads share the cells of one
                   anti-diagonal; __syncthreads() after each anti-diagonal. It writes DP and
                   the step direction of each cell (0 diagonal, 1 up, 2 left).
kernel1 (version 2, frontier privatization of the fill):
  dtw_fill_shared: each thread block (one sequence) keeps only the three active anti-diagonals
                   (s-2, s-1, s) of its DP table in shared memory, indexed by the row i, and rotates them
                   after each anti-diagonal. The full DP table is not kept in global memory; the kernel
                   writes the step direction of each cell to DIR (global memory, as in version 1) and the
                   final score of each sequence to BEST.
Both versions:
  dtw_traceback:   one thread for each sequence; it follows the saved directions (it reads only DIR).
MSA assembly, column assignment (used by kernel0 and kernel1; same result as ARIES.infer_columns_loop):
  msa_best_template: one thread for each path cell (i, j); each residue j keeps the template position i with
                     the highest score on its path (ties: lowest i), with an atomicMax on a (score, index) key.
  msa_anchor:        one thread for each residue j; each template position t keeps as anchor the assigned
                     residue with the highest score (ties: lowest j), with the same kind of key.

The calculation of each cell (max of diagonal, up, left; then + M) and the tie order (diagonal, then up,
then left) are the same in both versions and as torch.argmax in DTW.backtrack.
"""
import cupy as cp
import torch

from pairwise import DTW
from utils import Profiler

_NO_PROFILE = Profiler()            # disabled: the timers below do nothing

_SRC = r"""
extern "C" __global__
void dtw_fill(const float* __restrict__ M, float* DP, unsigned char* DIR,
              const int* L1, const int* L2, const int T_max, const int L_max)
{
    const int b = blockIdx.x;
    const int T = L1[b], L = L2[b];
    const float* Mb = M + (long long)b * T_max * L_max;
    float* D = DP + (long long)b * (T_max + 1) * (L_max + 1);
    unsigned char* R = DIR + (long long)b * (T_max + 1) * (L_max + 1);
    const int W = L_max + 1;
    for (int s = 2; s <= T + L; ++s) {
        const int i_lo = max(1, s - L), i_hi = min(T, s - 1);
        for (int i = i_lo + threadIdx.x; i <= i_hi; i += blockDim.x) {
            const int j = s - i;
            const float diag = D[(i - 1) * W + (j - 1)];
            const float up   = D[(i - 1) * W + j];
            const float left = D[i * W + (j - 1)];
            float best = diag; unsigned char mv = 0;
            if (up > best)   { best = up;   mv = 1; }
            if (left > best) { best = left; mv = 2; }
            D[i * W + j] = best + Mb[(i - 1) * L_max + (j - 1)];
            R[i * W + j] = mv;
        }
        __syncthreads();   // the next anti-diagonal reads the values of this one
    }
}

extern "C" __global__
void dtw_fill_shared(const float* __restrict__ M, unsigned char* DIR, float* BEST,
                     const int* L1, const int* L2, const int T_max, const int L_max)
{
    extern __shared__ float buf[];                 // 3 x (T_max + 1) floats, private to this block
    const int b = blockIdx.x;
    const int T = L1[b], L = L2[b];
    const float* Mb = M + (long long)b * T_max * L_max;
    unsigned char* R = DIR + (long long)b * (T_max + 1) * (L_max + 1);
    const int W = L_max + 1;
    const int N = T_max + 1;
    const float NEG = __int_as_float(0xff800000);  // -infinity
    float* d0 = buf;                               // anti-diagonal s-2, indexed by the row i
    float* d1 = buf + N;                           // anti-diagonal s-1
    float* d2 = buf + 2 * N;                       // anti-diagonal s (written in this step)

    // anti-diagonal 0: cell (0, 0) = 0; anti-diagonal 1: cells (0, 1) and (1, 0) = -inf
    for (int i = threadIdx.x; i < N; i += blockDim.x) { d0[i] = NEG; d1[i] = NEG; }
    __syncthreads();
    if (threadIdx.x == 0) d0[0] = 0.0f;
    __syncthreads();

    for (int s = 2; s <= T + L; ++s) {
        const int i_lo = max(1, s - L), i_hi = min(T, s - 1);
        for (int i = i_lo + threadIdx.x; i <= i_hi; i += blockDim.x) {
            const int j = s - i;
            const float diag = d0[i - 1];          // cell (i-1, j-1), anti-diagonal s-2
            const float up   = d1[i - 1];          // cell (i-1, j),   anti-diagonal s-1
            const float left = d1[i];              // cell (i,   j-1), anti-diagonal s-1
            float best = diag; unsigned char mv = 0;
            if (up > best)   { best = up;   mv = 1; }
            if (left > best) { best = left; mv = 2; }
            d2[i] = best + Mb[(i - 1) * L_max + (j - 1)];
            R[i * W + j] = mv;
        }
        if (threadIdx.x == 0) {                    // the edge cells of anti-diagonal s
            d2[0] = NEG;                           // cell (0, s)
            if (s <= T) d2[s] = NEG;               // cell (s, 0)
        }
        __syncthreads();                           // anti-diagonal s is complete
        float* t = d0; d0 = d1; d1 = d2; d2 = t;   // rotate: the oldest buffer is written next
    }
    if (threadIdx.x == 0) BEST[b] = d1[T];         // cell (T, L) of the last anti-diagonal
}

extern "C" __global__
void dtw_traceback(const unsigned char* DIR, const int* L1, const int* L2,
                   const int T_max, const int L_max, const int P_max,
                   int* PATH, int* PLEN, const int B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const int W = L_max + 1;
    const unsigned char* R = DIR + (long long)b * (T_max + 1) * W;
    int* P = PATH + (long long)b * P_max * 2;
    int i = L1[b], j = L2[b], k = 0;
    while (i > 0 && j > 0) {
        P[2 * k] = i - 1; P[2 * k + 1] = j - 1; ++k;
        const unsigned char mv = R[i * W + j];
        if (mv == 0 || mv == 1) --i;
        if (mv == 0 || mv == 2) --j;
    }
    while (i > 0) { P[2 * k] = i - 1; P[2 * k + 1] = j; ++k; --i; }
    while (j > 0) { P[2 * k] = i; P[2 * k + 1] = j - 1; ++k; --j; }
    PLEN[b] = k;      // the path is stored from the end cell to (0, 0)
}

// Key for atomicMax: the float score mapped to an order-preserving unsigned int in the high 32 bits,
// and the inverted index in the low 32 bits, so that equal scores keep the lowest index (as argmax does).
__device__ __forceinline__ unsigned long long score_key(float f, unsigned int idx)
{
    unsigned int u = __float_as_uint(f);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 32) | (0xFFFFFFFFu - idx);
}

extern "C" __global__
void msa_best_template(const float* __restrict__ M, const int* PATH, const int* PLEN, unsigned long long* KEY,
                       const int T_max, const int L_max, const int P_max, const int B)
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)B * P_max) return;
    const int b = g / P_max, k = g % P_max;
    if (k >= PLEN[b]) return;
    const int i = PATH[2 * g], j = PATH[2 * g + 1];
    const float s = M[((long long)b * T_max + i) * L_max + j];
    atomicMax(&KEY[(long long)b * L_max + j], score_key(s, i));
}

extern "C" __global__
void msa_anchor(const float* __restrict__ M, const unsigned long long* KEY, const int* L2, int* BEST_T,
                unsigned long long* AKEY, const int T_max, const int L_max, const int B)
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)B * L_max) return;
    const int b = g / L_max, j = g % L_max;
    if (j >= L2[b]) return;
    const int t = (int)(0xFFFFFFFFu - (unsigned int)(KEY[g] & 0xFFFFFFFFull));
    BEST_T[g] = t;
    const float s = M[((long long)b * T_max + t) * L_max + j];
    atomicMax(&AKEY[(long long)b * T_max + t], score_key(s, j));
}
"""

_module = cp.RawModule(code=_SRC)
_fill = _module.get_function("dtw_fill")
_fill_shared = _module.get_function("dtw_fill_shared")
_trace = _module.get_function("dtw_traceback")
_best_template = _module.get_function("msa_best_template")
_anchor = _module.get_function("msa_anchor")
MAX_SHARED_BYTES = 48 * 1024        # the default limit of shared memory for one block


def _c(t):
    return cp.asarray(t)          # zero-copy view of a torch CUDA tensor


def _pad(M):
    """List of B CUDA tensors (T_i, L_i) -> (M_pad, L1, L2, T_max, L_max)."""
    dev = M[0].device
    L1 = torch.tensor([m.shape[0] for m in M], dtype=torch.int32, device=dev)
    L2 = torch.tensor([m.shape[1] for m in M], dtype=torch.int32, device=dev)
    T_max, L_max = max(m.shape[0] for m in M), max(m.shape[1] for m in M)
    M_pad = torch.zeros((len(M), T_max, L_max), dtype=torch.float32, device=dev)
    for i, m in enumerate(M):
        M_pad[i, :m.shape[0], :m.shape[1]] = m
    return M_pad, L1, L2, T_max, L_max


# With --profile, each function below fills sub-rows of its stage:
#   *_kernel: GPU execution of the kernel only (CUDA events)
#   dp_fill_prep: padding M and creating the buffers on the GPU (no CPU<->GPU transfer)
#   traceback_copy: GPU -> CPU transfer of scores and paths; traceback_lists: building Python lists on the CPU
#   msa_assembly_prep: buffers on the GPU; msa_assembly_copy: GPU -> CPU transfer of the column assignment;
#   msa_assembly_strings: per-sequence lists on the CPU (the text itself is built in ARIES)

def dtw_fill_cuda(M, threads=256, prof=_NO_PROFILE):
    """Version 1 fill. Returns the state dtw_traceback_cuda needs: (best, DIR, L1, L2)."""
    with prof.part('dp_fill_prep'):
        M_pad, L1, L2, T_max, L_max = _pad(M)
        B, dev = len(M), M_pad.device
        DP = torch.full((B, T_max + 1, L_max + 1), float("-inf"), dtype=torch.float32, device=dev)
        DP[:, 0, 0] = 0.0
        DIR = torch.zeros((B, T_max + 1, L_max + 1), dtype=torch.uint8, device=dev)
        args = (_c(M_pad), _c(DP), _c(DIR), _c(L1), _c(L2), cp.int32(T_max), cp.int32(L_max))
    with prof.gpu_part('dp_fill_kernel'):
        _fill((B,), (threads,), args)
    with prof.part('dp_fill_prep'):
        best = DP[torch.arange(B, device=dev), L1.long(), L2.long()]
    return best, DIR, L1, L2, M_pad


def dtw_fill_shared_cuda(M, threads=256, prof=_NO_PROFILE):
    """Version 2 fill (three anti-diagonals in shared memory). Same return as dtw_fill_cuda."""
    with prof.part('dp_fill_prep'):
        M_pad, L1, L2, T_max, L_max = _pad(M)
        B, dev = len(M), M_pad.device
        shared_bytes = 3 * (T_max + 1) * 4
        if shared_bytes > MAX_SHARED_BYTES:
            raise ValueError(f"kernel1: template length {T_max} needs {shared_bytes} bytes of shared memory "
                             f"(limit {MAX_SHARED_BYTES}); use --align-device kernel0 for this input")
        DIR = torch.zeros((B, T_max + 1, L_max + 1), dtype=torch.uint8, device=dev)
        BEST = torch.empty((B,), dtype=torch.float32, device=dev)
        args = (_c(M_pad), _c(DIR), _c(BEST), _c(L1), _c(L2), cp.int32(T_max), cp.int32(L_max))
    with prof.gpu_part('dp_fill_kernel'):
        _fill_shared((B,), (threads,), args, shared_mem=shared_bytes)
    return BEST, DIR, L1, L2, M_pad


def dtw_traceback_gpu(DIR, L1, L2, prof=_NO_PROFILE):
    """Runs dtw_traceback; the paths stay on the GPU: PATH (B, P_max, 2) from the end cell, PLEN (B,)."""
    dev = DIR.device
    B, T_max, L_max = DIR.shape[0], DIR.shape[1] - 1, DIR.shape[2] - 1
    P_max = T_max + L_max
    PATH = torch.empty((B, P_max, 2), dtype=torch.int32, device=dev)
    PLEN = torch.empty((B,), dtype=torch.int32, device=dev)
    tpb = 64
    args = (_c(DIR), _c(L1), _c(L2), cp.int32(T_max), cp.int32(L_max), cp.int32(P_max), _c(PATH), _c(PLEN), cp.int32(B))
    with prof.gpu_part('traceback_kernel'):
        _trace(((B + tpb - 1) // tpb,), (tpb,), args)
    return PATH, PLEN


def dtw_traceback_cuda(best, DIR, L1, L2, M_pad=None, prof=_NO_PROFILE):
    """Runs dtw_traceback; returns (best_scores, paths) like WavefrontDP.align_batch."""
    B = DIR.shape[0]
    PATH, PLEN = dtw_traceback_gpu(DIR, L1, L2, prof=prof)

    # One copy to the CPU for all sequences, then the list format that build_msa expects.
    with prof.part('traceback_copy'):
        best, PATH, PLEN = best.cpu(), PATH.cpu(), PLEN.cpu()
    with prof.part('traceback_lists'):
        best, PLEN = best.tolist(), PLEN.tolist()
        paths = [list(map(tuple, PATH[b, :PLEN[b]].flip(0).tolist())) for b in range(B)]
    return best, paths


def dtw_cuda(M, threads=256):
    """Version 1. M: list of B CUDA tensors (T_i, L_i), float32.

    Returns (best_scores, paths) in the same format as WavefrontDP.align_batch.
    """
    return dtw_traceback_cuda(*dtw_fill_cuda(M, threads=threads))


def dtw_cuda_shared(M, threads=256):
    """Version 2; same arguments and return as dtw_cuda. It keeps no full DP table."""
    return dtw_traceback_cuda(*dtw_fill_shared_cuda(M, threads=threads))


def assign_columns_cuda(M_pad, PATH, PLEN, L1, L2, prof=_NO_PROFILE):
    """Column assignment of MSA assembly on the GPU, from the scores and paths left there by DTW.

    Returns, for each sequence, (best_t, anchor_j): best_t[j] = template position of residue j,
    anchor_j[t] = anchor residue of template position t (only meaningful where some residue has t).
    """
    B, T_max, L_max = M_pad.shape
    P_max, dev, tpb = PATH.shape[1], M_pad.device, 256
    with prof.part('msa_assembly_prep'):
        KEY = torch.zeros((B, L_max), dtype=torch.int64, device=dev)       # read as unsigned 64-bit keys
        AKEY = torch.zeros((B, T_max), dtype=torch.int64, device=dev)
        BEST_T = torch.full((B, L_max), -1, dtype=torch.int32, device=dev)
        args1 = (_c(M_pad), _c(PATH), _c(PLEN), _c(KEY), cp.int32(T_max), cp.int32(L_max), cp.int32(P_max), cp.int32(B))
        args2 = (_c(M_pad), _c(KEY), _c(L2), _c(BEST_T), _c(AKEY), cp.int32(T_max), cp.int32(L_max), cp.int32(B))
    # argument conversion stays outside the CUDA-event timer, so *_kernel is GPU execution only
    with prof.gpu_part('msa_assembly_kernel'):
        _best_template(((B * P_max + tpb - 1) // tpb,), (tpb,), args1)
        _anchor(((B * L_max + tpb - 1) // tpb,), (tpb,), args2)
        ANCHOR = (0xFFFFFFFF - (AKEY & 0xFFFFFFFF)).to(torch.int32)
    with prof.part('msa_assembly_copy'):
        BEST_T, ANCHOR, L1, L2 = BEST_T.cpu(), ANCHOR.cpu(), L1.cpu(), L2.cpu()
    with prof.part('msa_assembly_strings'):
        L1, L2 = L1.tolist(), L2.tolist()
        return [(BEST_T[b, :L2[b]].tolist(), ANCHOR[b, :L1[b]].tolist()) for b in range(B)]


class CudaKernelDTW(DTW):
    """DTW whose DP fill and traceback run as CUDA kernels; similarity is unchanged.

    fill: dtw_fill_cuda (kernel0) or dtw_fill_shared_cuda (kernel1). align_batch keeps the paths and
    scores on the GPU and returns the column assignment of MSA assembly instead of paths.
    """
    def __init__(self, **kwargs):
        super(CudaKernelDTW, self).__init__(**kwargs)
        self.threads = kwargs.get('threads', 256)
        self.fill = kwargs.get('fill', dtw_fill_cuda)

    def align_batch(self, M):
        with self.profiler.section('dp_fill'):
            best, DIR, L1, L2, M_pad = self.fill(M, threads=self.threads, prof=self.profiler)
        with self.profiler.section('traceback'):
            PATH, PLEN = dtw_traceback_gpu(DIR, L1, L2, prof=self.profiler)
        with self.profiler.section('msa_assembly'):
            return best, assign_columns_cuda(M_pad, PATH, PLEN, L1, L2, prof=self.profiler)


KERNELS = {'kernel0': dtw_fill_cuda, 'kernel1': dtw_fill_shared_cuda}
