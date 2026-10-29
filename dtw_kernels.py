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

The calculation of each cell (max of diagonal, up, left; then + M) and the tie order (diagonal, then up,
then left) are the same in both versions and as torch.argmax in DTW.backtrack.
"""
import cupy as cp
import torch

from pairwise import DTW

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
"""

_module = cp.RawModule(code=_SRC)
_fill = _module.get_function("dtw_fill")
_fill_shared = _module.get_function("dtw_fill_shared")
_trace = _module.get_function("dtw_traceback")
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


def dtw_fill_cuda(M, threads=256):
    """Version 1 fill. Returns the state dtw_traceback_cuda needs: (best, DIR, L1, L2)."""
    M_pad, L1, L2, T_max, L_max = _pad(M)
    B, dev = len(M), M_pad.device
    DP = torch.full((B, T_max + 1, L_max + 1), float("-inf"), dtype=torch.float32, device=dev)
    DP[:, 0, 0] = 0.0
    DIR = torch.zeros((B, T_max + 1, L_max + 1), dtype=torch.uint8, device=dev)
    _fill((B,), (threads,), (_c(M_pad), _c(DP), _c(DIR), _c(L1), _c(L2), cp.int32(T_max), cp.int32(L_max)))
    best = DP[torch.arange(B, device=dev), L1.long(), L2.long()]
    return best, DIR, L1, L2


def dtw_fill_shared_cuda(M, threads=256):
    """Version 2 fill (three anti-diagonals in shared memory). Same return as dtw_fill_cuda."""
    M_pad, L1, L2, T_max, L_max = _pad(M)
    B, dev = len(M), M_pad.device
    shared_bytes = 3 * (T_max + 1) * 4
    if shared_bytes > MAX_SHARED_BYTES:
        raise ValueError(f"kernel1: template length {T_max} needs {shared_bytes} bytes of shared memory "
                         f"(limit {MAX_SHARED_BYTES}); use --align-device kernel0 for this input")
    DIR = torch.zeros((B, T_max + 1, L_max + 1), dtype=torch.uint8, device=dev)
    BEST = torch.empty((B,), dtype=torch.float32, device=dev)
    _fill_shared((B,), (threads,), (_c(M_pad), _c(DIR), _c(BEST), _c(L1), _c(L2), cp.int32(T_max), cp.int32(L_max)),
                 shared_mem=shared_bytes)
    return BEST, DIR, L1, L2


def dtw_traceback_cuda(best, DIR, L1, L2):
    """Runs dtw_traceback; returns (best_scores, paths) like WavefrontDP.align_batch."""
    dev = DIR.device
    B, T_max, L_max = DIR.shape[0], DIR.shape[1] - 1, DIR.shape[2] - 1
    P_max = T_max + L_max
    PATH = torch.empty((B, P_max, 2), dtype=torch.int32, device=dev)
    PLEN = torch.empty((B,), dtype=torch.int32, device=dev)
    tpb = 64
    _trace(((B + tpb - 1) // tpb,), (tpb,),
           (_c(DIR), _c(L1), _c(L2), cp.int32(T_max), cp.int32(L_max), cp.int32(P_max), _c(PATH), _c(PLEN), cp.int32(B)))

    # One copy to the CPU for all sequences, then the list format that build_msa expects.
    best = best.cpu().tolist()
    PATH, PLEN = PATH.cpu(), PLEN.cpu().tolist()
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


class CudaKernelDTW(DTW):
    """DTW whose DP fill and traceback run as CUDA kernels; similarity is unchanged.

    fill: dtw_fill_cuda (kernel0) or dtw_fill_shared_cuda (kernel1).
    """
    def __init__(self, **kwargs):
        super(CudaKernelDTW, self).__init__(**kwargs)
        self.threads = kwargs.get('threads', 256)
        self.fill = kwargs.get('fill', dtw_fill_cuda)

    def align_batch(self, M):
        with self.profiler.section('dp_fill'):
            state = self.fill(M, threads=self.threads)
        with self.profiler.section('traceback'):
            return dtw_traceback_cuda(*state)


KERNELS = {'kernel0': dtw_fill_cuda, 'kernel1': dtw_fill_shared_cuda}
