"""Prototype CUDA DTW for ARIES (same result as pairwise.DTW with batch=True).

Two kernels, compiled at run time with CuPy (NVRTC, no C++ compiler needed):
  dtw_fill:      one thread block for each sequence; the threads share the cells of one
                 anti-diagonal; __syncthreads() after each anti-diagonal. It writes DP and
                 the step direction of each cell (0 diagonal, 1 up, 2 left).
  dtw_traceback: one thread for each sequence; it follows the saved directions.

The tie order is the same as torch.argmax in DTW.backtrack: diagonal, then up, then left.
Selected with --align-device kernel0.
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
_trace = _module.get_function("dtw_traceback")


def _c(t):
    return cp.asarray(t)          # zero-copy view of a torch CUDA tensor


def dtw_fill_cuda(M, threads=256):
    """M: list of B CUDA tensors (T_i, L_i). Pads them and runs dtw_fill.

    Returns the state dtw_traceback_cuda needs: (DP, DIR, L1, L2).
    """
    dev = M[0].device
    B = len(M)
    L1 = torch.tensor([m.shape[0] for m in M], dtype=torch.int32, device=dev)
    L2 = torch.tensor([m.shape[1] for m in M], dtype=torch.int32, device=dev)
    T_max, L_max = max(m.shape[0] for m in M), max(m.shape[1] for m in M)
    M_pad = torch.zeros((B, T_max, L_max), dtype=torch.float32, device=dev)
    for i, m in enumerate(M):
        M_pad[i, :m.shape[0], :m.shape[1]] = m
    DP = torch.full((B, T_max + 1, L_max + 1), float("-inf"), dtype=torch.float32, device=dev)
    DP[:, 0, 0] = 0.0
    DIR = torch.zeros((B, T_max + 1, L_max + 1), dtype=torch.uint8, device=dev)
    _fill((B,), (threads,), (_c(M_pad), _c(DP), _c(DIR), _c(L1), _c(L2), cp.int32(T_max), cp.int32(L_max)))
    return DP, DIR, L1, L2


def dtw_traceback_cuda(DP, DIR, L1, L2):
    """Runs dtw_traceback; returns (best_scores, paths) like WavefrontDP.align_batch."""
    dev = DP.device
    B, T_max, L_max = DP.shape[0], DP.shape[1] - 1, DP.shape[2] - 1
    P_max = T_max + L_max
    PATH = torch.empty((B, P_max, 2), dtype=torch.int32, device=dev)
    PLEN = torch.empty((B,), dtype=torch.int32, device=dev)
    tpb = 64
    _trace(((B + tpb - 1) // tpb,), (tpb,),
           (_c(DIR), _c(L1), _c(L2), cp.int32(T_max), cp.int32(L_max), cp.int32(P_max), _c(PATH), _c(PLEN), cp.int32(B)))

    # One copy to the CPU for all sequences, then the list format that build_msa expects.
    idx = torch.arange(B, device=dev)
    best = DP[idx, L1.long(), L2.long()].cpu().tolist()
    PATH, PLEN = PATH.cpu(), PLEN.cpu().tolist()
    paths = [list(map(tuple, PATH[b, :PLEN[b]].flip(0).tolist())) for b in range(B)]
    return best, paths


def dtw_cuda(M, threads=256, return_dp=False):
    """M: list of B CUDA tensors (T_i, L_i), float32.

    Returns (best_scores, paths) in the same format as WavefrontDP.align_batch.
    With return_dp=True, also returns the DP tensor (B, T_max + 1, L_max + 1).
    """
    DP, DIR, L1, L2 = dtw_fill_cuda(M, threads=threads)
    best, paths = dtw_traceback_cuda(DP, DIR, L1, L2)
    if return_dp:
        return best, paths, DP
    return best, paths


class CudaKernelDTW(DTW):
    """DTW whose DP fill and traceback run as the CUDA kernels above; similarity is unchanged."""
    def __init__(self, **kwargs):
        super(CudaKernelDTW, self).__init__(**kwargs)
        self.threads = kwargs.get('threads', 256)

    def align_batch(self, M):
        with self.profiler.section('dp_fill'):
            state = dtw_fill_cuda(M, threads=self.threads)
        with self.profiler.section('traceback'):
            return dtw_traceback_cuda(*state)
