// Copyright © 2025 Apple Inc.

// K3(i) 2026-11-11: uniform small-M row-fused 4-bit QMV — ONE structure for
// M = 1..4 (decode), the proven wave-1 (e5ef5b7) pattern: LDS staging +
// broadcast reads, shared weight dequant, in-register RNE output, 1 barrier
// per k-slice (fewer than wave-1's 2 per sub-tile).
//
// The d1c7a913 default small-M ladder re-dequantizes the 4-bit weight tile
// M times per (col, k-tile): the N >= 16384 rows_msg default decodes it once
// per (row, col) cell — 16 decodes per (col, k-tile) at M = 4 (4 cols x 4
// rows, 4x redundant across rows); the per-row wide default (M = 1 fallback
// and the M = 5..8 band) is one block per row, so the same 16-col weight
// tile is read + decoded once per output row. Here the 128-bit weight tile
// is decoded ONCE per (col, k-tile) into the in-register wn[32] (the proven
// mr3 form on this toolchain — the rows_msg note's 32-element qnan defect
// was position-specific to its 16-accumulator cell loop; the mr3 / wave-1
// kqfg form compiles clean) and is broadcast-MAC'd against ALL M rows.
//
// Morphology: grid (ceil(N/16), ceil(M/4)); block (32, 16) = 512 threads.
// Each thread owns ONE output column and up to 4 row slots; k is fully
// lane-strided (32-value tiles at k = it*1024 + lane*32, ascending).
//
// x (M x K; L2-resident) is staged into LDS once per (slice, row) by the
// 512 cooperating threads (contiguous 128 B per warp-row, 2 B / 256 B 4 B)
// and is broadcast-read by the 16 col threads — the 16x per-col x re-read
// becomes an LDS broadcast. The staging is double-buffered: fetch(it+1)
// into the opposite parity overlaps compute(it) (the wave-1 overlap), with
// ONE __syncthreads per slice — fewer than wave-1's 2 per sub-tile (there
// is no raw->dequanted LDS stage here; the dequant is in-register) and no
// in-loop barrier (the old MR staged kernel's barrier is gone). All threads
// reach every barrier (the col >= N guard wraps the compute, not the fetch
// or the barrier; the nrows <= 0 return is block-uniform).
//
// LDS: sx[2][4][1024] T = 16 KB (2 B) / 32 KB (4 B) for both parities —
// within the 64 KB budget at 512 threads.
//
// Register estimate (RDNA4 / gfx1201, 256 VGPR per thread): acc[4] + wn[32]
// + xv4[4] (transient) + wv[4] (transient) + loop state (~10) ~= 65 VGPR
// (pointers in the scalar file) — full occupancy headroom at 16 waves / CU
// per 512-thread block; vs rows_msg's acc[16] + xn[32] ~= 100 VGPR per
// thread with 4x redundant decode.
//
// Row-exact contract: the per-row math transcribes the one-row
// qmv_wide_kernel / qmv_wide_mr3_kernel traversal verbatim — k tiles in
// ascending order (tile i at k = it*1024 + lane*32, 32 consecutive values
// per tile, the same lane-strided partition as the per-row / mr3 /
// rows_msg kernels), la q-major / j-minor FMA order, per-value lx FADD
// guarded by has_bias in the same position (the F3 audit fix — the per-row
// kernel never accumulates lx when !has_bias, so no 0.0f * (finite lx
// chain) is ever formed), the same int->float conversions, the same
// `acc += s*la + b*lx` update, the same 5-stage __shfl_down reduction, the
// in-register RNE `static_cast<T>` (no f32 temp, no cast launch). The LDS-
// staged x values are bit-copied T (bf16 / fp16 / f32) with the same
// static_cast<float> as the direct global reads, so output row r ==
// quantized_matmul(x[r:r+1]) bit for bit — lossless gate: maxdiff 0.0 /
// 0 flips vs the d1c7a913 and 02c4a6ca per-row math.
//
// Requirements (dispatch gate): 4-bit affine (q4), transposed non-batched,
// M = 1..4, K % 32 == 0 (the wide gate) AND K % 1024 == 0 (slice
// alignment — K = 4096 / 8192 / 12288 all qualify, every shape in the
// M = 1..4 shape table), GROUP_SIZE in {32, 64, 128} (32 divides 1024 so a
// 32-value tile never straddles a group boundary — same as the per-row /
// mr3 / rows_msg kernels), bf16 / fp16 / fp32 x, wave32.
//
// Dispatch (qmm.hip, inside the wave32 wide_ok branch, AFTER the F3
// M4ROWEXACT / M4PREF opt-ins which run verbatim first, BEFORE the
// N >= 16384 rows band and the N < 16384 mr3 / mr2 / per-row ladder):
// M = 1..4, 4-bit, K % 1024 == 0, DEFAULT ON (opt out with
// MLX_QMV_NO_RF=1 — the full d1c7a913 ladder then runs verbatim: per-row
// wide / mr2 / mr3 / rows_msg / hoist, every existing env gate preserved).
namespace mlx::core::rocm {

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ void __launch_bounds__(512) qmv_rf_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    T* __restrict__ out,
    int M,
    int N,
    int K,
    bool has_bias) {
  constexpr int MAXM = 4;
  constexpr int MCOLS = 16;
  constexpr int SLICE = 1024;  // values per row per slice (32 lanes x 32)
  __shared__ T sx[2][MAXM][SLICE];

  const int lane = threadIdx.x;
  const int col = blockIdx.x * MCOLS + threadIdx.y;
  const int m0 = blockIdx.y * MAXM;
  int nrows = M - m0;
  if (nrows > MAXM)
    nrows = MAXM;
  // Block-uniform early return BEFORE any read (the MR12 defect class);
  // the col >= N guard wraps the per-thread compute so every surviving
  // thread reaches every barrier.
  if (nrows <= 0)
    return;
  const int nsl = K / SLICE;  // dispatch requires K % SLICE == 0
  const int ng = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int row_bytes = K / 2;  // 4-bit only
  const uint8_t* wr = w + static_cast<size_t>(col) * row_bytes;
  const ScaleT* scr = scales + static_cast<size_t>(col) * ng;
  const ScaleT* bcr =
      has_bias ? (biases + static_cast<size_t>(col) * ng) : nullptr;

  // x slice fetch: 512 threads, nrows x 1024 values; thread tid =
  // lane + 32*threadIdx.y takes value 2*tid and 2*tid+1 of each row —
  // contiguous 128 B (2 B) / 256 B (4 B) per warp-row.
#define FETCH_SLICE(SL, PAR)                                          \
  {                                                                   \
    const int fb = (SL) * SLICE;                                      \
    const int fv = 2 * (threadIdx.x + 32 * threadIdx.y);              \
    for (int m = 0; m < nrows; ++m) {                                 \
      const T* xp = &x[static_cast<size_t>(m0 + m) * K + fb];         \
      sx[(PAR)][m][fv] = xp[fv];                                      \
      sx[(PAR)][m][fv + 1] = xp[fv + 1];                              \
    }                                                                 \
  }

  float acc[MAXM];
#pragma unroll
  for (int m = 0; m < MAXM; ++m)
    acc[m] = 0.0f;

  FETCH_SLICE(0, 0);
  __syncthreads();
  for (int it = 0; it < nsl; ++it) {
    const int par = it & 1;
    if (it + 1 < nsl)
      // Opposite parity — cannot clobber the slice being computed.
      FETCH_SLICE(it + 1, (it + 1) & 1);
    if (col < N) {
      const int k = it * SLICE + lane * 32;
      const int g = k / GROUP_SIZE;
      const float s = load_scale_value<ScaleT, GROUP_SIZE, true>(scr[g]);
      const float b =
          has_bias ? static_cast<float>(bcr[g]) : 0.0f;
      const uint4 wv =
          *reinterpret_cast<const uint4*>(&wr[(k * 4) / 8]);
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
      // WEIGHT REUSE (K3(i)): the tile is decoded ONCE per (col, k-tile)
      // and broadcast-MAC'd against ALL M rows (the rows_msg 16-cell
      // re-dequant and the per-row M-block re-dequant are gone).
      float wn[32];
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          wn[q * 8 + j] = static_cast<float>((wp[q] >> (4 * j)) & 0xF);
      // x from LDS: 16 B vector loads (16 / sizeof(T) values each); the
      // values and the q-major / j-minor order are identical to the
      // per-row / mr3 direct reads, so the per-row FMA order is unchanged.
      constexpr int VEC = 16 / (int)sizeof(T);  // 8 (2 B) / 4 (4 B)
      constexpr int NU4 = 32 / VEC;            // 4 (2 B) / 8 (4 B)
#pragma unroll
      for (int m = 0; m < MAXM; ++m) {
        if (m >= nrows)
          break;
        uint4 xv4[8];
#pragma unroll
        for (int r = 0; r < NU4; ++r)
          xv4[r] = *reinterpret_cast<const uint4*>(
              &sx[par][m][lane * 32 + r * VEC]);
        const T* px = reinterpret_cast<const T*>(xv4);
        // Per-row math (EXACT qmv_wide / mr3 order): la FMA then
        // per-value lx FADD (guarded by has_bias, same position),
        // acc += s*la + b*lx.
        float la = 0.0f, lx = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int kb = k + q * 8;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            float xv = static_cast<float>(px[q * 8 + j]);
            la += xv * wn[q * 8 + j];
            if (has_bias)
              lx += xv;
          }
        }
        acc[m] += s * la + b * lx;
      }
    }
    __syncthreads();  // all threads done reading sx[par] before the next
                      // iteration's prefetch overwrites it
  }
  for (int m = 0; m < MAXM; ++m) {
    if (m >= nrows || col >= N)
      continue;
    float a = acc[m];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      a += __shfl_down(a, o);
    if (lane == 0)
      out[static_cast<size_t>(m0 + m) * N + col] = static_cast<T>(a);
  }
#undef FETCH_SLICE
}

} // namespace mlx::core::rocm
