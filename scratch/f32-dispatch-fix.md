# t1251 f32-dispatch fix (guest vendor-1c, wheel)

## Discovery (probe, new wheel)
`QuantizedMatmul::eval` (python/src + mlx/ops.cpp) does
`astype(x, promote_types(x.dtype(), scales.dtype()))` → **f32** when scales are
f32 (standard mlx_lm 4-bit layout: scales/biases f32, x bf16).
The wide_ok gate (and both MR dispatches) required `x.dtype() == bfloat16 || float16`,
so per-row-wide / old-MR / mr2 were **dead code** in real inference. M=4 large-N
lm_head ran qmv_tiled (f32) at 6.27ms. MLX_T1251_DBG trace confirmed `dt=other`
(f32) for M=4 N=248320 K=4096 bits=4 gs=64.

## Fix (qmm.hip)
1. `wide_ok`: dtype clause extended to include `float32`.
2. f32 launch branches (TT=float, ST=float) added to all three dispatches:
   - `qmv_wide_mr2_kernel` (mr2)
   - `qmv_wide_kernel` (per-row wide)
   - `qmv_wide_mr_kernel` (old MR, 8/4-bit)
Kernels are generic templates (no bf16-specific intrinsics — verified by grep);
bf16 lab results unaffected. f32 timing validated in-wheel after rebuild.

## Env knobs
- `MLX_T1251_DBG=1` — one-time trace of M/N/K/bits/gs/dt/gates before `if (wide_ok)`.
- `MLX_QMV_MR2_MIN_N` — mr2 N threshold (default 16384; 0 → 16384).
- `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD` — force QMV for M < threshold.

## G0 context (for report)
- G0 AR baseline: 80.96 tok/s (F32 wheel d1acf5d1, no MR).
- K=3 baseline: 34.68 tok/s; G3 target 37-38; G4 (4-bit drafter) target 39-40.
