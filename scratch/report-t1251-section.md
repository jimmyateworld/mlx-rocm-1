
## t1251 QMV verify fixes — f32-dispatch wheel (K=3 MTP drafter path)

**Verdict: lossless, net-positive, sub-target. K=3 MTP: 34.68 → 35.92 tok/s (+3.6%; 4-bit drafter 37.48, +4.3% over F32 drafter). The expected 37–40 tok/s was NOT reached because real inference promotes x to f32 (scales are f32) and the new M-row MR kernel (mr2) is 3.7× slower with f32 x than with bf16 x on gfx1201; the wheel therefore ships per-row wide-QMV for f32 by default and mr2-f32 is opt-in. The wheel's kernels are byte-identical in token output to the previous (d1acf5d1) wheel.**

### The critical discovery: x is f32 in the QMM kernel, so all wide/MR kernels were dead code

`QuantizedMatmul` does `astype(x, promote_types(x.dtype(), scales.dtype()))`. The standard mlx_lm 4-bit layout (bf16 x, **f32** scales/biases) promotes x to **f32** before the kernel runs. The wide-QMV gate required `x.dtype() == bfloat16 || float16`, and the old MR / per-row-wide / new mr2 launches only branched on bf16/f16 — so in real inference (and in every probe using the op) the M≤8 wide path never ran; M=4 large-N went to the f32 tiled kernel. An in-wheel trace (`MLX_T1251_DBG`) confirmed `dt=f32` for the K=3 verify shapes. Consequence: the entire bf16-timed lab result set (mr2 2.59ms at N=248320) was not the number the wheel would see.

### Fix set (one wheel, vendor `vendor-1c` = `75ca881` + grafts)

1. **f32 in the wide gate + all three dispatches** (`qmm.hip`): `wide_ok` now admits `float32`; per-row wide, old MR, and mr2 each gained an `else if (x.dtype() == float32)` branch (`TT=float, ST=float`).
2. **mr2 LDS sizing dtype-aware**: x-staging is `T`-sized; f32 needs 4 bytes/element (was hardcoded 2 → under-sized LDS on f32).
3. **mr2 inner x-reads dtype-generic**: 32 values per lane = `NU4 = 32/VEC` uint4 loads at offset `q*VEC` (VEC = 16/sizeof(T)); the original read 4×16B assuming 2-byte T (scrambled x for f32).
4. **Old MR (2-byte kernel) excluded for f32** (its LDS gate was passed spuriously at 32768 bytes).
5. **mr2-f32 gated OFF by default**: measured 9.68ms vs per-row 4.03ms at N=248320 K=4096 M=4 (M=1 f32 = 2.66ms ≈ bf16 M=4, i.e. the per-row m-loop is the cost; ~2.3ms per extra row). Opt-in via `MLX_QMV_MR2_F32=1`.
6. N-gate: mr2 only at N ≥ 16384 (`MLX_QMV_MR2_MIN_N`, default 16384) — per-row wins below that even for bf16.

### 4-bit MTP drafter re-quant (FIX 3)

Re-quantized the F32 MTP sidecar to 4-bit affine gs=64: `model-mtp-4bit.safetensors` (145 MB, 31 keys: 8 linears × {weight uint32, scales f32, biases f32}), max abs dequant error 0.0259. Loaded via the existing `MTP_4BIT=1` model-side branch (the `MTP_INJECT_PATCH` in the `qwen3_5.py` uv-cache copy, unchanged by this task). Sidecar glob collision handled by parking the F32 sidecar during the 4-bit run (alphabetical merge would otherwise let F32 win).

### Gates (GPU host, RX 9070 XT / gfx1201; env `LD_PRELOAD=libhsa-runtime64`, `HCC_AMDGPU_TARGET=gfx1201`, `MLX_ROCM_GDN_FUSE=1`; bench `mtp-bench6.py`, greedy, MTP_MAX_TOKENS=1024)

| Gate | Result |
|---|---|
| G0 AR baseline (old wheel) | 80.96 tok/s |
| G2 lossless | **PASS** — K=3 stream (1678 tokens) byte-identical old vs new wheel (md5 `7b8c8c6d…` both); new-wheel AR reproducible (md5 `b1b35eca…`, 81.05 tok/s). Caveat: K=3 diverges from AR at token 33 (ar=7752, k3=12103) — **pre-existing: the old wheel produces the identical K=3 stream**, so it is a property of the bench's batched-verify/GDN trim-restore protocol, not of this wheel's kernels. |
| G3 K=3 speed | 35.92 tok/s (baseline 34.68, **+3.6%**; target 37–38 not met) |
| G4 4-bit drafter | 37.48 tok/s (D1ACC 76.2% / D2 60.0% / D3 52.1%, TOK_PER_STEP 2.457; first-20 tokens identical to the F32 drafter; target 39–40 not met) |
| G5 shape probe (new wheel, hot) | lm_head M4 4.03ms (old wheel ~6.3ms), gate_up 0.12 (vs ~0.30), fa_q 0.09 (vs ~0.20), k_v 0.019, down 0.14 (vs ~0.29); `NO_WIDE` K=3 drops to 22.69/23.24 tok/s, i.e. the f32 wide path is what carries the gain |

Per-pair attribution: the f32 wide path saves ~2.2ms on lm_head (N=248320) and ~0.1–0.2ms per other 4-bit QMM — less than the bf16-timed lab suggested, because f32 staging/reads make the MR kernel lose to per-row.

### Artifacts

- `patches/t1251-qmm.hip.patch` — the `qmm.hip` delta vs `75ca881` (f32 gate + dispatches, mr2 branch + N-gate, `MLX_T1251_DBG` trace, old-MR f32 exclusion).
- `patches/qmv_wide_mr_kernel.hip.new` — the graft file (old MR + mr2 kernels; f32-generic x reads).
- Env knobs: `MLX_QMV_MR2_MIN_N` (default 16384), `MLX_QMV_MR2_F32=1` (opt-in mr2 for f32), `MLX_QMV_NO_MR=1`, `MLX_QMV_NO_WIDE=1`, `MLX_T1251_DBG=1` (one-shot dispatch trace).
- Sidecars: `model-mtp.safetensors` (F32, parked during 4-bit runs), `model-mtp-4bit.safetensors` (this task's 4-bit re-quant).
