# REPORT.md — t1917 acceptance probe (MTP multiplier #3)

Branch: `acceptance-probe` (base `r5-correction-fusion @ 4b9285b`).
Wheel under test: venv-p3 = r4v wheel line + TMAX=3 addition (3 files:
`gdn_fused_step.hip`, `gdn_fused_step_tape.hip`, `fast.cpp`); everything else
byte-identical to the verified r4v anchor (sidecar md5 02a54359..., libmlx
swapped: r4v 8acf13... -> p3 460c803d...).

## Mission

acc/verifies today 1.36-1.45 vs the llama anchor 2.15 (3.15 tok/step target).
Two sub-levers, code-first, env-gated:
- (a) TMAX=3: admit the S=3 / K=2 spec-decode verify into the GDN fused +
  fused-tape ops (`MLX_ROCM_GDN_T3=1`; default off = the clean error of today,
  no OOB fall-through to the TMAX=4 kernel).
- (b) Draft-head quality constraints, NO retraining: quantization bit-width
  (`MTP6_DRAFT_HEAD=bf16full` = bf16 dequant of the model's 4-bit lm_head vs
  the 4-bit model head), head input feature (`MTP6_HEAD_H=postnorm` = the
  trunk's final RMSNorm applied in-bench before the MTP module).

## Results (anchor MTP_MAX_TOKENS=322, 3 repeats interleaved, medians)

(filled at measurement time; per-run logs in /home/andryo/p3-full/)

## Sub-lever (a): TMAX=3

(filled: S=3 tape vs S=3 legacy vs S=4 tape — acc/verifies, tok/s, GDN tape
call counts, refwd/cdeferred counts.)

## Sub-lever (b): draft-head probe

(filled: bf16full vs 4-bit model head at S=3 and S=4; postnorm vs prenorm.)

## Probe-only findings (no retraining possible; expected gain, STOP)

1. **int2 draft head was never live on the r4v-line wheels**: the r4v line's
   `fast.cpp` has NO `LMHeadInt2` op and no `lm_head_int2` Python binding
   (verified: `hasattr(mx.fast, "lm_head_int2") == False` in venv-r4v, and the
   r4v wheel's libmlx carries no `lm_head_int2` symbol). The `MTP6_DRAFT_HEAD=
   int2` bench arm therefore silently degraded to the model's 4-bit lm_head in
   every r4v-line round (the bench gate is `hasattr(...)`). Any prior "int2"
   number on this line is actually a bf16/4-bit-head number. The 2-bit point of
   the bit-width probe is thus UNMEASURABLE on this wheel line without porting
   the op + binding (separate lane); the measurable bit-width pair is 4-bit vs
   bf16 (dequant).
2. **Head depth/width**: the MTP module is 1 decoder layer (fc 2H->H + attn +
   MLP + norm). Deeper/wider heads require new trained weights — expected
   gain: bounded by the acceptance-rate ceiling of the 4-bit head (the
   bf16full result below is the upper bound of the bit-width axis); STOP.
3. **Intermediate-layer hidden feature**: the model exposes only the final
   pre-norm hidden via `return_prenorm`; a mid-layer tap needs model plumbing
   and the head was trained on the final-layer feature — probe-only; STOP.

## Env gates added (all default-off = zero behavior change)

- `MLX_ROCM_GDN_T3=1`: admits T=3 into `GdnFusedStep::use_fallback` and
  `GdnFusedStepTape::use_fallback` (both kernels gain a `tmax == 3 -> <3>`
  dispatch branch; T=1/2/4 dispatch and codegen untouched; s_beta/s_g are
  sized [4] so TMAX=3 is a strict prefix). Default (unset/0): T=3 degrades to
  the clear error of today.
- `MTP6_TAPE` (bench, mode 3 now, as in mode 4): 1/hybrid/r5 = the S=3 tape
  arm (sets MLX_ROCM_GDN_TAPE=1 + MLX_ROCM_GDN_T3=1 + MLX_ROCM_GDN_FUSE=1);
  0 = the legacy save/restore + re-forward path (unchanged).
- `MTP6_DRAFT_HEAD`: int2 | bf16 (model 4-bit QuantizedLinear direct; the
  default's effective head) | bf16full (bf16 dequant; no new weights).
- `MTP6_HEAD_H`: prenorm (default) | postnorm (trunk final RMSNorm in-bench).
