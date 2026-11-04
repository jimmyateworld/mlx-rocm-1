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

Wheel under test: venv-p3 (libmlx md5 460c803d...). Sidecar md5 02a54359...
unchanged. SCLK locked 2200 MHz (verified before/after). All arms rc=0 except
T3T0 (the clean-error control, see below). tok/s = median TOKS_S; D1/D2 =
draft accept %; tok/step = 1 + (D1+D2+D3)/verifies.

| arm | head | head_h | tok/s | tok/step | D1% | D2% | REF | CDEF | tape_calls |
|---|---|---|---|---|---|---|---|---|---|
| B4 (S=4 tape, existing) | 4-bit | pre | 42.94 | 2.826 | 98.8 | 84.9 | 161 | 0 | 7728 |
| B4HF (S=4 tape, bf16full) | bf16full | pre | 30.77 | 2.826 | 98.8 | 84.9 | 161 | 0 | 7728 |
| B4PH (S=4 tape, postnorm) | 4-bit | post | 49.25 | 2.882 | 98.8 | 90.6 | 161 | 0 | 7728 |
| T3T1 (S=3 tape, NEW) | 4-bit | pre | **82.10** | **2.882** | 98.8 | **90.6** | 17 | 0 | 4272 |
| T3T0 (S=3 legacy) | - | - | N/A | N/A | - | - | - | - | - |
| T3HF (S=3 tape, bf16full) | bf16full | pre | 44.79 | 2.882 | 98.8 | 90.6 | 17 | 0 | 4272 |
| T3PH (S=3 tape, postnorm) | 4-bit | post | **83.80** | **2.907** | 98.8 | **93.1** | 13 | 0 | 4176 |

All arms: 616 (S=4) / 481 (S=3) tokens, 161 verifies, 3 interleaved rounds,
medians taken over the 3 runs. T3T1/T3HF/T3PH all run the NEW T=3 fused-tape
path (GDN_TAPE_CALLS 4272 vs 7728 for S=4 = 3-row vs 4-row admits).

### Sub-lever (a): TMAX=3

- S=3 tape = 82.10 tok/s vs S=4 tape = 42.94 tok/s = **1.91x faster**,
  tok/step 2.882 vs 2.826 (+2.0%), D2 90.6 vs 84.9.
- The 3rd draft (D3) is **never accepted** (D3ACC=0 in every S=4 run):
  the 4th verify row costs with zero acceptance, so S=3 (K=2) is strictly
  better on this model.
- T3T0 (legacy S=3 without the gate) is the clean error of today, verbatim:
  `gdn_fused_step: fallback disabled for this shape ... use the model default
  path (MLX_ROCM_GDN_FUSE=0) or T<=4 with MLX_ROCM_GDN_FUSE=1`. Note the
  r4v-line wheel's model file (qwen3_5.py) hard-calls `mx.fast.gdn_fused_step`
  unconditionally, so a GDN_FUSE=0 legacy S=3 arm is infeasible on this line:
  S=3 verify exists only through the TMAX=3 admission (with the gate) or the
  tape path. Default-off behavior confirmed: no OOB, clean error.

### Sub-lever (b): draft-head probe

- Bit-width: bf16full vs 4-bit model head -> **acceptance identical** (D2 90.6
  at S=3 and 84.9 at S=4 for both), but the bf16 draft is ~1.8x slower
  (44.79 vs 82.10 at S=3; 30.77 vs 42.94 at S=4). 4-bit head quantization is
  costless in draft quality on this 9B model; keep 4-bit.
- Head input feature: postnorm vs prenorm -> D2 +2.5pts at S=3 (93.1 vs 90.6)
  and +5.7pts at S=4 (90.6 vs 84.9); tok/step 2.907 vs 2.882; tok/s 83.80 vs
  82.10 (+2.1%). postnorm is a free gain, no retraining.
- r5 deferred correction (64-token smoke only): 2.00 tok/step vs 2.844 tape in
  a 32-verify sample (CDEFERRED=22, REFWDS=0); small sample, not part of the
  full battery; flag for a dedicated round.

## Probe-only findings (no retraining possible; expected gain, STOP)

1. **int2 draft head was never live on the r4v-line wheels**: the r4v line's
   `fast.cpp` has NO `LMHeadInt2` op and no `lm_head_int2` Python binding
   (verified: `hasattr(mx.fast, "lm_head_int2") == False` in venv-r4v, and the
   r4v wheel's libmlx carries no `lm_head_int2` symbol). The `MTP6_DRAFT_HEAD=
   int2` bench arm therefore silently degraded to the model's 4-bit lm_head in
   every r4v-line round (the bench gate is `hasattr(...)`). Any prior "int2"
   number on this line is actually a 4-bit/bf16-head number. The 2-bit point of
   the bit-width probe is thus UNMEASURABLE on this wheel line without porting
   the op + binding (separate lane); the measurable bit-width pair is 4-bit vs
   bf16 (dequant).
2. **Head depth/width**: the MTP module is 1 decoder layer (fc 2H->H + attn +
   MLP + norm). Deeper/wider heads require new trained weights — expected
   gain: bounded by the acceptance-rate ceiling of the 4-bit head (the
   bf16full result above is the upper bound of the bit-width axis); STOP.
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
