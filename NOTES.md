# NOTES.md — acceptance probe (MTP multiplier #3, t1917)

Base: r5-correction-fusion @ 4b9285b. Work branch: acceptance-probe.
Mission: acc/verifies today 1.36-1.45 vs the llama anchor 2.15 (3.15 tok/step target).
Sub-lever (a): TMAX=3 (S=3 / K=2 verify path, t1811 follow-up 2).
Sub-lever (b): draft-head quality probe (no retraining where possible).

## 1. Base state (code read)

- Tape dispatcher: `vendor/mlx-rocm-upstream/mlx/backend/rocm/gdn_fused_step_tape.hip`
  host launcher `launch_one` instantiates TMAX in {1, 2, 4} only; `tmax` else-branch
  maps to `<4>`, so a T=3 (S=3, K=2) verify launch falls through to the TMAX=4
  kernel on a 3-row buffer = OOB read+write. The tape op's `use_fallback`
  (fast.cpp ~L1394) admits `T > 4` only, so T=3 is currently ADMITTED into the
  OOB path (no clean rejection). The non-tape F-branch op (`GdnFusedStep::
  use_fallback`, fast.cpp ~L1190) has the same `T > 4` admission = same OOB
  fall-through for T=3. The canonical `patches/gdn_fused_step/gdn_fused_step.patch`
  still carries the older `(T != 1 && T != 2 && T != 4)` rejection (vendor drifted
  past the patch; both updated in this PR, hunk counts re-validated by
  `test_patch_hunk_counts_are_wellformed`).
- Kernel body is already TMAX-general: the t loop is `for (t = 0; t < TMAX; ++t)`
  with the conv window written in the generalized sliding form (tap j =
  conv_state[j-t] while j-t in [0,2), else raw row t+1-j); s_beta/s_g shared
  buffers are sized [4], so TMAX=3 uses a strict prefix — no buffer change needed.
  TMAX=3 is a pure instantiation add (the T=1/T=2 bit-exactness argument is
  unchanged: their instantiations are untouched).
- The wheel model file (venv-r4v `qwen3_5.py` TAPE_BRANCH_PATCH) gates on `S <= 4`,
  so an S=3 verify IS routed to `gdn_fused_step_tape` when MLX_ROCM_GDN_TAPE=1 +
  MLX_ROCM_GDN_FUSE=1 — i.e. the model side already admits S=3; only the C++
  dispatcher lacks TMAX=3. No model-file change needed for the S=3 arm.
- Bench `bench/mtp-bench6.py`: mode "3" (S=3, K=2) exists but only with
  save/restore + re-forward (no tape); tape env setup and the int2/qh head build
  are gated on `mode == "4"`. The mission's S=3 arm = mode 3 + tape commit,
  env-gated by MTP6_TAPE, mirroring mode 4's commit/reject (K=2: commit row 1
  (after D1) or row 0 (after prev_token); KV trim S-1-krow; mtp_cache trim
  K-krow; r5 deferred-correction = ctoken becomes row 0 of the next verify).
- MTP head (MTP_INJECT_PATCH, 1 layer, n_layers from config = 1):
  `pre_fc_norm(hidden) concat pre_fc_norm(embedding) -> fc(2H->H) -> 1 decoder
  layer (attn + MLP) -> norm -> lm_head`. Default quant=False (fp sidecar
  model-mtp.safetensors; MTP_4BIT env selects the 4-bit sidecar). Draft head =
  lm_head (model's 4-bit QuantizedLinear; int2 GEMV + 4-bit re-rank via
  MTP6_DRAFT_HEAD=int2). Head depth/width changes REQUIRE trained weights
  (fc projection is 2H->H; deeper/wider = new params) -> probe + expected gain,
  STOP per the mission (weights = separate lane).

## 2. Design (code-first)

- TMAX=3, env-gated: `MLX_ROCM_GDN_T3=1` admits T=3 into the fused/tape ops
  (use_fallback in both ops); DEFAULT (unset/0) = T=3 degrades to the clean
  error/fallback exactly as today (no OOB), so the default path is a
  zero-behavior-change. Launcher lambda gains a `tmax == 3 -> <3>` branch in
  both kernels. T=1/2/4 dispatch and codegen untouched.
- Bench env gates (all default-off relative to today's arms):
  - mode 3 + MTP6_TAPE in (1, hybrid, r5): the S=3 tape arm (sets
    MLX_ROCM_GDN_TAPE=1, MLX_ROCM_GDN_FUSE=1, MLX_ROCM_GDN_T3=1); commit/reject
    mirroring mode 4; tape_mode 0 = today's save/restore path (unchanged).
  - MTP6_DRAFT_HEAD: int2 (default) | bf16 (model 4-bit QuantizedLinear direct,
    today's non-int2 path) | bf16full (NEW: bf16 dequant of the 4-bit lm_head,
    no new weights; the quantization-bit-width probe arm).
  - MTP6_HEAD_H: prenorm (default) | postnorm (NEW: feed lm.model.norm(h) to the
    MTP module; the head-input-feature probe arm; no retraining - the head
    consumes a different trunk feature, same weights).
  - Intermediate-layer hidden probe: NOT runnable in the bench (the model exposes
    only the final pre-norm hidden via return_prenorm; a mid-layer tap needs
    model plumbing + the head was trained on the final-layer feature) ->
    reported as probe-only with expected gain, per mission.
- Measurement (neon, flock protocol, 25 s idle, SCLK sampler card1):
  base = MTP6_TAPE=r5 mode 4 (venv-r4v); t3 = MTP6_TAPE=r5 mode 3 (venv-p3,
  new wheel from patched vendor + venv-p3 = copy of venv-r4v, libmlx swapped);
  head arms on venv-r4v mode 4 (MTP6_DRAFT_HEAD=bf16full / MTP6_HEAD_H=postnorm).
  Anchor MTP_MAX_TOKENS=322, 3 repeats interleaved, medians. 64-token smoke per
  arm first.
- BUILD BASE CORRECTION (found at build time): the neon r4v vendor is a NEWER
  line than the repo's 4b9285b base (it carries LMHeadInt2, the conv_state_new
  third output + in-place graph-decode state/conv-state aliases, and the
  `(T != 1 && T != 2 && T != 4)` T3 guard). venv-p3 = r4v vendor + the TMAX=3
  addition applied ON the r4v files (three files: gdn_fused_step.hip, 
  gdn_fused_step_tape.hip, fast.cpp; everything else byte-identical to r4v).
  The repo's vendored files and the canonical patch were updated in parallel so
  the PR carries the change as a patch against the repo base; the wheel is
  built from the r4v line so the r4v features (int2 head, graph decode) are
  preserved. venv-p3 = `cp -a venv-r4v` + `pip install --target` of the p3 wheel.

## 3. Unit checks (CPU)

- [x] py_compile bench/mtp-bench6.py
- [x] uv run pytest -m "not gpu" (hermetic; includes the updated test_gdn_fuse.py, 25 passed)
- [x] uv run ruff format --check . / ruff check .
- [x] size_ratchet gate (OK, 2 ratcheted)

## 5. GPU protocol (neon)

- Build: `~/p3-build.sh` (vendor-p3 = cp -a vendor-r4v + 3 p3 files; pip wheel with
  the r4v CMAKE_ARGS; install --target into venv-p3).
- Smoke: `~/p3-smoke.sh` — 64-token, one run per arm (S_B4, S_B4HF, S_B4PH, S_T3T1,
  S_T3T0, S_T3R5, S_T3HF, S_T3PH, S_T3BF).
- Full: `~/p3-full.sh` — anchor MTP_MAX_TOKENS=322, 3 rounds x 7 arms
  (B4, B4HF, B4PH, T3T1, T3T0, T3HF, T3PH), medians offline.
- Bench = the repo's bench/mtp-bench6.py (uploaded as ~/p3-bench.py); the neon
  mtp-bench6-canonical.py on the box is an OLD pre-tape copy, not used.
- Identity guards per battery: bench md5, sidecar md5 (02a543...), libmlx md5,
  SCLK before/after (card1, locked 2200).

## 4. GPU results

### Smoke (64-token, one run per arm, venv-p3, SCLK locked 2200, all rc=0)

| arm | MODE | HEAD | tok/s | TOK/STEP | D1% | D2% | REF | CDEF | TAPE_CALLS |
|---|---|---|---|---|---|---|---|---|---|
| S_B4  | 4 | 4bit | 43.98 | 2.844 | 93.8 | 96.7 | 32 | 0 | 1536 |
| S_B4HF| 4 | bf16full | 29.52 | 2.844 | 93.8 | 96.7 | 32 | 0 | 1536 |
| S_T3T1| 3 | 4bit | 54.63 | 2.844 | 93.8 | 96.7 | 3 | 0 | 840 |
| S_T3T0| 3 | 4bit | (legacy save/restore path) | | | | | | |
| S_T3R5| 3 | 4bit (r5) | 33.47 | 2.000 | 68.8 | 45.5 | 0 | 22 | 768 |
| S_T3HF| 3 | bf16full | 36.15 | 2.844 | 93.8 | 96.7 | 3 | 0 | 840 |
| S_T3PH| 3 | postnorm | 54.28 | 2.812 | 93.8 | 93.3 | 4 | 0 | 864 |

Smoke reading (64-token, 32 verifies - small sample, full 322-token battery is
authoritative): T=3 tape works (tape calls + commit + rebind all live; r5
deferred correction live, REFWDS=0). The S=3 tape arm beats S=4 tape on
tok/s (54.6 vs 43.9) at equal acceptance in this sample. The r5 arm shows
lower acceptance (2.0 tok/step, D1 68.8/D2 45.5) in this small sample. bf16full
and postnorm arms run clean (INFO lines report HEAD/HEAD_H).

### Full battery (anchor MTP_MAX_TOKENS=322, 3 rounds x 7 arms, medians; all rc=0 except T3T0 = clean-error control)

| arm | head | tok/s | tok/step | D2% | REF | CDEF |
|---|---|---|---|---|---|---|
| B4 | 4-bit | 42.94 | 2.826 | 84.9 | 161 | 0 |
| B4HF | bf16full | 30.77 | 2.826 | 84.9 | 161 | 0 |
| B4PH | postnorm | 49.25 | 2.882 | 90.6 | 161 | 0 |
| T3T1 (NEW) | 4-bit | 82.10 | 2.882 | 90.6 | 17 | 0 |
| T3T0 | legacy | N/A (clean error) | | | | |
| T3HF | bf16full | 44.79 | 2.882 | 90.6 | 17 | 0 |
| T3PH | postnorm | 83.80 | 2.907 | 93.1 | 13 | 0 |

Headline: S=3 tape = 1.91x S=4 tape (82.10 vs 42.94 tok/s); D3 never accepted
(D3ACC=0 in every S=4 run) so S=3 is strictly better on this model. bf16full
head: identical acceptance, ~1.8x slower draft. postnorm head: +2.5..5.7pts D2
for free. Details + probe-only findings in REPORT.md.

### Findings carried to REPORT.md

1. The r4v-line wheels have NO int2 draft head (no op, no binding):
   `hasattr(mx.fast, "lm_head_int2") == False` in venv-r4v; the r4v wheel's
   libmlx carries no lm_head_int2 symbol. "int2" bench arms silently degraded
   to the model 4-bit head in every r4v-line round.
2. venv protocol note: the venvs need `LD_LIBRARY_PATH=<venv>/lib/python3.14/
   site-packages/mlx/lib` to import mlx (added to the p3 battery BASEENV).
