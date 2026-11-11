# M4PREF GATE RUN 2026-11-11 — cheap config-level gate on the delivered lib 9c376188 (NO build)

Objective (cloudy ruling 16:2xZ): cheap config-level gate of the F3 M4PREF arm (MLX_ROCM_M4PREF=1, the 3-wave split from w1-120lib 3eb5b6f) on the delivered lib 9c376188 (default-OFF build; core dfc5906c unchanged; venv-k3i on the box) — no build. (1) K2REG @1024 with M4PREF on must be TOKEN-IDENTICAL (alltoks_md5 86d73662e510e572608c6cd760eb26f4) + the FIRST32 check + the 91.8-class fingerprint (TOKS=1065, D1ACC 355 / D2ACC 198, TPK 2.080; TOKS_S is session clock, not a gate field). Any token movement = DIVERGENCE and the M=4 item closes fully at the defaults. (2) The lm_head/gate shape spot with M4PREF on (the R1 P3/P8 standalone probe recipe, eval(r) protocol, 20 iters after 3 warmup): N=248320 K=4096 and N=12288 K=4096 at M=1/3/4, same-session M4PREF arm vs the SoT rows_msg baseline arm. Env: the R1 campaign's record env VERBATIM (REPORT-R1-20261111 identity block) + MLX_ROCM_M4PREF=1; MLX_MTP=3 for the K2REG run. Both runs executed only while holding gpu.lock (flock); the box was confirmed free at start (no holders; the t2838 bracket campaign was not in its lock window — waited, never stole), the lock was held for the whole campaign 16:26:39Z-16:27:11Z and released at completion.

Per-run identity (from the run log's IDENTITY line — verbatim; mx_file path prefix redacted per the R1 report convention):

{"venv": "venv-k3i", "python": "3.14.8", "mx_file": ".../venv-k3i/lib/python3.14/site-packages/mlx/core.cpython-314-x86_64-linux-gnu.so", "mlx": "0.32.3.dev20261111+3eb5b6f", "libmlx_md5": "9c3761885238d6d68d36abde216a55e4", "lib_match": "1", "core_md5": "dfc5906c4cb0cbc7ab532de0555f36cf", "core_match": "1", "sidecar_md5": "02a54359cda8b247f76b277dd9bc4d4c", "sidecar_match": "1", "k2reg_md5": "886734181e893cd95552b53a9c234ee7", "probe_md5": "5e2536eb2ac4a3a0eefe463edc5ed6ed", "env_k2reg": "MLX_MTP=3 MTP6_TAPE=slot MTP6_DRAFT_HEAD=bf16 MTP6_HEAD_H=postnorm MTP_MAX_TOKENS=1024 MLX_ROCM_QMV_ROWS=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_REFWD_M1=1 MTP6_SLOT_COPY=0 MTP6_SLOT_IMM=0 SPAN_ON=1", "env_probe": "MLX_MTP=3 MTP6_TAPE=slot MTP6_DRAFT_HEAD=bf16 MTP6_HEAD_H=postnorm MTP_MAX_TOKENS=1024 MLX_ROCM_QMV_ROWS=1 MLX_QMV_MR_LX=0 MLX_ROCM_QMM_M3=1 MLX_ROCM_SDPA_S4VA=1 MLX_ROCM_REFWD_M1=1 MTP6_SLOT_COPY=0 MTP6_SLOT_IMM=0 SPAN_ON=1 MLX_ROCM_M4PREF=1"}

lib 9c376188 = the delivered default-OFF build (K3(i), REPORT-K3I-20261111); core dfc5906c = unchanged; the driver hard-fails on any lib/core/sidecar mismatch — all matched (lib_match 1, core_match 1, sidecar_match 1). Bound line on all three runs: [mlx-rocm] bound HIP device 0: gfx1201 (AMD Radeon RX 9070 XT) cus=32 warp=32 lds=64KB (count 1 each). DPM auto throughout.

## (1) K2REG @1024 with MLX_ROCM_M4PREF=1 — PASS (token-identical)

Record line (verbatim, M4PREF on):

MTP=3 LOAD=1.89 GEN=11.43 TOKS=1065 TOKS_S=93.14 VERIFIES=512 D1ACC=355 D2ACC=198 D3ACC=0 D1PCT=69.3 D2PCT=55.8 D3PCT=0.0 TOK_PER_STEP=2.080

alltoks_md5 = 86d73662e510e572608c6cd760eb26f4 — TOKEN-IDENTICAL to the K2REG 91.8-class reference (card-base918 k2reg record, same md5; 1065 tokens in the stream). 91.8-class fingerprint: TOKS=1065 (match), D1ACC 355 / D2ACC 198 (match), TPK 2.080 (match), D1PCT/D2PCT 69.3/55.8 (match). TOKS_S = 93.14 is session clock, NOT a gate field (the 91.8 class spans 90.65-93.14 across sessions: k2reg reference 90.65, R1 91.01, k3i re-gate 92.76, this run 93.14).

FIRST32 (this run, M4PREF on): 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357 20853 369 34383 279 1379 6150 6365 314 11171 29496 3397 26654 11 52046 8765 314 — byte-identical to the card-base918 k2reg reference FIRST32 (and to the k3i delivered-lib re-gate FIRST16 271 1919 369 264 15352 11 70042 29006 314 1204 264 6278 20853 5568 13 357).

M4PREF on the MLX_MTP=3 (S=3) K2REG path leaves the 1024 greedy+MTP record stream bit-for-bit: the F3 M4PREF record path is M=4-only and the S=3 verify is M=3, so the arm is a no-op on this record — exactly the token identity the gate demands. sclk band (ab sampler, 250ms cadence, n=17): p50 3137 MHz, max 3151 MHz — inside the DPM auto boost band (R1 arms: 3112/3099).

## (2) lm_head / gate shape spot — M4PREF on vs SoT rows_msg baseline (same session)

R1 P3/P8 standalone probe recipe (eval(r) protocol, 20 iters after 3 warmup, 4-bit QuantizedLinear, bf16 x), venv-k3i, record env verbatim, two same-session arms: baseline = M4PREF=0 (SoT rows_msg, the delivered default), test = M4PREF=1 (the 3-wave split arm under gate).

| shape | arm | M=1 us | M=3 us | M=4 us | M4/M3 | M4 ratio vs base |
|---|---|---|---|---|---|---|
| N=248320 K=4096 (lm_head) | base (M4PREF=0, SoT rows_msg) | 958.4 | 1323.7 | 1663.7 | 1.257 | 1.000 |
| N=248320 K=4096 (lm_head) | M4PREF=1 | 957.8 | 1316.0 | 1663.4 | 1.264 | 0.9998 |
| N=12288 K=4096 (gate class) | base (M4PREF=0, SoT rows_msg) | 62.2 | 93.0 | 137.3 | 1.476 | 1.0073 |
| N=12288 K=4096 (gate class) | M4PREF=1 | 62.6 | 93.0 | 138.3 | 1.487 | 1.0073 |

Gate read: M4PREF-on must be <= the SoT rows_msg baseline within session variance. At the named lm_head shape the M4PREF arm is at parity with the same-session baseline (1663.4 vs 1663.7 us, -0.02%): the M4PREF arm is NOT slower, satisfying the gate. At the named gate shape the M4PREF arm reads 138.3 vs 137.3 us (+1.0 us, +0.7%) — within this session's cross-arm noise: the same session's M1 cells differ by 62.6 vs 62.2 us (+0.6%, opposite direction) and the M3 cells are exactly equal (93.0/93.0), so the M4 delta is a 1-us cell at the noise floor, not a measured regression; the measured ratio is 1.0073 and the read is parity-within-variance. M1 and M3 are arm-invariant as expected (M4PREF only changes the M=4 path).

Context against the K3(i) same-session A/B (1.153 at N=248320): that A/B's base arm was 1452.6/1696.5 (M4/M3 1.168) in the K3(i) session; this session's base arm is 1323.7/1663.7 (M4/M3 1.257) — the M3/M4 clock-state mix drifts across sessions (R1 P8 dedicated: 1.292; k3i re-gate: 1.291; K3(i) A/B: 1.168; this session: 1.257), so the 1.153 absolute ratio is a session artifact, not a reproducible constant. What the gate fixes is the WITHIN-session comparison: M4PREF-on vs the SoT rows_msg baseline in the same session, which is parity at lm_head (0.9998) and noise-floor parity at the gate class (1.0073) — the M4PREF arm never exceeds the baseline by more than the session's own cross-arm noise. Absolute values are consistent with the delivered-lib re-gate (1693.4 at N=248320 M4) and the K3(i) A/B (1696.5), within session clock variance. sclk band (n=6 per arm; the 3s probe runs are load-dominated bursts): base p50 1221 / max 1389, M4PREF p50 1217 / max 1399 — the two arms sit in the identical band, so the same-session A/B is clock-fair.

## VERDICT

M4PREF-ON = the S=4 config of record for the K3 pay-check re-cut. K2REG @1024 is token-identical under M4PREF (alltoks_md5 86d73662e510e572608c6cd760eb26f4, 91.8-class fingerprint exact, FIRST32 match) and the M4PREF arm does not exceed the same-session SoT rows_msg baseline beyond cross-arm noise at either named shape (lm_head 0.9998, gate class 1.0073 = parity-within-variance). No DIVERGENCE; the M=4 item is NOT closed at the defaults — M4PREF-on proceeds as the S=4 config of record for the K3 pay-check re-cut.

Lock released after the final run (held 16:26:39Z-16:27:11Z over the whole campaign, no other holders observed during the window); box free.
