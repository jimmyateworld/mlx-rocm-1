# WS1 report — ticket 2541 salvage (wheel rebuild + G1 CPU gate) — 2026-11-08

**VERDICT: FAIL** — the pinned wheel recipe now builds cleanly on neon (task 2 gate PASS), but the G1 gate could not complete after the allowed fix round (task 3 gate FAIL); per the one-fix-round rule the verdict stands as FAIL with the evidence below.

## Pins verified (before any work)

Pins all verified before starting: branch w1-ws1-loader at e85b0aa, the GGUF wheel 5868826976 bytes with md5 c367cfb9…, and all three table md5s (gguf-table, mapping-table, layout-spec) matched the ticket; python-gguf 0.19.0 was at /var/tmp/gglib2 with the C library symbols intact.

## Task 1+2 — the failed build, diagnosed and rebuilt

The ticket-2541 pre-stop build launched `pip wheel` without CMAKE_ARGS, so CMake fell back to the default multi-arch list (gfx908…gfx1102) and the lane's r4d_gdn_wmma.h gfx12 builtin (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12, which requires the wmma-128b-insts,wavefrontsize32 target features) failed to compile; the fix was to pass the pinned CMAKE_ARGS='-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS' with a fresh venv (python 3.14.5, OpenBLAS, no ccache), plus two source fixes in the first build round: the python/src/gguf_loader.cpp binding needed `using namespace nanobind::literals;` (the `_a` argument literals live in nanobind::literals, not directly in nanobind), and the wheel had to be built with TMPDIR on /home/andryo because /tmp was at quota.

The G1 import then failed with `undefined symbol: mlx::core::gguf_load_tensor`: the lane target compiles with -fvisibility=hidden and the lane's convention is to mark public APIs with MLX_API (mlx/api.h, "must add for public APIs"), which the host header had not done; marking the five public loader functions (parse_gguf_header, read_gguf_tensor, gguf_bytes_to_array, gguf_apply_layout, gguf_load_tensor) fixed the export (all five show as T in nm -D of the built lib).

The next G1 run failed with UnicodeDecodeError on apply_layout: nanobind UTF-8-decodes std::string return values, so the two byte-returning bindings (read_gguf_tensor, apply_layout) had to return nb::bytes instead of std::string (nb::bytes was verified to copy correctly in both directions, 4KB…64MB round-trip probe passed); the same G1 round also added the Q2_K dequant type (1 Q2_K tensor is in the table+spec; python-gguf supports the q2_k dequant) to the script's qtypes map.

The rebuilt wheel: mlx-0.32.3.dev20261108+0d3e6df-cp314-cp314-linux_x86_64.whl, 22,597,891 bytes, sha256 b79d62d2c4c7a7941619f50c9a38c76b0bbd48302d7843f95d637e676ef3869b; libmlx.so md5 31d464391c8efe46ab10f0999a42b064, size 149,884,624 B (record: 259f9aeefe3ec02f8993fe562f5ee62b / 149,821,136 B) — the task-2 "lib md5 must differ" gate PASSES; core.so is present in the wheel (4.7MB) and the gguflib lane is not vendored (extern "C" include, python-gguf used only on the oracle side).

## Task 3 — G1 gate: FAIL (evidence)

G1 run 1 (07:0x) died at import (undefined symbol, fixed by the MLX_API change); G1 run 2 (07:1x) died in gate S with UnicodeDecodeError (fixed by the nb::bytes return change); G1 run 3 (07:2x) got further — f32 bit-exact, f16 bit-exact, u8 packed-view, and layout-identity sub-checks all PASSED — but then crashed in the gate-S mid_transpose section: `ValueError: cannot reshape array of size 9437184 into shape (32,128,32,128,144)` (9,437,184 = 65,536 × 144, the buffer the script allocated with 4096*4096//256*bb, versus the 131,072 × 144 = 18,874,368 bytes its own reshape target requires for 4096×4096 f32 = 131,072 128-element blocks).

Root-cause analysis (deterministic, from the committed sources): (a) the script's gate-S buffer size math is off by 2 (//256 instead of //128 — fixed in b225061 but unverified, see below); (b) the script's independent numpy reference window map is NOT the canonical inverse — its formula sends file slot 1 to engine block 0 (the kept end block) where the canonical map (host header docstring: file stores [0][2,4,…,30][1,3,…,29][31]) sends it to engine block 2; the host's own window math matches the canonical map exactly; (c) the host's axis-1 (and 1D) slab addressing — `base + (c*128*ns + b0) * 128 * block_bytes` — is 128× off for a packed buffer: for a 4096×4096 Q4_K tensor it indexes up to (31×4096+31)×18,432 ≈ 2.34 GB into an 18.8 MB buffer (out-of-bounds read/write, UB), while the host's axis-0 path is consistent with the packed layout ([row 4096][block 32][144]).

Per the ticket's one-fix-round rule, G1 failing after the fix round means FAIL: fixing (b) and (c) — and re-verifying (a) — requires a fresh fix round (reference-map correction + host axis-1/1D layout reconciliation + another wheel build), which is disallowed, so the verdict is FAIL with the evidence above; gates H (442-record header parse), R (231 f32 + 209 quant raw-copy vs the python-gguf oracle), and L (layout D2 permutation-consistency) were never reached in any run.

## State for the next ticket (WS2 / G1 retry)

Everything is in place on neon: the source is at /home/andryo/ws1-build (mirror of branch w1-ws1-loader @ b225061), the working wheel is at /home/andryo/ws1-build/wheelout-w1d/ (installed in /home/andryo/ws1-g1-venv), the G1 venv + tables are at /home/andryo/g1-tables/ and the G1 script at /home/andryo/g1_verify.py; the wheel recipe is `CMAKE_ARGS='-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS'` with TMPDIR=/home/andryo/ws1-tmp; a full wheel build on neon takes ~5 minutes; the known remaining work is the gate-S mid_transpose reconciliation (script reference map + host axis-1/1D slab addressing) and then a full G1 pass.

Commits this ticket (branch w1-ws1-loader, pushed, mirror verified at each step): 2a7d451 build-fix note, d5ae77c nanobind::literals fix, b6b44bd MLX_API visibility fix, 5507e31 nb::bytes returns + Q2_K dequant, b225061 gate-S buffer math (incomplete, recorded for this verdict).
