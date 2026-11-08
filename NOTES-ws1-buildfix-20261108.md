# WS1 build failure diagnosis + fix (ticket 2541 continuation, 2026-11-08)

## Diagnosis

The pre-stop run (ticket 2541) launched the neon wheel build WITHOUT the pinned
`CMAKE_ARGS`. Evidence: `ws1-build.log` line 101 prints
`-- Setting CMAKE_HIP_ARCHITECTURES to: gfx908;gfx90a;gfx942;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102`
(the project default list, i.e. the variable was never passed; the lane
`setup.py` forwards `os.environ["CMAKE_ARGS"]` verbatim, lines 118-119, so an
absent env var means no `-D` flags at all).

Consequence: the multi-arch default set includes gfx1010 (and other non-gfx12
targets). The lane's `mlx/backend/rocm/r4d_gdn_wmma.h:130` uses the
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` builtin, which requires the
target features `wmma-128b-insts,wavefrontsize32` — only enabled on gfx1201+.
Compile error (log line 19370):

    r4d_gdn_wmma.h:130:10: error: '__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12'
    needs target feature wmma-128b-insts,wavefrontsize32
    1 error generated when compiling for gfx1010.

This is a build-invocation failure, not a source defect: the source tree
(scratch = pinned lane tree, diff vs vendor-verify-s1 = exactly our 3 added
files + 1 CMakeLists wiring line) is unchanged by this fix.

## Fix

Rebuild per the pinned recipe (vendor.json `build.cmake_args`), single arch
`gfx1201` so the WMMA builtin compiles with its default target features:

    CMAKE_PREFIX_PATH=/opt/rocm \
    CMAKE_ARGS='-DMLX_BUILD_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBLA_VENDOR=OpenBLAS' \
    python3.14 -m pip wheel . --no-deps -w wheelout/

with the standard hygiene: fresh scratch venv (python3.14), wipe `build/` +
`dist/`, purge pip cache, touch the parent dirs of changed files.

Gate: new `libmlx.so` md5 MUST differ from the record lib
`259f9aeefe3ec02f8993fe562f5ee62b` (venv-vr3).
