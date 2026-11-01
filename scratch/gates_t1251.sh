#!/bin/bash
# t1251 gates (rev2): G2 lossless (prefix-aware, awk-safe), G3 K=3 speed,
# G4 4-bit drafter (F32 sidecar parked during run), G5 shapebench + lmh probe.
set -u
PY=/home/<host-user>/venv-fused2/bin/python
B=bench/mtp-bench6.py
MD=/mnt/models/qwen3-9b-mlx
cd /home/<host-user>
mkdir -p t1251/gates
E="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 HCC_AMDGPU_TARGET=gfx1201"

echo "=== SIDECAR PIN (G2/G3: F32 sidecar only)"
cd $MD
for f in model-mtp-4bit.safetensors model-mtp-fixed.safetensors; do
  [ -f "$f" ] && mv "$f" "$f.disabled"
done
ls model-*.safetensors* | head
cd /home/<host-user>

run() { # name mtp
  local name=$1 mtp=$2
  echo "=== GATE RUN: $name (MLX_MTP=$mtp)"
  env $E MLX_MTP=$mtp MLX_MTP_DRAFT=3 \
  MTP_MODEL=$MD MTP_MAX_TOKENS=1024 \
  $PY $B > "t1251/gates/gate_$name.log" 2>&1
  grep -E "TOKS_S|TOK_PER_STEP" "t1251/gates/gate_$name.log" | head -2
}

# G2: lossless (prefix or full-stream match)
run g2_ar 0
run g2_k3 4
m5()  { grep "^ALLTOKS:" "$1" | md5sum | awk '{print $1}'; }
m5p() { grep "^ALLTOKS:" "$1" | awk '{for(i=2;i<=513;i++) printf "%s ", $i; print ""}' | md5sum | awk '{print $1}'; }
A=$(m5 t1251/gates/gate_g2_ar.log);  C=$(m5 t1251/gates/gate_g2_k3.log);  G0A=$(m5 t1251/g0-baseline.log)
AP=$(m5p t1251/gates/gate_g2_ar.log); CP=$(m5p t1251/gates/gate_g2_k3.log); G0AP=$(m5p t1251/g0-baseline.log)
echo "md5 full  : new_ar=$A new_k3=$C old_g0=$G0A"
echo "md5 pref512: new_ar=$AP new_k3=$CP old_g0=$G0AP"
{ [ "$A" = "$C" ] || [ "$AP" = "$CP" ]; } && echo "G2a PASS (K=3 matches AR: prefix or full)" || echo "G2a FAIL"
{ [ "$A" = "$G0A" ] || [ "$AP" = "$G0AP" ]; } && echo "G2b PASS (new AR matches G0: prefix or full)" || echo "G2b FAIL (check drift)"
NAR=$(grep "^ALLTOKS:" t1251/gates/gate_g2_ar.log | awk '{print NF-1}')
N3=$(grep "^ALLTOKS:" t1251/gates/gate_g2_k3.log | awk '{print NF-1}')
NG0=$(grep "^ALLTOKS:" t1251/g0-baseline.log | awk '{print NF-1}')
echo "len: new_ar=$NAR new_k3=$N3 old_g0=$NG0"

# G3: K=3 speed
grep -E "TOKS_S" t1251/gates/gate_g2_k3.log | head -1 | sed "s/^/G3: /"

# G4: 4-bit drafter (park F32 sidecar so the glob merge picks the 4-bit one)
cd $MD
[ -f model-mtp-4bit.safetensors.disabled ] && mv model-mtp-4bit.safetensors.disabled model-mtp-4bit.safetensors
[ -f model-mtp-fixed.safetensors ] && mv model-mtp-fixed.safetensors model-mtp-fixed.safetensors.disabled
mv model-mtp.safetensors model-mtp.safetensors.disabled 2>/dev/null || true
cd /home/<host-user>
echo "=== GATE RUN: g4_k3_4bit (MLX_MTP=4 MTP_4BIT=1, F32 sidecar parked)"
env $E MLX_MTP=4 MLX_MTP_DRAFT=3 MTP_4BIT=1 \
MTP_MODEL=$MD MTP_MAX_TOKENS=1024 \
$PY $B > "t1251/gates/gate_g4_4bit.log" 2>&1
grep -E "TOKS_S|VERIFIES|D1ACC" "t1251/gates/gate_g4_4bit.log" | head -2 | sed "s/^/G4: /"
mv $MD/model-mtp.safetensors.disabled $MD/model-mtp.safetensors 2>/dev/null || true
D=$(grep "^ALLTOKS:" t1251/gates/gate_g4_4bit.log | awk '{for(i=2;i<=21;i++) printf "%s ", $i; print ""}')
F=$(grep "^FIRST20:" t1251/gates/gate_g2_k3.log | awk '{for(i=2;i<=21;i++) printf "%s ", $i; print ""}')
echo "G4 first20 4bit: $D"
echo "G4 first20 f32 : $F"

# G5: in-wheel shapebench + lmh probe (default vs NOMR)
if [ ! -f t1251/shape-bench.py ]; then
cat > t1251/shape-bench.py << 'PYBENCH'
import time
import mlx.core as mx

shapes = [("lm_head", 4096, 248320), ("gate_up", 4096, 12288),
          ("fa_q", 4096, 8192), ("k_v", 4096, 1024),
          ("fa_o_gdn_o", 8192, 4096), ("down", 12288, 4096)]

def one(M, N, K):
    x = mx.random.uniform(0, 1, (M, K))
    w = mx.random.uniform(-1, 1, (N, K))
    q, sc, bz = mx.quantize(w, bits=4, group_size=64)
    f = lambda: mx.quantized_matmul(x, q, sc, bz)
    for _ in range(5): f()
    mx.eval(f())
    t0 = time.perf_counter()
    n = 20
    for _ in range(n): f()
    mx.eval(f())
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1000

for name, K, N in shapes:
    a = one(4, N, K)
    b = one(1, N, K)
    print(f"{name}\tM4={a:.3f}\tM1={b:.3f}")
PYBENCH
fi
env $E $PY t1251/shape-bench.py > t1251/gates/g5-shapebench.log 2>&1
env $E $PY t1251/lmh2.py 4096 248320 > t1251/gates/g5-lmh-def.log 2>&1
env $E MLX_QMV_NO_MR=1 $PY t1251/lmh2.py 4096 248320 > t1251/gates/g5-lmh-nomr.log 2>&1
tail -4 t1251/gates/g5-shapebench.log | sed "s/^/G5 /"
echo "G5 lmh def : $(tail -1 t1251/gates/g5-lmh-def.log)"
echo "G5 lmh nomr: $(tail -1 t1251/gates/g5-lmh-nomr.log)"

# restore sidecars
cd $MD
for f in model-mtp-4bit.safetensors.disabled model-mtp-fixed.safetensors.disabled model-mtp.safetensors.disabled; do
  [ -f "$f" ] && mv "$f" "${f%.disabled}"
done
echo "=== SIDECARS RESTORED"
