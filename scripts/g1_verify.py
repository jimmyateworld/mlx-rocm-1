#!/usr/bin/env python3
"""G1 bit-exact gate for the W1 table-driven GGUF loader.

Runs on neon (CPU only, no GPU needed). Gates:

S  synthetic self-tests: f32/f16/u8 bit-exact array construction, layout
   identity, layout mid_transpose exact recovery vs an independent numpy
   reference (both axes + 1D), round-trip, computed max-index bound.
H  header parse (C++): 442 records, data_start == 10,969,440 (computed, not
   assumed), 1:1 cross-check vs the pinned byte-verified table (440 records),
   the 2 extra head records reported as parsed-but-unmapped.
R  raw copy: F32 engine arrays bit-exact vs the python-gguf dequantize
   oracle (231 tensors); quant raw packed bytes == file bytes at the table
   offsets (209 tensors), (n_blocks, block_bytes) shaped.
L  per-tensor layout (D2 method): permutation-consistency multisets for all
   209 quant tensors; D2-method anchor re-derivation at layers 0 and 12
   (attn_qkv / attn_gate / ssm_out + ffn_gate control) against the local
   mxfp4a16 reference model.

Exits 0 iff every gate passes. Verdict JSON: <outdir>/w1-g1-verdict.json
"""

import argparse
import json
import struct
import sys

import numpy as np

GGUF_PATH_DEFAULT = "/home/andryo/perclass-probe/Qwen3.5-9B-Q4_K_M.gguf"
TABLE_DEFAULT = "tables/w1-gguf-table.json"
MAP_DEFAULT = "tables/w1-mapping-table.json"
SPEC_DEFAULT = "tables/w1-layout-spec.json"
OURS_DEFAULT = "/home/andryo/models-radiance/qwen3.5-9b-mxfp4a16-mtp/model.safetensors"

# File type code -> bytes. Data, not code: supplied to the C++ parser.
FLOAT_ELEM_BYTES = {0: 4, 1: 2}  # F32, F16
QUANT_BLOCK_BYTES = {8: 64, 12: 144, 13: 176, 14: 210}  # Q2_K, Q4_K, Q5_K, Q6_K


def log(msg):
    print(msg, flush=True)


def die(gate, msg):
    log(f"FAIL[{gate}] {msg}")
    sys.exit(1)


def multiset_dist(a, b):
    """Relative L1 distance between two sorted norm vectors."""
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    return float(np.abs(a - b).sum() / max(1.0, np.abs(a).sum()))


def row_col_block_norms(w, block_axis):
    rows = np.linalg.norm(w, axis=1, ord=2)
    cols = np.linalg.norm(w, axis=0, ord=2)
    d0, d1 = w.shape
    wb = w.reshape(d0, d1 // 128, 128) if block_axis == 1 else w.reshape(d0 // 128, 128, d1)
    blocks = np.linalg.norm(wb, axis=2 if block_axis == 1 else 1, ord=2).ravel()
    return rows, cols, blocks


def _ref_slot(f, groups=32, keep_ends=1, pairs=15):
    """Canonical inverse of the host window map (gguf_loader.h docstring):
    the file stores, within each `groups`-block window, [block 0][engine
    blocks 2,4,...,30][engine blocks 1,3,...,29][block 31]. For the 0-based
    file middle index e (0..M-1) the engine middle (1-based) is 2e+1
    (e < pairs, the even positions 2,4,...,M) or 2(e-pairs)+1 (e >= pairs,
    the odd positions 1,3,...,M-1); absolute slot = keep_ends + middle - 1.
    The host reads engine middle p (1..M) from file middle
    f = (p even) ? (p/2 - 1) : (pairs + (p-1)/2); this helper is its
    exact inverse (verified in gate S by composition)."""
    if f < keep_ends or f >= groups - keep_ends:
        return f
    e = f - keep_ends  # 0-based file middle index (0..M-1)
    if e < pairs:
        return keep_ends + 2 * e + 1
    return keep_ends + 2 * (e - pairs)


def file_map_pattern(W, axis, groups=32, keep_ends=1, pairs=15):
    """Independent numpy reference: engine layout -> file layout, via
    _ref_slot. `W` is a block view: (n, groups, bb) for axis 1 and
    (groups, 128, nchunks, bb) for axis 0 - a 128-element block is one
    bb-byte unit (no separate element dimension); the window unit (slab)
    is 128 blocks."""
    B = W.reshape(W.shape[0], groups, -1) if axis == 1 else W.reshape(W.shape[0], 128, W.shape[2], -1)
    out = np.empty_like(B)
    for f in range(groups):
        p = _ref_slot(f, groups, keep_ends, pairs)
        if axis == 1:
            out[:, f] = B[:, p]
        else:
            out[f] = B[p]
    return out.reshape(W.shape)


def file_map_pattern_1d(W, groups=32, keep_ends=1, pairs=15):
    """1D reference (same _ref_slot slot map): windows of `groups` blocks
    along the only axis; `W` is (n_blocks, bb)."""
    B = W.reshape(W.shape[0] // groups, groups, W.shape[1])
    out = np.empty_like(B)
    for f in range(groups):
        out[:, f] = B[:, _ref_slot(f, groups, keep_ends, pairs)]
    return out.reshape(W.shape)


# ---------------- Gate S: synthetic self-tests ----------------


def gate_s(gl):
    rng = np.random.default_rng(20261108)
    bb = 144

    # f32 bit-exact (incl. NaN/inf/sign patterns)
    bits = rng.integers(0, 2**32, 1024, dtype=np.uint64).astype(np.uint32)
    a = gl.bytes_to_array(bits.tobytes(), "f32", [1024])
    back = np.from_dlpack(a).view(np.uint32)
    if not np.array_equal(back, bits):
        die("S", "f32 round-trip not bit-exact")

    # f16 bit-exact (pins the f16 path; this file has no f16 tensors)
    hbits = rng.integers(0, 2**16, 512, dtype=np.uint64).astype(np.uint16)
    a = gl.bytes_to_array(hbits.tobytes(), "f16", [512])
    back = np.from_dlpack(a).view(np.uint16)
    if not np.array_equal(back, hbits):
        die("S", "f16 round-trip not bit-exact")

    # u8 packed view
    nb = 7
    raw = rng.integers(0, 256, nb * bb, dtype=np.uint8)
    a = gl.bytes_to_array(raw.tobytes(), "u8", [nb, bb])
    back = np.from_dlpack(a).reshape(nb, bb)
    if not np.array_equal(back, raw.reshape(nb, bb)):
        die("S", "u8 packed round-trip not bit-exact")

    # layout identity
    raw2 = rng.integers(0, 256, 4096 * 4096 // 128 * bb, dtype=np.uint8)
    out = gl.apply_layout(raw2.tobytes(), [4096, 4096], bb, "identity", 0, 32, 1, 15)
    if bytes(out) != raw2.tobytes():
        die("S", "layout identity not a copy")

    # Computed max-index assert (item 2 class guard): the host addresses
    # the packed buffer as [d0 element row][d1 block slot] for 2D (axis 0
    # and axis 1, strided per-block) and as [d0/128] blocks for 1D; the
    # largest byte offset the layout may address must stay inside the
    # buffer, so the 128x-off slab-addressing class is caught by the
    # gate, not by eye.
    d0 = d1 = 4096
    ns = d1 // 128
    buf = d0 * d1 // 128 * bb
    assert ((d0 - 1) * ns + (ns - 1)) * bb + bb <= buf, "2D max index out of buffer"
    assert (d0 // 128 - 1) * bb + bb <= (d0 // 128) * bb, "1D max index out of buffer"

    # Reference-map inverse check (no host involved): the _ref_slot
    # permutation composed with the host window formula (gguf_loader.h
    # docstring: engine middle p (1..M) reads file middle
    # f = (p even) ? (p/2 - 1) : (pairs + (p-1)/2)) is the identity.
    for p in range(1, 32 - 2 * 1 + 1):
        fs = 1 + ((p // 2 - 1) if p % 2 == 0 else (15 + (p - 1) // 2))
        if _ref_slot(fs, 32, 1, 15) != 1 + p - 1:
            die("S", f"reference map is not the inverse of the host window formula (p={p})")
    if _ref_slot(1, 32, 1, 15) != 2:
        die("S", "file slot 1 must map to engine slot 2 (canonical)")

    # mid_transpose exact recovery vs the independent numpy reference
    # (packed buffer: 4096*4096//128 = 131,072 blocks of bb; view =
    # [rows][slots][bb] for axis 1, [block-rows][row-offsets][slots][bb]
    # for axis 0 - a block is one bb unit, no element dimension).
    for axis in (0, 1):
        W = rng.integers(0, 256, 4096 * 4096 // 128 * bb, dtype=np.uint8)
        B = (  # rows, slots (32), block bytes
            W.reshape(4096, 32, bb)
            if axis == 1
            # block rows, row offsets, slots, block bytes
            else W.reshape(32, 128, 4096 // 128, bb)
        )
        Wf = np.ascontiguousarray(file_map_pattern(B, axis))
        out = gl.apply_layout(Wf.tobytes(), [4096, 4096], bb, "mid_transpose", axis, 32, 1, 15)
        got = np.frombuffer(out, dtype=np.uint8)
        got = got.reshape(4096, 32, bb) if axis == 1 else got.reshape(32, 128, 4096 // 128, bb)
        if not np.array_equal(got, B):
            die("S", f"mid_transpose axis {axis} did not restore the engine layout")
    # mid_transpose 1D: the packed 1D buffer is [d0/128] blocks of bb; the
    # host 1D path is a per-slot block copy (the item-2 1D fix under test).
    W1 = rng.integers(0, 256, (32, bb), dtype=np.uint8)  # 4096 elems / 128 = 32 blocks, (n_blocks, bb)
    Wf1 = np.ascontiguousarray(file_map_pattern_1d(W1))
    out1 = gl.apply_layout(Wf1.tobytes(), [4096], bb, "mid_transpose", 0, 32, 1, 15)
    if not np.array_equal(np.frombuffer(out1, dtype=np.uint8).reshape(32, bb), W1):
        die("S", "mid_transpose 1D did not restore the engine layout")
    log(
        "S  PASS  f32/f16/u8 bit-exact; layout identity; mid_transpose exact "
        "recovery vs independent reference (axes 0, 1 + 1D) + round-trip; "
        "computed max-index bound inside the buffer"
    )


# ---------------- Gate H: header parse ----------------


def gate_h(gl, path, pinned):
    h = gl.parse_gguf_header(path, [0, 1, 8, 12, 13, 14], QUANT_BLOCK_BYTES, FLOAT_ELEM_BYTES)
    if h["magic"] != 0x46554747 or h["version"] != 3:
        die("H", f"bad magic/version: {h['magic']:#x} / {h['version']}")
    if h["n_tensors_declared"] != 442:
        die("H", f"n_tensors_declared {h['n_tensors_declared']} != 442")
    if h["data_start"] != 10969440:
        die("H", f"data_start {h['data_start']} != 10,969,440 (computed, not assumed)")
    if len(h["records"]) != 442:
        die("H", f"parsed {len(h['records'])} records, expected 442")
    pin = {r["name"]: r for r in pinned["tensors"]}
    # 1:1 cross-check: every pinned record must appear verbatim
    seen = {}
    for r in h["records"]:
        seen.setdefault(r["name"], []).append(r)
    for name, pr in pin.items():
        cand = seen.get(name) or []
        ok = any(
            c["dims"] == pr["dims"]
            and c["type"] == pr["type"]
            and c["offset"] == pr["offset"]
            and c["size"] == pr["size"]
            for c in cand
        )
        if not ok:
            die("H", f"cross-check mismatch for {name}")
    # extra records: parsed records whose (name, offset) is not in the pinned set
    extras = [
        (r["name"], r["offset"], r["size"])
        for r in h["records"]
        if r["name"] in pin and r["offset"] != pin[r["name"]]["offset"]
    ]
    if len(extras) != 2:
        die("H", f"expected 2 extra records, got {extras}")
    log(
        f"H  PASS  442 records; data_start {h['data_start']}; 440/440 cross-check "
        f"vs pinned table; extras parsed-but-unmapped: {sorted(extras)}; "
        f"warnings: {h['warnings']}"
    )
    return h


# ---------------- Gate R: raw copy ----------------


def read_file_bytes(path, off, size):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(size)


def gate_r(gl, gguf_mod, path, h, pinned):
    data_start = h["data_start"]
    n_f32 = n_q = 0
    for pr in pinned["tensors"]:
        name = pr["name"]
        raw = gl.read_gguf_tensor(path, data_start, pr["offset"], pr["size"])
        direct = read_file_bytes(path, data_start + pr["offset"], pr["size"])
        if bytes(raw) != direct:
            die("R", f"raw bytes differ from the file at {name}")
        if pr["class"] == "F32":
            n_f32 += 1
            arr = gl.bytes_to_array(bytes(raw), "f32", pr["dims"])
            got = np.from_dlpack(arr).reshape(pr["dims"])
            oracle = gguf_mod.dequantize(
                np.frombuffer(direct, dtype=np.uint8),
                gguf_mod.constants.GGMLQuantizationType(0),
            ).reshape(pr["dims"])
            if got.shape != oracle.shape:
                die("R", f"{name}: shape {got.shape} != {oracle.shape}")
            if not np.array_equal(got.view(np.uint32), oracle.view(np.uint32)):
                k = int((got.view(np.uint32) != oracle.view(np.uint32)).sum())
                die("R", f"{name}: f32 not bit-exact ({k}/{got.size} words differ)")
        else:
            n_q += 1
            bpb = QUANT_BLOCK_BYTES[pr["type"]]
            nelems = 1
            for d in pr["dims"]:
                nelems *= d
            nb = nelems // 256
            if pr["size"] != nb * bpb:
                die("R", f"{name}: size {pr['size']} != {nb}*{bpb}")
            arr = gl.bytes_to_array(bytes(raw), "u8", [nb, bpb])
            got = np.from_dlpack(arr).reshape(nb, bpb)
            if not np.array_equal(got, np.frombuffer(raw, np.uint8).reshape(nb, bpb)):
                die("R", f"{name}: u8 packed array not bit-exact")
    log(
        f"R  PASS  {n_f32} F32 tensors bit-exact vs the python-gguf oracle; "
        f"{n_q} quant tensors raw packed bytes == file bytes at table offsets "
        f"((n_blocks, block_bytes) shaped); f16 path pinned in gate S"
    )


# ---------------- Gate L: per-tensor layout ----------------

E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def mxfp4_dequant(packed, scale):
    """packed u8 [out, in/2] (low nibble = first element), scale u8
    [out, in/32] e8m0 -> f32 [out, in]."""
    lo = (packed & 0x0F).astype(np.int32)
    hi = ((packed >> 4) & 0x0F).astype(np.int32)
    v = np.stack([E2M1_LUT[lo], E2M1_LUT[hi]], axis=-1).reshape(packed.shape[0], -1)
    e = scale.astype(np.int32) - 127
    s = np.repeat(np.repeat(e[:, :, None], 8, axis=1), 4, axis=2)
    s = np.clip(s, 0, 15)  # e8m0 is finite here
    return (v * (1 << s)).astype(np.float32)


def read_safetensors_tensor(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        info = hdr[name]
        f.seek(n + 8 + info["data_offset"])
        nb = 1
        for d in info["shape"]:
            nb *= d
        itemsize = {"F32": 4, "BF16": 2, "U8": 1, "F16": 2}[info["dtype"]]
        raw = f.read(nb * itemsize)
    if info["dtype"] == "BF16":
        arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype={"F32": np.float32, "U8": np.uint8}[info["dtype"]])
    return arr.reshape(info["shape"]).copy()


def read_ours(path, name):
    if name.endswith("_proj"):
        packed = read_safetensors_tensor(path, name + ".weight_packed")
        scale = read_safetensors_tensor(path, name + ".weight_scale")
        return mxfp4_dequant(packed, scale)
    return read_safetensors_tensor(path, name)


def dequant_gguf(gguf_mod, raw, rec, qtypes):
    if rec["class"] == "F32":
        return np.frombuffer(raw, dtype=np.uint8).view(np.float32).reshape(rec["dims"])
    return gguf_mod.dequantize(np.frombuffer(raw, dtype=np.uint8), qtypes[rec["class"]]).reshape(rec["dims"])


def gate_l(gguf_mod, gl, path, h, pinned, spec, ours_path):
    data_start = h["data_start"]
    qtypes = {
        "Q2_K": gguf_mod.constants.GGMLQuantizationType(8),
        "Q4_K": gguf_mod.constants.GGMLQuantizationType(12),
        "Q5_K": gguf_mod.constants.GGMLQuantizationType(13),
        "Q6_K": gguf_mod.constants.GGMLQuantizationType(14),
    }
    pin = {r["name"]: r for r in pinned["tensors"]}
    n_ok = 0
    for name, s in spec["tensors"].items():
        pr = pin[name]
        raw = read_file_bytes(path, data_start + pr["offset"], pr["size"])
        w_file = dequant_gguf(gguf_mod, raw, pr, qtypes)
        if s["mode"] == "identity":
            w_eng = w_file
        else:
            laid = gl.apply_layout(
                bytes(raw),
                pr["dims"],
                QUANT_BLOCK_BYTES[pr["type"]],
                s["mode"],
                s["axis"],
                s["group"],
                s["keep_ends"],
                s["pairs"],
            )
            w_eng = dequant_gguf(gguf_mod, laid, pr, qtypes)
        r1, c1, b1 = row_col_block_norms(w_file, 1)
        r2, c2, b2 = row_col_block_norms(w_eng, 1)
        for tag, x, y in (("row-norm", r1, r2), ("col-norm", c1, c2), ("block-norm", b1, b2)):
            d = multiset_dist(x, y)
            if d > 1e-5:
                die("L", f"{name}: {tag} multiset distance {d:.3e} > 1e-5")
        n_ok += 1
    log(
        f"L  PASS  permutation-consistency multisets (row/col/block) on all "
        f"{n_ok} quant tensors (Q2_K included via the gguf q2_k dequant; "
        f"its layout is identity, so consistency is trivial)"
    )

    # ---- D2-method anchor re-derivation at layers 0 and 12 ----
    our_names = {
        "attn_qkv": "linear_attn.in_proj_qkv.weight",
        "attn_gate": "linear_attn.in_proj_z.weight",
        "ssm_out": "linear_attn.out_proj.weight",
        "ffn_gate": "mlp.gate_proj",
    }
    for L in (0, 12):
        for fam in ("attn_qkv", "attn_gate", "ssm_out", "ffn_gate"):
            their_name = f"blk.{L}." + fam + ".weight"
            pr = pin[their_name]
            raw = read_file_bytes(path, data_start + pr["offset"], pr["size"])
            theirs = dequant_gguf(gguf_mod, raw, pr, qtypes)
            ours = read_ours(
                ours_path,
                "model.language_model.layers." + str(L) + "." + our_names[fam],
            )
            expected_transposed = fam != "ffn_gate"
            expected_axis = spec["tensors"][their_name]["axis"] if expected_transposed else None
            scores = {}
            for lay_axis in (None, 0, 1):
                if lay_axis is None:
                    Wfile = np.ascontiguousarray(theirs)
                else:
                    laid = gl.apply_layout(
                        bytes(raw),
                        pr["dims"],
                        QUANT_BLOCK_BYTES[pr["type"]],
                        "mid_transpose",
                        lay_axis,
                        32,
                        1,
                        15,
                    )
                    Wfile = np.ascontiguousarray(dequant_gguf(gguf_mod, laid, pr, qtypes))
                for orient in ("I", "T"):
                    W = Wfile if orient == "I" else Wfile.T
                    if W.shape != ours.shape:
                        continue
                    r1, c1, _ = row_col_block_norms(W, 1)
                    r2, c2, _ = row_col_block_norms(ours, 1)
                    s_ms = multiset_dist(r1, r2) + multiset_dist(c1, c2)
                    g = float(np.linalg.norm(W - ours) / (np.linalg.norm(ours) + 1e-30))
                    scores[(lay_axis, orient)] = (s_ms, g)
            if not scores:
                die("L", f"anchor {their_name}: no candidate matched our shape")
            (bl, bo), (bs, bg) = min(scores.items(), key=lambda kv: kv[1][0])
            if expected_transposed:
                ok = (
                    bl is not None
                    and ((None, "T") not in scores or bs < scores[(None, "T")][0] * 0.5)
                    and bl == expected_axis
                )
            else:
                ok = bl is None
            if not ok:
                die(
                    "L",
                    f"anchor {their_name}: best=({bl},{bo}) score={bs:.4f} "
                    f"gdist={bg:.4f} scores={scores} expected_axis={expected_axis}",
                )
            verdict_str = "transposed axis " + str(bl) if expected_transposed else "identity"
            log(
                f"L  anchor {their_name}: best=({bl},{bo}) ms-dist={bs:.4f} "
                f"gdist={bg:.4f} -> {verdict_str} (D2 verdict reproduced)"
            )
    log("L  PASS  D2-method anchor re-derivation (layers 0/12, 4 tensors) matches the spec")


# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=GGUF_PATH_DEFAULT)
    ap.add_argument("--table", default=TABLE_DEFAULT)
    ap.add_argument("--map", default=MAP_DEFAULT)
    ap.add_argument("--spec", default=SPEC_DEFAULT)
    ap.add_argument("--ours", default=OURS_DEFAULT)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    import gguf as gguf_mod
    import mlx.core

    gl = mlx.core.gguf_loader

    with open(a.table) as f:
        pinned = json.load(f)
    with open(a.spec) as f:
        spec = json.load(f)
    with open(a.map) as f:
        mapping = json.load(f)

    assert {r["their"] for r in mapping} == {r["name"] for r in pinned["tensors"]}, (
        "mapping table and gguf table do not match 1:1"
    )

    gate_s(gl)
    h = gate_h(gl, a.gguf, pinned)
    gate_r(gl, gguf_mod, a.gguf, h, pinned)
    gate_l(gguf_mod, gl, a.gguf, h, pinned, spec, a.ours)

    verdict = {
        "gate": "G1",
        "verdict": "PASS",
        "gguf_md5": "c367cfb9d3b251e38c4b354327b6863b (verified at G0)",
        "data_start": h["data_start"],
        "n_records": len(h["records"]),
        "spec_counts": spec["counts"],
    }
    with open(a.outdir + "/w1-g1-verdict.json", "w") as f:
        json.dump(verdict, f, indent=1)
    log("G1 VERDICT: PASS")


if __name__ == "__main__":
    main()
