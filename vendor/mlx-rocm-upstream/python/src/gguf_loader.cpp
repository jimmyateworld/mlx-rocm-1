// Copyright © 2026. Table-driven GGUF loader bindings (host side).
//
// Exposes the host GGUF loader as the `mlx.core.gguf_loader` submodule:
// header parse (tensor table walk with the mandatory computed data_start),
// verbatim mmap reads, bit-exact array construction, and per-tensor layout
// unscramble of packed quant bytes.
#include <cstring>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "mlx/gguf_loader.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nanobind::literals;  // arg-name literal operator""_a (nanobind 3.0.1, nb_attr.h)

void init_gguf_loader(nb::module_& m) {
  auto sub = m.def_submodule(
      "gguf_loader",
      "Table-driven GGUF loader (host): header parse, verbatim reads, "
      "bit-exact arrays, per-tensor layout unscramble.");

  sub.def(
      "parse_gguf_header",
      [](const std::string& path,
         const std::vector<uint32_t>& allowed_type_codes,
         const std::unordered_map<uint32_t, uint64_t>& quant_block_bytes,
         const std::unordered_map<uint32_t, uint64_t>& float_elem_bytes) {
        mx::GgufHeader h = mx::parse_gguf_header(
            path,
            allowed_type_codes,
            quant_block_bytes,
            float_elem_bytes);
        nb::dict d;
        d["magic"] = (uint32_t)h.magic;
        d["version"] = (uint32_t)h.version;
        d["n_tensors_declared"] = (uint64_t)h.n_tensors_declared;
        d["n_kv"] = (uint64_t)h.n_kv;
        d["table_start"] = (int64_t)h.table_start;
        d["table_end"] = (int64_t)h.table_end;
        d["alignment"] = (uint32_t)h.alignment;
        d["data_start"] = (uint64_t)h.data_start;
        d["warnings"] = h.warnings;
        nb::list recs;
        for (const auto& r : h.records) {
          nb::dict rj;
          rj["name"] = r.name;
          nb::list dims;
          for (int i = 0; i < r.ndim; ++i) {
            dims.append((uint64_t)r.dims[i]);
          }
          rj["dims"] = dims;
          rj["type"] = (uint32_t)r.type;
          rj["offset"] = (uint64_t)r.offset;
          rj["size"] = (uint64_t)r.size;
          recs.append(rj);
        }
        d["records"] = recs;
        return d;
      },
      "path"_a,
      "allowed_type_codes"_a,
      "quant_block_bytes"_a,
      "float_elem_bytes"_a,
      R"pbdoc(
Parse the GGUF header: walk and skip the KV fields (extracting
general.alignment), then walk the declared tensor records (name u64-len +
bytes, ndim u32, dims u64[ndim], type u32, offset u64 relative to the data
section start), validating each. A record that fails validation stops the
walk (writer quirk: declared count > present count) and is reported in
`warnings`. `data_start` is the table end padded up to `alignment`.
quant_block_bytes: type code -> bytes per 256-element block;
float_elem_bytes: type code -> bytes per element.
)pbdoc");

  sub.def(
      "read_gguf_tensor",
      [](const std::string& path,
         uint64_t data_start,
         uint64_t offset,
         uint64_t size) {
        std::vector<uint8_t> v =
            mx::read_gguf_tensor(path, data_start, offset, size);
        return nb::bytes(reinterpret_cast<const char*>(v.data()), v.size());
      },
      "path"_a,
      "data_start"_a,
      "offset"_a,
      "size"_a,
      R"pbdoc(
Read [data_start + offset, +size) verbatim via mmap and return the bytes.
)pbdoc");

  sub.def(
      "bytes_to_array",
      [](nb::bytes bytes,
         const std::string& dtype,
         const std::vector<uint64_t>& shape) {
        std::vector<uint8_t> v(bytes.size());
        std::copy(reinterpret_cast<const uint8_t*>(bytes.data()),
                  reinterpret_cast<const uint8_t*>(bytes.data()) + bytes.size(),
                  v.begin());
        return mx::gguf_bytes_to_array(std::move(v), dtype, shape);
      },
      "bytes"_a,
      "dtype"_a,
      "shape"_a,
      R"pbdoc(
Build an mx.array from raw bytes with a bit-exact copy (no conversion):
dtype "f32" (4 B/element), "f16" (2 B/element, bits kept as float16) or
"u8" (the packed [n_blocks, block_bytes] view).
)pbdoc");

  sub.def(
      "apply_layout",
      [](nb::bytes raw,
         const std::vector<uint64_t>& dims,
         uint64_t block_bytes,
         const std::string& mode,
         int axis,
         int group,
         int keep_ends,
         int pairs) {
        std::vector<uint8_t> v(raw.size());
        std::copy(reinterpret_cast<const uint8_t*>(raw.data()),
                  reinterpret_cast<const uint8_t*>(raw.data()) + raw.size(),
                  v.begin());
        std::vector<uint8_t> out = mx::gguf_apply_layout(
            v,
            dims,
            block_bytes,
            mode,
            axis,
            group,
            keep_ends,
            pairs);
        return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
      },
      "raw"_a,
      "dims"_a,
      "block_bytes"_a,
      "mode"_a,
      "axis"_a,
      "group"_a,
      "keep_ends"_a,
      "pairs"_a,
      R"pbdoc(
Layout unscramble of packed quant bytes (bit-exact block rearrangement).
mode "identity" copies; mode "mid_transpose" restores the engine (canonical)
block order from the file order: within each 32-block window (along `axis`,
128-element blocks) the first/last `keep_ends` blocks stay and the middle
30 are stored in the file as [evens 2..30][odds 1..29] (pairs = 15).
)pbdoc");

  sub.def(
      "load_tensor",
      [](const std::string& path,
         uint64_t data_start,
         uint64_t offset,
         uint64_t size,
         uint64_t block_bytes,
         const std::string& dtype,
         const std::vector<uint64_t>& shape,
         const std::string& mode,
         int axis,
         int group,
         int keep_ends,
         int pairs) {
        return mx::gguf_load_tensor(
            path,
            data_start,
            offset,
            size,
            block_bytes,
            dtype,
            shape,
            mode,
            axis,
            group,
            keep_ends,
            pairs);
      },
      "path"_a,
      "data_start"_a,
      "offset"_a,
      "size"_a,
      "block_bytes"_a,
      "dtype"_a,
      "shape"_a,
      "mode"_a,
      "axis"_a,
      "group"_a,
      "keep_ends"_a,
      "pairs"_a,
      R"pbdoc(
Read a tensor from the file (offset relative to data_start), apply the
layout unscramble, and build the mx.array (one call, one mmap).
)pbdoc");
}

