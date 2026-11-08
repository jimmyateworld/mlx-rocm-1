// Table-driven GGUF loader (host side).
//
// Parses GGUF headers (KV-skipping walk + tensor table with per-record
// validation), reads tensor bytes verbatim via mmap, builds bit-exact
// mx::arrays (f32/f16/uint8), and applies per-tensor layout unscrambles
// (128-element block pair-transpose within 32-block windows) to packed
// quant bytes. All per-file parameters (allowed type codes, block bytes,
// data start, layout class) are data supplied by the caller — nothing
// model-specific is hardcoded here.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"

namespace mlx::core {

struct GgufTensorRecord {
  std::string name;
  int ndim = 0;
  std::array<uint64_t, 4> dims{};
  uint32_t type = 0;
  uint64_t offset = 0;  // relative to data_start
  uint64_t size = 0;    // computed byte count
};

struct GgufHeader {
  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t n_tensors_declared = 0;
  uint64_t n_kv = 0;
  int64_t table_start = -1;  // byte position of the first tensor record
  int64_t table_end = -1;    // byte position after the last walked record
  uint32_t alignment = 30;   // general.alignment kv (spec default 30)
  uint64_t data_start = 0;   // table_end padded up to `alignment`
  std::vector<GgufTensorRecord> records;  // file order (declared count)
  std::vector<std::string> warnings;
};

// Parse the GGUF header.
//
// Walk: magic u32, version u32, n_tensors u64, n_kv u64, then the KV fields
// (name: u64 length + bytes, type u32, value per the GGUF scalar table;
// STRING = u64 length + bytes, ARRAY = item type u32 + count u64). The walk
// extracts general.alignment (u32) and general.architecture (string) and
// skips everything else.
//
// Tensor records: name (u64 length + bytes), ndim u32, dims u64[ndim],
// type u32, offset u64 (relative to the data section start). Each record is
// validated (non-empty printable name, ndim 1..3, type in
// allowed_type_codes, nelems consistent with the byte maps, offset + size
// within the file). If a record fails validation the walk stops and a
// warning is recorded (writer quirk: declared count > present count).
//
// quant_block_bytes: type code -> bytes per 256-element block.
// float_elem_bytes: type code -> bytes per element (f32: 4, f16: 2).
GgufHeader parse_gguf_header(
    const std::string& path,
    const std::vector<uint32_t>& allowed_type_codes,
    const std::unordered_map<uint32_t, uint64_t>& quant_block_bytes,
    const std::unordered_map<uint32_t, uint64_t>& float_elem_bytes);

// mmap read of [data_start + offset, data_start + offset + size).
// Returns the bytes verbatim. offset is relative to data_start (as produced
// by parse_gguf_header).
std::vector<uint8_t> read_gguf_tensor(
    const std::string& path,
    uint64_t data_start,
    uint64_t offset,
    uint64_t size);

// Build an mx::array from raw bytes with a bit-exact copy (no reordering,
// no conversion): dtype "f32" (4 B/element), "f16" (2 B/element, bits kept
// as float16), or "u8" (shape is the packed [n_blocks, block_bytes] view).
array gguf_bytes_to_array(
    std::vector<uint8_t> bytes,
    const std::string& dtype,
    const std::vector<uint64_t>& shape);

// Layout unscramble of packed quant bytes: bit-exact block rearrangement,
// no conversion. dims is the row-major shape in elements; block_bytes is
// the packed size of one 128-element block (144/176/210/64/... from the
// tensor table data).
//
// mode "identity": copy.
// mode "mid_transpose": 128-element blocks are grouped along `axis` into
// windows of `group` blocks. Within each window the first and last
// `keep_ends` blocks stay in place; the middle M = group - 2*keep_ends
// blocks are stored in the file as [even engine positions 2, 4, ..., M]
// followed by [odd 1, 3, ..., M-1] (pairs = M/2 of each). The engine
// (canonical) order is restored by taking, for engine middle position p
// (1..M), file middle index f = (p even) ? (p/2 - 1) : (pairs + (p-1)/2).
std::vector<uint8_t> gguf_apply_layout(
    const std::vector<uint8_t>& raw,
    const std::vector<uint64_t>& dims,
    uint64_t block_bytes,
    const std::string& mode,
    int axis,
    int group,
    int keep_ends,
    int pairs);

// Combined: read + layout + array (one call, one mmap).
array gguf_load_tensor(
    const std::string& path,
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
    int pairs);

}  // namespace mlx::core
