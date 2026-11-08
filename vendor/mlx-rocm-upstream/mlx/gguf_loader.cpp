// Table-driven GGUF loader (host side). See gguf_loader.h.
#include "mlx/gguf_loader.h"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <system_error>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mlx::core {

namespace {

struct MmapFile {
  void* ptr = nullptr;
  size_t size = 0;
  int fd = -1;

  explicit MmapFile(const std::string& path) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("gguf_loader: cannot open " + path);
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      throw std::runtime_error("gguf_loader: fstat failed for " + path);
    }
    size = st.st_size;
    ptr = ::mmap(
        nullptr,
        size,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        0);
    if (ptr == MAP_FAILED) {
      throw std::runtime_error("gguf_loader: mmap failed for " + path);
    }
  }

  ~MmapFile() {
    if (ptr != nullptr && ptr != MAP_FAILED) {
      ::munmap(ptr, size);
    }
    if (fd >= 0) {
      ::close(fd);
    }
  }

  bool operator==(const MmapFile&) const = delete;
  MmapFile& operator=(const MmapFile&) = delete;
};

const uint8_t* data_of(const MmapFile& f) {
  return static_cast<const uint8_t*>(f.ptr);
}

inline uint32_t read_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

inline uint64_t read_u64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

// GGUF scalar value sizes (bytes). STRING (8) and ARRAY (9) are handled
// separately (variable length).
int scalar_size(int t) {
  switch (t) {
    case 0:  // UINT8
    case 1:  // INT8
    case 7:  // BOOL
      return 1;
    case 2:  // UINT16
    case 3:  // INT16
      return 2;
    case 4:  // UINT32
    case 5:  // INT32
    case 6:  // FLOAT32
      return 4;
    case 10:  // UINT64
    case 11:  // INT64
    case 12:  // FLOAT64
      return 8;
    default:
      return -1;
  }
}

bool is_printable_name(const uint8_t* p, uint64_t n) {
  if (n == 0) {
    return false;
  }
  for (uint64_t i = 0; i < n; ++i) {
    if (p[i] < 0x20 || p[i] > 0x7e) {
      return false;
    }
  }
  return true;
}

}  // namespace

GgufHeader parse_gguf_header(
    const std::string& path,
    const std::vector<uint32_t>& allowed_type_codes,
    const std::unordered_map<uint32_t, uint64_t>& quant_block_bytes,
    const std::unordered_map<uint32_t, uint64_t>& float_elem_bytes) {
  MmapFile file(path);
  const uint8_t* base = data_of(file);
  GgufHeader h;
  if (file.size < 24) {
    throw std::runtime_error("gguf_loader: file smaller than the 24-byte header");
  }
  h.magic = read_u32(base);
  h.version = read_u32(base + 4);
  h.n_tensors_declared = read_u64(base + 8);
  h.n_kv = read_u64(base + 16);
  if (h.magic != 0x46464747) {  // "GGUF"
    throw std::runtime_error("gguf_loader: bad magic 0x" +
        std::to_string(h.magic));
  }

  int64_t pos = 24;
  h.alignment = 30;  // spec default

    std::function<int64_t(int64_t, int)> skip_value =
      [&](int64_t p, int t) -> int64_t {
    if (t == 8) {  // STRING: u64 length + bytes
      uint64_t n = read_u64(base + p);
      if (p + 8 + static_cast<int64_t>(n) > static_cast<int64_t>(file.size)) {
        throw std::runtime_error("gguf_loader: string overruns the file");
      }
      return p + 8 + static_cast<int64_t>(n);
    }
    if (t == 9) {  // ARRAY: item type u32, count u64
      int it = read_u32(base + p);
      uint64_t count = read_u64(base + p + 4);
      int64_t q = p + 12;
      for (uint64_t i = 0; i < count; ++i) {
        q = skip_value(q, it);
      }
      return q;
    }
    int sz = scalar_size(t);
    if (sz < 0) {
      throw std::runtime_error(
          "gguf_loader: unknown KV value type " + std::to_string(t));
    }
    if (p + sz > static_cast<int64_t>(file.size)) {
      throw std::runtime_error("gguf_loader: value overruns the file");
    }
    return p + sz;
  };

  for (uint64_t i = 0; i < h.n_kv; ++i) {
    uint64_t n = read_u64(base + pos);
    int64_t p2 = pos + 8 + static_cast<int64_t>(n);
    if (p2 > static_cast<int64_t>(file.size)) {
      throw std::runtime_error("gguf_loader: KV name overruns the file");
    }
    std::string key(reinterpret_cast<const char*>(base + pos + 8), n);
    int t = read_u32(base + p2);
    if (key == "general.alignment" && t == 4) {
      h.alignment = read_u32(base + p2 + 4);
      if (h.alignment == 0) {
        h.alignment = 30;
        h.warnings.push_back("general.alignment is 0; using spec default 30");
      }
    }
    pos = skip_value(p2 + 4, t);
  }

  h.table_start = pos;
  int64_t table_end = pos;
  for (uint64_t i = 0; i < h.n_tensors_declared; ++i) {
    uint64_t nlen = read_u64(base + table_end);
    int64_t q = table_end + 8 + static_cast<int64_t>(nlen);
    if (q > static_cast<int64_t>(file.size)) {
      h.warnings.push_back("tensor table ends at record " + std::to_string(i) +
          " (declared " + std::to_string(h.n_tensors_declared) +
          "); last valid table end " + std::to_string(table_end));
      break;
    }
    if (!is_printable_name(base + table_end + 8, nlen)) {
      h.warnings.push_back("tensor record " + std::to_string(i) +
          " has a non-printable name; stopping the table walk");
      break;
    }
    std::string name(reinterpret_cast<const char*>(base + table_end + 8), nlen);
    int nd = read_u32(base + q);
    q += 4;
    if (nd < 1 || nd > 3 || q + 8 * static_cast<int64_t>(nd) >
            static_cast<int64_t>(file.size)) {
      h.warnings.push_back("tensor record " + std::to_string(i) +
          " has an invalid ndim; stopping the table walk");
      break;
    }
    uint64_t nelems = 1;
    for (int d = 0; d < nd; ++d) {
      uint64_t dim = read_u64(base + q + 8 * d);
      if (dim > 1ULL << 40) {
        h.warnings.push_back("tensor record " + std::to_string(i) +
            " has an implausible dimension; stopping the table walk");
        break;
      }
      nelems *= dim;
    }
    if (q + 8 * static_cast<int64_t>(nd) > static_cast<int64_t>(file.size)) {
      break;
    }
    int32_t type = read_u32(base + q + 8 * nd);
    uint64_t offset = read_u64(base + q + 8 * nd + 4);
    int64_t rec_end = q + 8 * nd + 12;
    bool valid = true;
    bool is_quant = false;
    bool is_float = false;
    uint64_t size = 0;
    auto qit = quant_block_bytes.find(static_cast<uint32_t>(type));
    auto fit = float_elem_bytes.find(static_cast<uint32_t>(type));
    if (qit != quant_block_bytes.end()) {
      is_quant = true;
      if (nelems % 256 != 0) {
        valid = false;
      } else {
        size = (nelems / 256) * qit->second;
      }
    } else if (fit != float_elem_bytes.end()) {
      is_float = true;
      size = nelems * fit->second;
    } else {
      valid = false;
    }
    if (!std::any_of(
            allowed_type_codes.begin(),
            allowed_type_codes.end(),
            [&](uint32_t c) { return c == static_cast<uint32_t>(type); })) {
      valid = false;
    }
    if (valid && (offset + size > file.size)) {
      valid = false;
    }
    if (!valid) {
      h.warnings.push_back("tensor record " + std::to_string(i) +
          " (name \"" + name + "\") failed validation; stopping the table walk");
      break;
    }
    GgufTensorRecord rec;
    rec.name = std::move(name);
    rec.ndim = nd;
    for (int d = 0; d < nd; ++d) {
      rec.dims[d] = read_u64(base + q + 8 * d);
    }
    rec.type = static_cast<uint32_t>(type);
    rec.offset = offset;
    rec.size = size;
    h.records.push_back(std::move(rec));
    table_end = rec_end;
  }
  if (static_cast<uint64_t>(h.records.size()) < h.n_tensors_declared) {
    h.warnings.push_back("walked " + std::to_string(h.records.size()) +
        " tensor records, header declares " +
        std::to_string(h.n_tensors_declared));
  }
  h.table_end = table_end;
  h.data_start =
      (static_cast<uint64_t>(table_end) + h.alignment - 1) / h.alignment *
      h.alignment;
  return h;
}

std::vector<uint8_t> read_gguf_tensor(
    const std::string& path,
    uint64_t data_start,
    uint64_t offset,
    uint64_t size) {
  MmapFile file(path);
  if (data_start + offset + size > file.size) {
    throw std::runtime_error("gguf_loader: tensor bytes overrun the file");
  }
  const uint8_t* p = data_of(file) + data_start + offset;
  return std::vector<uint8_t>(p, p + size);
}

namespace {

Dtype dtype_from_string(const std::string& s) {
  if (s == "f32") {
    return float32;
  }
  if (s == "f16") {
    return float16;
  }
  if (s == "u8") {
    return uint8;
  }
  throw std::invalid_argument("gguf_loader: unknown dtype " + s);
}

}  // namespace

array gguf_bytes_to_array(
    std::vector<uint8_t> bytes,
    const std::string& dtype,
    const std::vector<uint64_t>& shape) {
  Dtype dt = dtype_from_string(dtype);
  size_t elem_bytes = dt == uint8 ? 1 : (dt == float16 ? 2 : 4);
  size_t need = elem_bytes;
  for (uint64_t d : shape) {
    need *= d;
  }
  if (bytes.size() != need) {
    throw std::invalid_argument(
        "gguf_loader: byte size " + std::to_string(bytes.size()) +
        " does not match the shape (needs " + std::to_string(need) + ")");
  }
  auto sp = std::make_shared<std::vector<uint8_t>>(std::move(bytes));
  Shape shp;
  for (uint64_t d : shape) {
    shp.push_back(static_cast<int32_t>(d));
  }
  return array(sp->data(), std::move(shp), dt, [sp](void*) {});
}

std::vector<uint8_t> gguf_apply_layout(
    const std::vector<uint8_t>& raw,
    const std::vector<uint64_t>& dims,
    uint64_t block_bytes,
    const std::string& mode,
    int axis,
    int group,
    int keep_ends,
    int pairs) {
  if (mode == "identity") {
    return raw;
  }
  if (mode != "mid_transpose") {
    throw std::invalid_argument("gguf_loader: unknown layout mode " + mode);
  }
  if (dims.size() != 1 && dims.size() != 2) {
    throw std::invalid_argument(
        "gguf_loader: layout expects 1 or 2 dimensions");
  }
  if (axis < 0 || axis > 1) {
    throw std::invalid_argument("gguf_loader: axis must be 0 or 1");
  }
  uint64_t d0 = dims[0];
  uint64_t d1 = dims.size() == 2 ? dims[1] : 1;
  uint64_t dim_a = axis == 0 ? d0 : d1;
  uint64_t dim_o = axis == 0 ? d1 : d0;
  int M = group - 2 * keep_ends;
  if (M <= 0 || (M & 1) || M / 2 != pairs || group <= 2 * keep_ends) {
    throw std::invalid_argument("gguf_loader: inconsistent layout spec");
  }
  if (dim_a % 128 != 0 || dim_a / 128 % static_cast<uint64_t>(group) != 0 ||
      (dim_o % 128 != 0 && dim_o != 1)) {
    throw std::invalid_argument(
        "gguf_loader: dims are not compatible with the layout spec");
  }
  uint64_t nb_a = dim_a / 128;  // blocks along the block axis
  uint64_t nchunks = dim_o == 1 ? 1 : dim_o / 128;
  uint64_t ns = d1 / 128;  // block slots per row (axis-1 addressing)

  const uint8_t* src = raw.data();
  std::vector<uint8_t> out(raw.size());
  uint8_t* dst = out.data();
  std::vector<uint8_t> slab(128 * block_bytes);

  // Copy one 128-element block slab (128 blocks x block_bytes) from
  // (chunk c, window w, in-window position q) into `slab`.
  auto read_slab = [&](const uint8_t* base_, uint64_t c, uint64_t w, int q) {
    uint64_t b0 = (w * static_cast<uint64_t>(group) +
                   static_cast<uint64_t>(q));
    if (dims.size() == 1) {
      // 1D: one 128-element block per slot; the packed buffer is
      // [d0/128] blocks of block_bytes (axis-0 convention).
      std::memcpy(slab.data(), base_ + b0 * block_bytes, block_bytes);
      return;
    }
    if (axis == 1) {
      // rows c*128..(c+1)*128, slot b0: 128 strided blocks in the packed
      // [d0 element row][d1 block slot] layout (axis-0 convention).
      for (uint64_t i = 0; i < 128; ++i) {
        std::memcpy(
            slab.data() + i * block_bytes,
            base_ + ((c * 128 + i) * ns + b0) * block_bytes,
            block_bytes);
      }
    } else {
      // rows b0*128..(b0+1)*128, column chunk c: 128 strided blocks.
      for (uint64_t i = 0; i < 128; ++i) {
        std::memcpy(
            slab.data() + i * block_bytes,
            base_ + ((b0 * 128 + i) * ns + c) * block_bytes,
            block_bytes);
      }
    }
  };

  auto write_slab = [&](
      uint8_t* base_,
      uint64_t c,
      uint64_t w,
      int q,
      const uint8_t* from) {
    uint64_t b0 = (w * static_cast<uint64_t>(group) +
                   static_cast<uint64_t>(q));
    if (dims.size() == 1) {
      std::memcpy(base_ + b0 * block_bytes, from, block_bytes);
      return;
    }
    if (axis == 1) {
      // strided per-block, same convention as the axis-0 path below.
      for (uint64_t i = 0; i < 128; ++i) {
        std::memcpy(
            base_ + ((c * 128 + i) * ns + b0) * block_bytes,
            from + i * block_bytes,
            block_bytes);
      }
    } else {
      for (uint64_t i = 0; i < 128; ++i) {
        std::memcpy(
            base_ + ((b0 * 128 + i) * ns + c) * block_bytes,
            from + i * block_bytes,
            block_bytes);
      }
    }
  };

  for (uint64_t c = 0; c < nchunks; ++c) {
    for (uint64_t w = 0; w < nb_a / static_cast<uint64_t>(group); ++w) {
      for (int q = 0; q < group; ++q) {
        int src_q = q;
        if (q >= keep_ends && q < group - keep_ends) {
          // engine middle position p (1..M) <- file middle index f
          int p = q - keep_ends + 1;
          int f = (p % 2 == 0) ? (p / 2 - 1) : (pairs + (p - 1) / 2);
          src_q = keep_ends + f;
        }
        read_slab(src, c, w, src_q);
        write_slab(dst, c, w, q, slab.data());
      }
    }
  }
  return out;
}

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
    int pairs) {
  std::vector<uint8_t> raw = read_gguf_tensor(path, data_start, offset, size);
  if (mode == "identity" && dtype == "u8") {
    return gguf_bytes_to_array(std::move(raw), dtype, shape);
  }
  std::vector<uint8_t> laid =
      gguf_apply_layout(raw, shape, block_bytes, mode, axis, group, keep_ends, pairs);
  return gguf_bytes_to_array(std::move(laid), dtype, shape);
}

}  // namespace mlx::core
