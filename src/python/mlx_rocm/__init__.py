"""mlx-rocm — ROCm backend for MLX (MVP).

MVP skeleton: vendored mlx ROCm backend tree (maternion/mlx:rocm-current,
forward-merge of ml-explore/mlx PR #2300) + a minimal standalone hipBLASLt
GEMM layer (bf16 + fp8 e4m3) with a CPU reference for validation.
"""

__version__ = "0.0.1"
__all__ = ["gemm", "__version__"]
