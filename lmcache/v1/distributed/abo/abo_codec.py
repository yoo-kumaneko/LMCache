# SPDX-License-Identifier: Apache-2.0

"""
ABO codec basics.

Aligned with tx-lmcache's abo_basics.py pattern:
- ABOConfig: codec configuration
- ABOCodecFactory: creates abokvpress.HuffmanCodec directly
- estimate_compressed_bytes: standalone utility function
- resolve_abo_dtype: convert torch.dtype to ABO dtype string

The codec returned by ABOCodecFactory.create_codec() is a raw
abokvpress.HuffmanCodec instance. Callers invoke codec.compress()
and codec.decompress() directly with numpy buffers:
    result = codec.compress(dst_np, dst_size, src_np, src_size, dtype="bf16")
    result = codec.decompress(dst_np, dst_size, src_np, src_size)
"""

# Standard
import math
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

try:
    from abokvpress import HuffmanCodec
except ImportError as e:
    raise ImportError(
        "abokvpress is not installed. Please install it with: pip install abokvpress"
    ) from e

# PyTorch dtype -> ABO dtype string
PYTORCH_DTYPE_TO_ABO: dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.uint8: "fp8e4m3",
    torch.float8_e4m3fn: "fp8e4m3",
}

# Default compression ratio per ABO dtype
_DEFAULT_RATIO: dict[str, int] = {
    "bf16": 22,
    "fp8e4m3": 26,
}

# Alignment granularity (bytes)
_ALIGN_BYTES = 4096


@dataclass
class ABOConfig:
    """ABO codec configuration."""

    ratio: Optional[int] = None
    """Compression ratio (20-32), None = auto-select based on dtype."""

    codec_method: str = "huffman"
    """Codec method name."""

    num_threads: int = 32
    """Number of threads used for compress/decompress."""

    # Internal field: ABO dtype string (set by create_codec or caller)
    _dtype: Optional[str] = None


def resolve_abo_dtype(dtype: torch.dtype) -> str:
    """Convert torch.dtype to ABO dtype string.

    Args:
        dtype: PyTorch dtype.

    Returns:
        ABO dtype string (e.g. "bf16", "fp8e4m3").

    Raises:
        ValueError: If dtype is not supported by ABO.
    """
    abo_dtype = PYTORCH_DTYPE_TO_ABO.get(dtype)
    if abo_dtype is None:
        raise ValueError(
            f"Unsupported dtype for ABO: {dtype}. "
            f"Supported: {list(PYTORCH_DTYPE_TO_ABO.keys())}"
        )
    return abo_dtype


class ABOCodecFactory:
    """Factory for creating ABO codec instances.

    Returns raw abokvpress.HuffmanCodec — no Python wrapper layer.
    dtype and ratio are NOT passed to the constructor; they are
    specified per compress() call.
    """

    @staticmethod
    def create_codec(method: str, config: ABOConfig):
        """Create a codec instance based on the specified method.

        Args:
            method: Codec method name (e.g. "huffman").
            config: ABOConfig with num_threads, etc.

        Returns:
            abokvpress.HuffmanCodec instance.

        Raises:
            ValueError: If method is unsupported.
        """
        method = method.lower()

        if method == "huffman":
            codec = HuffmanCodec()
            codec.set_num_threads(config.num_threads)

            logger.info(
                "ABO HuffmanCodec created: num_threads=%d",
                config.num_threads,
            )
            return codec
        else:
            raise ValueError(
                f"Unsupported ABO codec method: {method}. Supported: huffman"
            )


def estimate_compressed_bytes(
    shape: torch.Size,
    dtype: torch.dtype,
    ratio: Optional[int] = None,
) -> int:
    """Estimate compressed size in bytes.

    Formula: original_bytes * ratio / 32, aligned up to _ALIGN_BYTES.

    Args:
        shape: Original tensor shape.
        dtype: Original tensor dtype.
        ratio: Compression ratio. None = auto-select based on dtype.

    Returns:
        Estimated compressed bytes (aligned).

    Raises:
        ValueError: If dtype is not supported by ABO.
    """
    abo_dtype = PYTORCH_DTYPE_TO_ABO.get(dtype)
    if abo_dtype is None:
        raise ValueError(
            f"Unsupported dtype for ABO: {dtype}. "
            f"Supported: {list(PYTORCH_DTYPE_TO_ABO.keys())}"
        )

    if ratio is None:
        ratio = _DEFAULT_RATIO[abo_dtype]

    # Original bytes
    numel = 1
    for s in shape:
        numel *= s
    element_size = torch._utils._element_size(dtype)
    original_bytes = numel * element_size

    # Estimated compressed bytes
    compressed_bytes = int(math.ceil(original_bytes * ratio / 32))

    # Align up to _ALIGN_BYTES
    aligned_bytes = (compressed_bytes + _ALIGN_BYTES - 1) // _ALIGN_BYTES * _ALIGN_BYTES

    return aligned_bytes
