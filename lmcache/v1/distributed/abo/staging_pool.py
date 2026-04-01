# SPDX-License-Identifier: Apache-2.0

"""
StagingPool: Index-based pinned memory buffer pool.

Used as GPU↔CPU data transfer intermediary in ABO compression mode.
Each buffer is an independent pinned memory tensor sized to one
uncompressed KV chunk.

Thread-safe: uses threading.Condition for blocking wait and release notification.
"""

# Standard
from collections import deque
from typing import Optional
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class StagingPool:
    """Index-based pinned memory buffer pool for GPU↔CPU data transfer intermediary.

    Each buffer is an independent pinned memory tensor allocated via
    torch.empty(..., pin_memory=True). Provides blocking acquire and
    release interfaces. Thread-safe.

    Attributes:
        pool_size: Total number of buffers.
        buffer_bytes: Size of each buffer in bytes.
    """

    def __init__(self, pool_size: int, buffer_bytes: int):
        """Initialize StagingPool.

        Args:
            pool_size: Number of pre-allocated buffers.
            buffer_bytes: Size of buffer in bytes (= uncompressed size of one KV chunk).

        Raises:
            ValueError: If pool_size <= 0 or buffer_bytes <= 0.
        """
        if pool_size <= 0:
            raise ValueError(
                f"StagingPool pool_size must be a positive integer, got: {pool_size}"
            )
        if buffer_bytes <= 0:
            raise ValueError(
                f"StagingPool buffer_bytes must be a positive integer, "
                f"got: {buffer_bytes}"
            )

        self.pool_size = pool_size
        self.buffer_bytes = buffer_bytes
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Pre-allocate pinned memory buffers
        self._buffers: list[torch.Tensor] = []
        for i in range(pool_size):
            buf = torch.empty(buffer_bytes, dtype=torch.uint8, pin_memory=True)
            self._buffers.append(buf)

        # Free buffer index queue
        self._free_indices: deque[int] = deque(range(pool_size))

        logger.info(
            "StagingPool initialized: pool_size=%d, buffer_bytes=%d, "
            "total_pinned_memory=%.2f GB",
            pool_size,
            buffer_bytes,
            pool_size * buffer_bytes / (1 << 30),
        )

    def try_acquire(self) -> Optional[torch.Tensor]:
        """Non-blocking acquire a staging buffer.

        Returns immediately with a buffer if one is available,
        or None if the pool is exhausted. Never blocks.

        Returns:
            A pinned memory tensor (torch.uint8), or None if no buffer available.
        """
        with self._lock:
            if len(self._free_indices) == 0:
                return None
            idx = self._free_indices.popleft()
            return self._buffers[idx]

    def release(self, buf: torch.Tensor) -> None:
        """Return a staging buffer to the pool.

        Args:
            buf: Buffer previously obtained via acquire().

        Raises:
            ValueError: If buf does not belong to this StagingPool.
        """
        # Find the index of the buffer
        idx = self._find_buffer_index(buf)
        if idx is None:
            raise ValueError(
                "Attempting to release a buffer not belonging to this StagingPool "
                f"(data_ptr={buf.data_ptr():#x})"
            )

        with self._cond:
            if idx in self._free_indices:
                logger.warning(
                    "StagingPool.release: buffer idx=%d already in free queue, "
                    "possible duplicate release",
                    idx,
                )
                return
            self._free_indices.append(idx)
            self._cond.notify()

    def _find_buffer_index(self, buf: torch.Tensor) -> Optional[int]:
        """Find buffer index in the pool by data_ptr."""
        target_ptr = buf.data_ptr()
        for i, pool_buf in enumerate(self._buffers):
            if pool_buf.data_ptr() == target_ptr:
                return i
        return None

    @property
    def available_count(self) -> int:
        """Number of currently available buffers (no lock, for monitoring/debugging)."""
        return len(self._free_indices)

    @property
    def total_bytes(self) -> int:
        """Total pinned memory bytes occupied by StagingPool."""
        return self.pool_size * self.buffer_bytes

    def __repr__(self) -> str:
        return (
            f"StagingPool(pool_size={self.pool_size}, "
            f"buffer_bytes={self.buffer_bytes}, "
            f"available={self.available_count})"
        )
