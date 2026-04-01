# SPDX-License-Identifier: Apache-2.0

"""
CompressedMemoryObj: ABO-compression-aware MemoryObj subclass.

Extends TensorMemoryObj with staging buffer management, async
compress/decompress state tracking, and compression-aware
.tensor / .byte_array property behaviour.

Lifecycle (STORE path):
  allocate → [staging buffer + raw_data (compressed space, fixed size)]
    │
    ▼ GPU→CPU D2H copy → data written into staging buffer
    │
    ▼ _compress_task (async thread):
    │   staging → codec.compress → raw_data
    │   release staging buffer
    │
    ▼ L2 store:
        byte_array → memoryview of raw_data (fixed size, codec self-describes boundaries)

Lifecycle (RETRIEVE path):
  L2 load → raw_data (compressed data)
    │
    ▼ acquire staging buffer + decompress (async CUDA stream):
    │   raw_data → codec.decompress → staging buffer
    │   record _decompress_event
    │
    ▼ CPU→GPU H2D copy ← staging buffer (.tensor property wait_event)
    │
    ▼ release staging buffer
"""

# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

if TYPE_CHECKING:
    from lmcache.v1.distributed.abo.staging_pool import StagingPool
    from lmcache.v1.memory_management import MemoryAllocatorInterface

logger = init_logger(__name__)


class CompressedMemoryObj(TensorMemoryObj):
    """ABO-compression-aware MemoryObj subclass.

    Extends TensorMemoryObj with:
    - Staging buffer management (for D2H/H2D transfer intermediary)
    - Async compress/decompress state tracking
    - Multi-state .tensor property behaviour
    - Compression-aware .byte_array property behaviour
    """

    # Class-level staging pool shared by all instances.
    # Injected once via set_staging_pool() after StagingPool is created.
    _cls_staging_pool: Optional["StagingPool"] = None

    @classmethod
    def set_staging_pool(cls, pool: "StagingPool") -> None:
        """Inject the global StagingPool reference (called once at init).

        Args:
            pool: StagingPool instance shared by all CompressedMemoryObj.
        """
        cls._cls_staging_pool = pool

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        parent_allocator: Optional["MemoryAllocatorInterface"],
        staging_tensor: Optional[torch.Tensor],
        original_shape: torch.Size,
        original_dtype: torch.dtype,
    ):
        """Initialize CompressedMemoryObj.

        Args:
            raw_data: Compressed-size space allocated from L1 pool (uint8 flat tensor).
            metadata: Memory object metadata.
            parent_allocator: Parent allocator (LazyMemoryAllocator).
            staging_tensor: Pinned buffer borrowed from StagingPool
                (optional, None for RETRIEVE).
            original_shape: Shape of the original uncompressed tensor.
            original_dtype: Dtype of the original uncompressed tensor.
        """
        super().__init__(raw_data, metadata, parent_allocator)

        # Staging buffer related
        self._staging_tensor: Optional[torch.Tensor] = staging_tensor
        self._original_shape: torch.Size = original_shape
        self._original_dtype: torch.dtype = original_dtype

        # Compress state (STORE path)
        self._need_compress: bool = (
            staging_tensor is not None
        )  # needs compress when holding staging
        self._compress_failed: bool = (
            False  # True if compress failed (graceful degradation)
        )

        # Decompress state (RETRIEVE path)
        self._need_decompress: bool = False
        self._decompress_failed: bool = (
            False  # True if decompress failed (graceful degradation)
        )
        self._decompress_event: Optional[torch.cuda.Event] = None
        self._h2d_event: Optional[torch.cuda.Event] = (
            None  # CUDA event for H2D completion (for per-chunk release)
        )

        # Backup of meta.address (set to 0 during staging phase, restored
        # after compress)
        self._original_address: Optional[int] = None

        # If holding staging, set up memcpy adaptation immediately
        if self._staging_tensor is not None:
            self._setup_staging_memcpy_meta()

    def _setup_staging_memcpy_meta(self):
        """Modify meta during staging phase so lmcache_memcpy_async works correctly.

        Sets meta.address to 0 because the staging buffer is an independent
        pinned memory region. Combined with PIN_CHUNK_SIZE being set to
        >= kv_chunk_size, this makes lmcache_memcpy_async issue a single
        cudaMemcpyAsync for the staging buffer (no segmentation).
        """
        self._original_address = self.meta.address
        self.meta.address = 0

    def _restore_raw_data_meta(self):
        """Restore meta to raw_data's address after compress is done."""
        if self._original_address is not None:
            self.meta.address = self._original_address
            self._original_address = None

    def get_size(self) -> int:
        """Get the effective byte size for memcpy operations.

        During staging phase (holding staging buffer), returns the
        uncompressed size so that lmcache_memcpy_async copies the
        full KV data. After staging is released, falls back to the
        base class which returns the compressed (raw_data) size.
        """
        if self._staging_tensor is not None:
            return self._original_shape.numel() * self._original_dtype.itemsize
        return super().get_size()

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        """Get tensor view.

        Multi-state behaviour:
        1. _compress_failed or _decompress_failed: return None (graceful degradation)
        2. _need_compress=True and staging is None: lazy acquire staging and return
        3. Holding staging and _need_decompress=True: wait_event then return
            staging view
        4. Holding staging and _need_decompress=False: return staging view directly
        5. Staging released and no compress needed: return raw_data view
            (compressed data)
        """
        if not self.valid:
            logger.warning("Trying to access an invalidated CompressedMemoryObj")
            return None

        # Compress or decompress failed: no valid data available
        if self._compress_failed or self._decompress_failed:
            return None

        # Lazy staging allocation: in STORE path, need compress but staging
        # not yet allocated
        if self._need_compress and self._staging_tensor is None:
            if not self.try_acquire_staging():
                logger.error(
                    "StagingPool exhausted, cannot lazy acquire staging buffer "
                    "(address=%d)",
                    self.meta.address,
                )
                return None

        if self._staging_tensor is not None:
            # If decompress is needed, wait for decompress to complete first
            if self._need_decompress and self._decompress_event is not None:
                # Wait for decompress on the current CUDA stream (stream-level sync)
                current_stream = torch.cuda.current_stream()
                current_stream.wait_event(self._decompress_event)
                self._need_decompress = False

            # Return staging buffer view (original shape/dtype)
            original_bytes = (
                self._original_shape.numel() * self._original_dtype.itemsize
            )
            return (
                self._staging_tensor[:original_bytes]
                .view(self._original_dtype)
                .view(self._original_shape)
            )

        # Staging released, return raw_data view
        return super().tensor

    @property
    def byte_array(self) -> Optional[memoryview]:
        """Get binary data view.

        Compression-aware behaviour:
        - If _compress_failed=True: return None (graceful degradation)
        - If compression is still in progress (_need_compress=True),
          wait for compress_event to complete. This blocks the caller
          (StoreController's background thread) until compressed data
          is ready in raw_data, without blocking the main thread.
        - Returns memoryview of full raw_data (fixed size buffer).
          The codec embeds block metadata so the decompressor
          self-describes boundaries; no need to trim to actual
          compressed size.
        """
        # Compress failed: no valid compressed data available
        if self._compress_failed:
            return None

        # Compression still in progress: wait for it to complete
        if self._need_compress and hasattr(self, "_compress_event"):
            self._compress_event.synchronize()

        return super().byte_array

    def set_compress_event(self, event: torch.cuda.Event):
        """Set the CUDA event for compress completion (stream mode).

        Args:
            event: Compress-done event recorded on compress_stream.
        """
        self._compress_event = event

    def mark_compress_done(self):
        """Mark compress as done."""
        self._need_compress = False

    def mark_compress_failed(self):
        """Mark compress as failed (graceful degradation).

        Resets compress state so that the object is not stuck in
        _need_compress=True forever. The object will not have valid
        compressed data, so L2 store should skip it.
        Sets _compress_failed=True so that .tensor returns None.
        """
        self._need_compress = False
        self._compress_failed = True

    def mark_decompress_failed(self):
        """Mark decompress as failed (graceful degradation).

        Sets _decompress_failed=True so that .tensor returns None,
        which triggers an error in the H2D path and falls into
        the retrieve failure handling.
        """
        self._decompress_failed = True

    def try_acquire_staging(self) -> bool:
        """Acquire a staging buffer from the class-level StagingPool.

        On success, sets self._staging_tensor and calls
        _setup_staging_memcpy_meta().

        Returns:
            True if staging buffer was acquired, False otherwise.
        """
        pool = CompressedMemoryObj._cls_staging_pool
        if pool is None:
            logger.error("StagingPool not initialized (set_staging_pool not called)")
            return False
        staging = pool.try_acquire()
        if staging is None:
            return False
        self._staging_tensor = staging
        self._setup_staging_memcpy_meta()
        return True

    def setup_decompress(
        self,
        decompress_event: torch.cuda.Event,
    ):
        """Set up decompress state (used in RETRIEVE path).

        Staging buffer must already be acquired via try_acquire_staging()
        before calling this method.

        Args:
            decompress_event: CUDA event for decompress completion.
        """
        assert self._staging_tensor is not None, (
            "setup_decompress requires staging buffer (call try_acquire_staging first)"
        )
        self._decompress_event = decompress_event
        self._need_decompress = True

    def mark_decompress_done(self):
        """Mark decompress as done.

        Clears decompress state after decompressed data has been consumed.
        """
        self._need_decompress = False
        self._decompress_event = None

    def mark_staging_released(self):
        """Mark staging buffer as released.

        Clears staging reference and restores meta.address.
        The actual buffer return to pool should be done by the caller.

        Returns:
            The staging tensor that was held (or None if already released).
        """
        staging = self._staging_tensor
        self._staging_tensor = None
        self._restore_raw_data_meta()
        self._need_decompress = False
        self._decompress_failed = False
        self._decompress_event = None
        self._h2d_event = None
        return staging

    def release_staging(self):
        """Release staging buffer back to StagingPool.

        Before releasing, ensures:
        - If decompress is in progress, synchronize event

        NOTE: No h2d_event.synchronize() here. In per-chunk release path,
        the caller uses stream-level wait_event on _decompress_stream.
        No compress wait needed either. In STORE path, the compress
        callback itself releases staging via mark_staging_released().
        """
        if self._staging_tensor is None:
            return

        # Synchronize decompress event
        if self._need_decompress and self._decompress_event is not None:
            self._decompress_event.synchronize()

        # NOTE: No h2d_event.synchronize() here.
        # In per-chunk release path, the caller (submit_per_chunk_release)
        # uses stream-level wait_event(h2d_event) on _decompress_stream,
        # so the release callback only fires after H2D is done.
        # Calling h2d_event.synchronize() in a host callback would hold GIL
        # and deadlock with other host callbacks on gpu_context.stream.
        # Use mark_staging_released to clear state and get staging tensor
        staging = self.mark_staging_released()

        # Return staging buffer to pool
        pool = CompressedMemoryObj._cls_staging_pool
        if pool is not None and staging is not None:
            pool.release(staging)

    @property
    def has_staging(self) -> bool:
        """Whether currently holding a staging buffer."""
        return self._staging_tensor is not None

    def __del__(self):
        """Destructor: ensure staging buffer is returned to pool."""
        if self._staging_tensor is not None:
            try:
                self.release_staging()
            except Exception:
                pass
        # Call parent destructor
        super().__del__()

    def __repr__(self) -> str:
        return (
            f"CompressedMemoryObj("
            f"address={self.meta.address}, "
            f"phy_size={self.meta.phy_size}, "
            f"original_shape={self._original_shape}, "
            f"original_dtype={self._original_dtype}, "
            f"has_staging={self.has_staging}, "
            f"need_compress={self._need_compress}, "
            f"compress_failed={self._compress_failed}, "
            f"need_decompress={self._need_decompress}, "
            f"decompress_failed={self._decompress_failed})"
        )
