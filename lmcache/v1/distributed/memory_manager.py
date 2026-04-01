# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
    MixedMemoryAllocator,
)

# ABO compression modules (optional dependency)
try:
    from lmcache.v1.distributed.abo.abo_codec import estimate_compressed_bytes
    from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj
except ImportError:
    estimate_compressed_bytes = None
    CompressedMemoryObj = None

logger = init_logger(__name__)


# HELPER FUNCTIONS
def create_memory_allocator(config: L1MemoryManagerConfig) -> MemoryAllocatorInterface:
    """
    Create a memory allocator based on the provided configuration.

    Args:
        config (L1MemoryManagerConfig): Configuration for the memory manager.

    Returns:
        MemoryAllocatorInterface: An instance of a memory allocator.
    """
    if config.use_lazy:
        logger.debug(
            "use lazy memory allocator, init size is %d bytes, "
            "final size is %d bytes, align bytes is %d bytes",
            config.init_size_in_bytes,
            config.size_in_bytes,
            config.align_bytes,
        )
        return LazyMemoryAllocator(
            config.init_size_in_bytes, config.size_in_bytes, config.align_bytes
        )
    else:
        logger.debug(
            "use mixed memory allocator, total size is %d bytes, "
            "align bytes is %d bytes",
            config.size_in_bytes,
            config.align_bytes,
        )
        return MixedMemoryAllocator(
            config.size_in_bytes,
            align_bytes=config.align_bytes,
        )


# MAIN CLASS
class L1MemoryManager:
    """
    L1MemoryManager manages the allocation and deallocation of L1 memory.

    Observability metrics to emit:
    1. Memory usage
    2. Active allocations
    """

    def __init__(self, config: L1MemoryManagerConfig):
        self._allocator = create_memory_allocator(config)
        self._size_in_bytes = config.size_in_bytes
        self._align_bytes = config.align_bytes

        # ABO compression components (injected by StorageManager after init)
        self._abo_codec: Optional[object] = None  # abokvpress.HuffmanCodec instance
        self._abo_codec_config: Optional[object] = None  # ABOConfig instance
        self._is_abo_enabled: bool = False

    def setup_abo(self, abo_codec, codec_config=None) -> None:
        """Inject ABO compression components. Called by StorageManager at init.

        Args:
            abo_codec: abokvpress.HuffmanCodec instance (from ABOCodecFactory).
            codec_config: ABOConfig instance (for estimate_compressed_bytes).
        """
        self._abo_codec = abo_codec
        if codec_config is not None:
            self._abo_codec_config = codec_config
        self._is_abo_enabled = True
        logger.info("L1MemoryManager ABO enabled")

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """
        Allocate memory objects based on the provided layout description and count.
        This function should be thread-safe

        Args:
            layout_desc (MemoryLayoutDesc): Description of the memory layout.
            count (int): Number of memory objects to allocate.

        Returns:
            tuple[L1Error, list[MemoryObj]]: Error code and list of
            allocated memory objects.
            Error code will be `L1Error.OUT_OF_MEMORY` if allocation
            fails; otherwise, it will be `L1Error.SUCCESS`.

        Note:
            If the allocation fails, the memory object list will be empty.
        """
        objects = self._allocator.batched_allocate(
            layout_desc.shapes, layout_desc.dtypes, count
        )
        if objects is None:
            return L1Error.OUT_OF_MEMORY, []
        return L1Error.SUCCESS, objects

    def allocate_compressed(
        self,
        layout_desc: MemoryLayoutDesc,
        count: int,
        need_compresses: list[bool] = None,
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate compressed MemoryObj (ABO mode).

        Allocates compressed-size L1 space as raw_data and creates CompressedMemoryObj.
        Staging buffer is not allocated here but lazy-acquired in the .tensor getter.

        Args:
            layout_desc: Layout description of the original (uncompressed) data.
            count: Number of objects to allocate.
            need_compress: Whether compress is needed (True for STORE path,
                False when loading from L2 in RETRIEVE path).

        Returns:
            tuple[L1Error, list[MemoryObj]]: Error code and list of allocated MemoryObj.
        """
        if not self._is_abo_enabled:
            return L1Error.GENERIC_ERROR, []

        assert self._abo_codec is not None

        # Estimate compressed size using standalone function
        original_shape = layout_desc.shapes[0]
        original_dtype = layout_desc.dtypes[0]
        ratio = (
            self._abo_codec_config.ratio if self._abo_codec_config is not None else None
        )
        compressed_bytes = estimate_compressed_bytes(
            original_shape, original_dtype, ratio=ratio
        )

        # Build compressed-size layout_desc
        compressed_shape = torch.Size([compressed_bytes])
        compressed_layout = MemoryLayoutDesc(
            shapes=[compressed_shape],
            dtypes=[torch.uint8],
        )

        # Allocate compressed-size space from L1 memory pool
        raw_objects = self._allocator.batched_allocate(
            compressed_layout.shapes, compressed_layout.dtypes, count
        )
        if raw_objects is None:
            return L1Error.OUT_OF_MEMORY, []

        if need_compresses is None:
            need_compresses = [True] * count
        # Wrap as CompressedMemoryObj
        # Staging buffer is not allocated here but lazy-acquired in .tensor getter
        compressed_objs: list[MemoryObj] = []
        for raw_obj, need_compress in zip(raw_objects, need_compresses, strict=True):
            # Save original metadata info into CompressedMemoryObj
            # raw_obj's meta comes from allocator with correct address and phy_size
            compressed_obj = CompressedMemoryObj(
                raw_data=raw_obj.raw_data,
                metadata=raw_obj.meta,
                parent_allocator=raw_obj.parent_allocator,
                staging_tensor=None,  # lazy allocation
                original_shape=original_shape,
                original_dtype=original_dtype,
            )

            # Mark whether compress is needed
            compressed_obj._need_compress = need_compress

            # Set original shape/dtype into meta (so upper-layer code can correctly
            # interpret data)
            compressed_obj.meta.shape = original_shape
            compressed_obj.meta.dtype = original_dtype
            compressed_obj.meta.shapes = layout_desc.shapes
            compressed_obj.meta.dtypes = layout_desc.dtypes

            compressed_objs.append(compressed_obj)

            # Invalidate original raw_obj to avoid double-free
            # (CompressedMemoryObj has taken over its raw_data and parent_allocator)
            raw_obj.parent_allocator = None

        return L1Error.SUCCESS, compressed_objs

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """
        Free the provided memory objects.
        This function should be thread-safe.

        Args:
            mem_objs (list[MemoryObj]): List of memory objects to free.

        Returns:
            L1Error: Error code indicating the result of the operation.
            It will be `L1Error.SUCCESS` if the operation succeeds.
        """
        # Release staging buffer for CompressedMemoryObj first
        for obj in mem_objs:
            if isinstance(obj, CompressedMemoryObj) and obj.has_staging:
                obj.release_staging()

        self._allocator.batched_free(mem_objs)
        return L1Error.SUCCESS

    def get_memory_usage(self) -> tuple[int, int]:
        """
        Get the current memory usage. This function will mainly be used to support
        eviction decision.

        Returns:
            tuple[int, int]: A tuple containing used memory in bytes and total memory
            in bytes.

        Note:
            In the future, we may want to make a "callback" based mechanism to
            trigger eviction when the memory usage reaches a watermark.
        """

        # HACK: now trying to read this from the address manager in a ad-hoc
        # manner
        def get_address_manager(allocator: MemoryAllocatorInterface):
            if isinstance(allocator, MixedMemoryAllocator) and hasattr(
                allocator.pin_allocator, "address_manager"
            ):
                return allocator.pin_allocator.address_manager
            elif isinstance(allocator, LazyMemoryAllocator):
                return allocator.get_address_manager()
            else:
                raise NotImplementedError(
                    "get_memory_usage is not implemented for this allocator type."
                )

        address_manager = get_address_manager(self._allocator)
        free_size = address_manager.get_free_size()
        total_size = address_manager.get_heap_size()
        used_size = total_size - free_size
        return used_size, total_size

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """
        Return an L1MemoryDesc describing the underlying memory buffer.

        Returns:
            L1MemoryDesc: Pointer, size, and alignment of the L1 buffer.

        Raises:
            NotImplementedError: If the allocator type does not support this operation.
        """
        if isinstance(self._allocator, MixedMemoryAllocator):
            buffer = self._allocator.buffer
        elif isinstance(self._allocator, LazyMemoryAllocator):
            # TODO(ApostaC): need to test if the RDMA registration works
            # before the lazy expansion is finished
            buffer = self._allocator.get_underlying_buffer()
        else:
            raise NotImplementedError(
                "get_l1_memory_desc is not implemented for this allocator type."
            )
        return L1MemoryDesc(
            ptr=buffer.data_ptr(),
            size=self._size_in_bytes,
            align_bytes=self._align_bytes,
        )

    def close(self) -> None:
        """
        Close the memory manager and release all resources.
        """
        self._allocator.close()

    # Debugging APIs
    def memcheck(self):
        return self._allocator.memcheck()
