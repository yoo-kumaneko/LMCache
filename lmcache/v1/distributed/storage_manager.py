# SPDX-License-Identifier: Apache-2.0
"""
Distributed multi-tier storage manager for MP mode
"""

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.error import L1Error, strerror
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controllers import (
    L1EvictionController,
    L2AdapterEvictionState,
    L2EvictionController,
    PrefetchController,
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    create_prefetch_policy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    create_store_policy,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

# ABO compression modules (optional dependency)
try:
    from lmcache.v1.distributed.abo.abo_codec import (
        _DEFAULT_RATIO,
        ABOCodecFactory,
        ABOConfig,
        resolve_abo_dtype,
    )
    from lmcache.v1.distributed.abo.compress_manager import ABOCompressManager
    from lmcache.v1.distributed.abo.decompress_manager import ABODecompressManager
    from lmcache.v1.distributed.abo.staging_pool import StagingPool
    from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
    from lmcache.v1.memory_management import get_size_bytes
except ImportError:
    _DEFAULT_RATIO = None
    ABOCodecFactory = None
    ABOConfig = None
    resolve_abo_dtype = None
    ABOCompressManager = None
    ABODecompressManager = None
    StagingPool = None
    LazyMemoryAllocator = None
    get_size_bytes = None

logger = init_logger(__name__)


@dataclass(frozen=True)
class PrefetchHandle:
    prefetch_request_id: int
    """Opaque ID for tracking L2 prefetch in the controller.
    -1 if no L2 request was submitted."""

    external_request_id: str
    """Request ID from the caller for end-to-end tracing."""

    l1_prefix_hit_count: int
    """Number of leading keys already in L1 at submission time."""

    total_requested_keys: int
    """Total number of keys originally requested."""

    submit_time: float
    """Monotonic timestamp when the prefetch task was submitted."""


class StorageManager:
    def __init__(self, config: StorageManagerConfig):
        self._l1_manager = L1Manager(config.l1_manager_config)
        self._event_bus = get_event_bus()

        # ABO compression initialization
        self._abo_config = config.abo_config
        self._abo_codec = None
        self._staging_pool = None
        self._abo_compress_manager = None
        self._abo_decompress_manager = None
        if self._abo_config.enable:
            self._init_abo(config)

        # Eviction controller
        self._eviction_controller = L1EvictionController(
            l1_manager=self._l1_manager,
            eviction_config=config.eviction_config,
        )
        self._eviction_controller.start()

        # L2 adapters and store controller
        l1_memory_desc = self._l1_manager.get_l1_memory_desc()
        self._l2_adapters: list[L2AdapterInterface] = [
            create_l2_adapter(ac, l1_memory_desc)
            for ac in config.l2_adapter_config.adapters
        ]

        # Unified L2 eviction controller for all adapters with eviction config
        l2_eviction_states = [
            L2AdapterEvictionState(
                adapter=adapter,
                eviction_config=ac.eviction_config,
            )
            for adapter, ac in zip(
                self._l2_adapters, config.l2_adapter_config.adapters, strict=False
            )
            if ac.eviction_config is not None
        ]
        self._l2_eviction_controller = L2EvictionController(l2_eviction_states)
        self._l2_eviction_controller.start()

        adapter_descriptors = [
            AdapterDescriptor(index=i, config=ac)
            for i, ac in enumerate(config.l2_adapter_config.adapters)
        ]

        self._store_controller = StoreController(
            l1_manager=self._l1_manager,
            l2_adapters=self._l2_adapters,
            adapter_descriptors=adapter_descriptors,
            policy=create_store_policy(config.store_policy),
        )
        self._store_controller.start()

        # Prefetch controller
        self._prefetch_controller = PrefetchController(
            l1_manager=self._l1_manager,
            l2_adapters=self._l2_adapters,
            adapter_descriptors=adapter_descriptors,
            policy=create_prefetch_policy(config.prefetch_policy),
        )
        self._prefetch_controller.start()

    def _init_abo(self, config: StorageManagerConfig) -> None:
        """
        Initialize ABO compression components (StagingPool, codec, PIN_CHUNK_SIZE).
        """
        abo = config.abo_config
        logger.info(
            "Initializing ABO compression:"
            "staging_size=%.1fGB, ratio=%s, codec=%s, threads=%d",
            abo.staging_size_gb,
            abo.ratio,
            abo.codec,
            abo.num_threads,
        )

        # 1. Create codec via ABOCodecFactory (returns abokvpress.HuffmanCodec directly)
        codec_config = ABOConfig(
            ratio=abo.ratio,
            codec_method=abo.codec,
            num_threads=abo.num_threads,
        )
        self._abo_codec = ABOCodecFactory.create_codec(abo.codec, codec_config)
        self._abo_codec_config = (
            codec_config  # Keep config for estimate_compressed_bytes
        )

        # 2. Create ABOCompressManager (stream mode) with on_compress_failed callback
        # NOTE: abo_dtype and ratio are set later in _ensure_staging_pool
        # once the KV dtype is known from layout_desc.
        def _on_compress_failed(obj_key):
            """
            Callback when ABO compress fails: abort write to remove invalid key from L1.
            """
            try:
                result = self._l1_manager.abort_write([obj_key])
                logger.warning(
                    "ABO compress failed for key %s, abort_write result: %s",
                    obj_key,
                    result,
                )
            except Exception as e:
                logger.error(
                    "Failed to abort_write after compress failure (key=%s): %s",
                    obj_key,
                    e,
                )

        self._abo_compress_manager = ABOCompressManager(
            codec=self._abo_codec,
            on_compress_failed=_on_compress_failed,
        )

        # 3. Create ABODecompressManager
        self._abo_decompress_manager = ABODecompressManager(
            codec=self._abo_codec,
        )

        # 4. Inject codec + config into L1MemoryManager (no StagingPool yet)
        # So Prefetch path can also allocate compressed-size space
        self._l1_manager._memory_manager.setup_abo(
            self._abo_codec, codec_config=codec_config
        )

        # 5. Record staging config for lazy initialization
        # Note: staging buffer size needs to be determined by actual KV chunk size
        # at first reserve_write. Record config here, lazily initialize StagingPool
        # on first use.
        self._staging_size_gb = abo.staging_size_gb
        self._staging_pool = None  # Lazily initialized

        logger.info(
            "ABO compression initialized"
            "(StagingPool will be lazily created on first reserve_write)"
        )

    def _ensure_staging_pool(self, layout_desc: MemoryLayoutDesc) -> None:
        """Ensure StagingPool is initialized (lazy initialization).

        On first call, computes KV chunk size from layout_desc and creates StagingPool.
        Also adjusts LazyMemoryAllocator.PIN_CHUNK_SIZE.

        Args:
            layout_desc: Layout description of KV data.
        """
        if self._staging_pool is not None:
            return

        # Compute uncompressed byte size of a single KV chunk
        kv_chunk_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)

        # Adjust PIN_CHUNK_SIZE to be >= kv_chunk_size
        old_pin_chunk_size = LazyMemoryAllocator.PIN_CHUNK_SIZE
        new_pin_chunk_size = max(old_pin_chunk_size, kv_chunk_bytes)
        if new_pin_chunk_size != old_pin_chunk_size:
            LazyMemoryAllocator.PIN_CHUNK_SIZE = new_pin_chunk_size
            logger.info(
                "Adjusted LazyMemoryAllocator.PIN_CHUNK_SIZE: %d -> %d "
                "(kv_chunk_bytes=%d)",
                old_pin_chunk_size,
                new_pin_chunk_size,
                kv_chunk_bytes,
            )

        # Compute pool_size
        staging_size_bytes = int(self._staging_size_gb * (1 << 30))
        pool_size = staging_size_bytes // kv_chunk_bytes
        if pool_size < 32:
            logger.warning(
                "StagingPool pool_size=%d is too small (recommend >= 32), "
                "staging_size=%.1fGB, kv_chunk_bytes=%d",
                pool_size,
                self._staging_size_gb,
                kv_chunk_bytes,
            )
            pool_size = max(pool_size, 32)

        self._staging_pool = StagingPool(
            pool_size=pool_size,
            buffer_bytes=kv_chunk_bytes,
        )

        # Inject StagingPool into CompressedMemoryObj class-level variable
        from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj

        CompressedMemoryObj.set_staging_pool(self._staging_pool)

        # Resolve ABO dtype and ratio from layout_desc, then set on compress manager
        kv_dtype = layout_desc.dtypes[0]
        abo_dtype = resolve_abo_dtype(kv_dtype)
        ratio = self._abo_config.ratio
        if ratio is None:
            ratio = _DEFAULT_RATIO[abo_dtype]

        self._abo_compress_manager.set_abo_params(abo_dtype, ratio)

        logger.info(
            "StagingPool created and ABO params set: "
            "abo_dtype=%s, ratio=%d, kv_dtype=%s",
            abo_dtype,
            ratio,
            kv_dtype,
        )

    @property
    def is_abo_enabled(self) -> bool:
        """Whether ABO compression is enabled."""
        return self._abo_config.enable

    @property
    def abo_compress_manager(self):
        """Get the ABO compress manager (may be None)."""
        return self._abo_compress_manager

    @property
    def abo_decompress_manager(self):
        """Get the ABO decompress manager (may be None)."""
        return self._abo_decompress_manager

    # External APIs for serving engine integration code to call
    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"],
    ) -> dict[ObjectKey, MemoryObj]:
        """
        Reserve the object for writing into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for writing.
            layout_desc (MemoryLayoutDesc): Description of the memory layout
                for the objects to be reserved.
            mode (Literal["new", "update", "all"]): Reservation mode.
            - "new": Reserve only new objects that do not exist.
            - "update": Reserve only existing objects for update.
            - "all": Reserve all writable objects regardless of existence.

        Returns:
            dict[ObjectKey, MemoryObj]: A dictionary mapping object keys to their
                reserved memory objects. Note that not all requested keys could be
                reserved (e.g., out of memory or write conflict)
        """
        # ABO mode: ensure StagingPool is initialized
        if self._abo_config.enable:
            self._ensure_staging_pool(layout_desc)

        reserve_result = self._l1_manager.reserve_write(
            keys=keys,
            is_temporary=[False] * len(keys),
            layout_desc=layout_desc,
            mode=mode,
        )

        result = {k: m for k, (e, m) in reserve_result.items() if m is not None}
        successful_keys = list(result.keys())
        failed_keys = [k for k, (e, m) in reserve_result.items() if m is None]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_WRITE_RESERVED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )
        return result

    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Finish writing the objects into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys that have been written.
        """
        finish_result = self._l1_manager.finish_write(keys)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_WRITE_FINISHED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

        # TODO: global key states update

    @contextmanager
    def read_prefetched_results(
        self,
        keys: list[ObjectKey],
    ) -> Iterator[list[MemoryObj] | None]:
        """
        Read the memory objects from L1 storage that has been prefetched beforehand.
        Yielding an optional list of memory objects corresponding to the requested
        keys. If any the object is not found in L1, None is yielded.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for reading.

        Returns:
            Iterator[list[MemoryObj] | None]: An iterator yielding an optional list of
                memory objects corresponding to the requested keys.

        Note:
            If any object is not found in L1 storage, None is yielded. In this case,
            this function will release release the read lock of all successfully read
            memory objects when exiting the context.

            If the caller raised exception during the processing of the yielded memory
            objects, this function will ensure that the read locks will be decreased.
        """
        read_results = self._l1_manager.unsafe_read(keys)
        good_keys: list[ObjectKey] = []
        good_objs: list[MemoryObj] = []
        bad_keys: list[ObjectKey] = []
        all_good = True
        for k, (e, o) in read_results.items():
            if o is None:
                logger.error(
                    "Failed to read prefetched object %s from L1 storage: %s",
                    k,
                    strerror(e),
                )
                bad_keys.append(k)
                all_good = False
                continue

            good_keys.append(k)
            good_objs.append(o)

        successfully_yielded = False

        try:
            # ABO: submit async decompress tasks for CompressedMemoryObj
            if all_good and self._abo_decompress_manager is not None:
                self._abo_decompress_manager.prepare_decompress_batch(
                    good_objs, is_retrieve=True
                )

            yield good_objs if all_good else None
            successfully_yielded = True
        except Exception:
            logger.exception(
                "Exception occurred while processing read prefetched results",
            )
            raise
        finally:
            # Decrease the read lock for all successfully read memory objects
            # if None is yielded or exception occurs during caller's processing
            if not all_good or not successfully_yielded:
                self._l1_manager.finish_read(good_keys)
                self._event_bus.publish(
                    Event(
                        event_type=EventType.SM_READ_PREFETCHED_FINISHED,
                        metadata={
                            "succeeded_keys": good_keys,
                            "failed_keys": bad_keys,
                        },
                    )
                )

    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> None:
        """Finish reading prefetched objects.

        Args:
            keys: Object keys that have been read.
            extra_count: Extra read locks to release per key
                (on top of the default 1).
        """
        finish_result = self._l1_manager.finish_read(keys, extra_count=extra_count)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_READ_PREFETCHED_FINISHED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

    def submit_prefetch_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        extra_count: int = 0,
        external_request_id: str = "",
    ) -> PrefetchHandle:
        """Prefetch objects into L1 asynchronously.

        Args:
            keys: Object keys to prefetch.
            layout_desc: Memory layout description.
            extra_count: Extra workers (on top of the default
                1) that will independently retrieve the same
                key.  Total locks = 1 + extra_count.
            external_request_id: Request ID from the caller
                for end-to-end log tracing.

        Returns:
            PrefetchHandle to track the task.
        """
        # NOTE: now we only have L1, so the prefetch is essentially checking how many
        # objects are already in L1, and adding read locks to them.

        l1_read_result = self._l1_manager.reserve_read(keys, extra_count=extra_count)
        hit_count = 0
        for key in keys:
            entry = l1_read_result.get(key, None)
            if entry is None:
                break

            err, obj = entry
            if err != L1Error.SUCCESS:
                break

            hit_count += 1

        # NOTE: For L1, there will be cases that "object in the middle" is not found.
        # In this case, we need to `finish_read` for the latter objects so that
        # there won't be dangling read locks.
        skipped_keys = []
        for key in keys[hit_count:]:
            if key in l1_read_result and l1_read_result[key][1] is not None:
                # this key is actually reserved, need to release the read lock
                skipped_keys.append(key)

        if skipped_keys:
            self._l1_manager.finish_read(skipped_keys, extra_count=extra_count)

        self._event_bus.publish(
            Event(
                event_type=EventType.SM_READ_PREFETCHED,
                metadata={
                    "succeeded_keys": keys[:hit_count],
                    "failed_keys": keys[hit_count:],
                },
            )
        )

        # ABO: trigger early decompress for L1-hit objects (non-blocking)
        if hit_count > 0 and self._abo_decompress_manager is not None:
            l1_hit_objs = [
                l1_read_result[k][1]
                for k in keys[:hit_count]
                if k in l1_read_result and l1_read_result[k][1] is not None
            ]
            if l1_hit_objs:
                self._abo_decompress_manager.prepare_decompress_batch(
                    l1_hit_objs, is_retrieve=False
                )

        # Submit remaining keys to L2 prefetch controller
        remaining_keys = keys[hit_count:]
        prefetch_request_id = -1
        if remaining_keys and self._l2_adapters:
            prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
                remaining_keys,
                layout_desc,
                extra_count=extra_count,
            )

        submit_time = time.monotonic()
        logger.debug(
            "Prefetch request submitted: "
            "%d total keys, %d L1 prefix hits, "
            "%d remaining for L2 "
            "(external_request_id=%s, "
            "prefetch_request_id=%d)",
            len(keys),
            hit_count,
            len(remaining_keys),
            external_request_id,
            prefetch_request_id,
        )

        return PrefetchHandle(
            prefetch_request_id=prefetch_request_id,
            external_request_id=external_request_id,
            l1_prefix_hit_count=hit_count,
            total_requested_keys=len(keys),
            submit_time=submit_time,
        )

    def query_prefetch_lookup_hits(
        self,
        handle: PrefetchHandle,
    ) -> int | None:
        """
        Query the number of prefix hit chunks for a prefetch task before
        the L2 prefetching is done.

        Args:
            handle (PrefetchHandle): The handle of the lookup task.

        Returns:
            the number of prefix hit chunks if the lookup is done, None if
            it's still in progress,  or the prefetch task is already done.

        Note:
            This function is designed for the scenario where the caller wants
            to check the L1 prefix hits as soon as possible without waiting for
            the whole prefetch task to be done.
            When the prefetch task is already done and the prefetch task result
            has already been queried by `query_prefetch_status`, this function
            will return None forever for the same prefetch handle.
            Therefore, it's the caller’s responsibility to make sure not calling
            this function after the prefetch task is done.
        """
        if handle.prefetch_request_id == -1:
            # No L2 request, the prefix hit count is final
            return handle.l1_prefix_hit_count

        # Have L2 request, need to check the status from prefetch controller
        l2_r = self._prefetch_controller.query_lookup_result(handle.prefetch_request_id)

        if l2_r is None:
            # L2 prefetch is still in progress or it's already done and
            # the result has been consumed by `query_prefetch_status`
            return None

        # L2 lookup is done, return the total prefix hit count (L1 + L2)
        return handle.l1_prefix_hit_count + l2_r

    def query_prefetch_status(
        self,
        handle: PrefetchHandle,
    ) -> int | None:
        """
        Query the status of the prefetch task.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.

        Returns:
            the number of prefix hit chunks if the prefetch is done, None if
            it's still in progress.
        """
        l2_result: int = 0

        # Have L2 request, need to check the result from prefetch controller
        if handle.prefetch_request_id != -1:
            l2_r = self._prefetch_controller.query_prefetch_result(
                handle.prefetch_request_id
            )

            if l2_r is None:
                return None
            l2_result = l2_r  # Just to make linter happy

        total_hits = handle.l1_prefix_hit_count + l2_result
        elapsed_ms = (time.monotonic() - handle.submit_time) * 1000

        if total_hits > 0:
            logger.info(
                "Prefetch request completed (L1+L2): "
                "%d/%d prefix hits (%d L1, %d L2) "
                "in %.1f ms "
                "(external_request_id=%s, "
                "prefetch_request_id=%d)",
                total_hits,
                handle.total_requested_keys,
                handle.l1_prefix_hit_count,
                l2_result,
                elapsed_ms,
                handle.external_request_id,
                handle.prefetch_request_id,
            )
        return total_hits

    def clear(self, force: bool = False):
        """
        Clear data in the storage manager.

        Args:
            force: If True, clear ALL objects including locked ones.
                This may corrupt in-flight store/prefetch operations.
                If False (default), only clear unlocked objects, keeping
                write-locked and read-locked objects intact.
        """
        self._l1_manager.clear(force=force)

    def close(self):
        """
        Close the storage manager and release all resources.
        """
        self._prefetch_controller.stop()
        self._store_controller.stop()
        self._eviction_controller.stop()
        self._l2_eviction_controller.stop()

        for adapter in self._l2_adapters:
            adapter.close()

        # Close ABO compress manager
        if self._abo_compress_manager is not None:
            self._abo_compress_manager.close()

        # Close ABO decompress manager
        if self._abo_decompress_manager is not None:
            self._abo_decompress_manager.close()

        self._l1_manager.close()

    def report_status(self) -> dict:
        """Return a status dict aggregating all sub-component statuses."""
        l1 = self._l1_manager.report_status()
        store = self._store_controller.report_status()
        prefetch = self._prefetch_controller.report_status()
        l1_eviction = self._eviction_controller.report_status()
        l2_eviction = self._l2_eviction_controller.report_status()
        adapters = [a.report_status() for a in self._l2_adapters]
        children = [l1, store, prefetch, l1_eviction, l2_eviction] + adapters
        return {
            "is_healthy": all(c["is_healthy"] for c in children),
            "l1_manager": l1,
            "store_controller": store,
            "prefetch_controller": prefetch,
            "l1_eviction_controller": l1_eviction,
            "l2_eviction_controller": l2_eviction,
            "l2_adapters": adapters,
            "num_l2_adapters": len(self._l2_adapters),
        }

    # Functions for debugging and testing
    def memcheck(self) -> bool:
        """
        Perform memory check for all storage tiers.

        Returns:
            True if memory is consistent, False otherwise.
        """
        return self._l1_manager.memcheck()
