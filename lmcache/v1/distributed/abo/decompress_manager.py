# SPDX-License-Identifier: Apache-2.0

"""
ABO decompress manager (CUDA stream + launch_host_func mode).

Manages async decompression in the RETRIEVE path:
1. Batch-submit decompress tasks via cupy launch_host_func on _decompress_stream
2. Each decompress callback runs sync decompress (codec.decompress with numpy buffers)
3. After decompress, record event for GPU-side sync (current_stream.wait_event)
4. After H2D, release staging buffer via launch_host_func callback

Key constraint:
  cudaLaunchHostFunc callbacks need GIL. Never call stream.synchronize()
  or event.synchronize() from the same thread that holds GIL while
  host callbacks are pending on that stream.

Timing design:
  _decompress_stream:
    launch_host_func(decomp0) → decomp_event0 → launch_host_func(decomp1) → ...
  high_priority stream:
    wait(decomp_event0) → H2D chunk0 → wait(decomp_event1) → H2D chunk1 → ...
  _decompress_stream:
    wait(h2d_event0) → launch_host_func(release_staging0) → ...
"""

# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import cupy
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj

if TYPE_CHECKING:
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


class ABODecompressManager:
    """Manages async decompression in the RETRIEVE path (stream mode).

    Uses a dedicated CUDA stream + cupy launch_host_func to schedule
    sync decompress as host callbacks. The callback executes in the
    CUDA driver's callback thread (which acquires GIL automatically).

    H2D loop uses .tensor property's wait_event for GPU-side sync.
    After H2D, staging release is also done via launch_host_func.
    """

    def __init__(
        self,
        codec,
        device: Optional[torch.device] = None,
    ):
        """Initialize decompress manager.

        Args:
            codec: abokvpress.HuffmanCodec instance (from ABOCodecFactory).
            device: CUDA device (None = current device).
        """
        self._codec = codec

        # Lazily created CUDA stream + cupy wrapper
        self._decompress_stream: Optional[torch.cuda.Stream] = None
        self._cupy_decompress_stream: Optional[cupy.cuda.ExternalStream] = None
        self._device = device

    def _get_decompress_stream(
        self,
    ) -> tuple[torch.cuda.Stream, cupy.cuda.ExternalStream]:
        """Get or create the dedicated decompress CUDA stream + cupy wrapper."""
        if self._decompress_stream is None:
            device = self._device or torch.cuda.current_device()
            self._decompress_stream = torch.cuda.Stream(device=device)
            self._cupy_decompress_stream = cupy.cuda.ExternalStream(
                self._decompress_stream.cuda_stream
            )
        return self._decompress_stream, self._cupy_decompress_stream

    def prepare_decompress_batch(
        self,
        memory_objs: list["MemoryObj"],
        is_retrieve: bool = False,
    ) -> list["MemoryObj"]:
        """Batch-submit async decompress tasks for a list of MemoryObj.

        For CompressedMemoryObj without staging buffer (data in raw_data),
        acquire staging buffer, submit sync decompress via launch_host_func
        on _decompress_stream, record decompress_event.

        For non-compressed MemoryObj or those already having staging, do nothing.

        Args:
            memory_objs: MemoryObj list from read_prefetched_results.
            is_retrieve: If True, this is the retrieve (final) path;
                acquire failure marks decompress_failed on the obj.
                If False, this is the prefetch (early) path;
                acquire failure silently skips the obj.

        Returns:
            Processed MemoryObj list (same list, internal state modified).
        """
        decompress_stream, cupy_decompress_stream = self._get_decompress_stream()

        for obj in memory_objs:
            if not isinstance(obj, CompressedMemoryObj):
                continue

            # Already has staging (data still in staging, no decompress needed)
            if obj.has_staging:
                continue

            # Acquire staging buffer via obj's class-level pool
            if not obj.try_acquire_staging():
                if is_retrieve:
                    logger.warning(
                        "StagingPool exhausted during retrieve, "
                        "cannot acquire staging buffer for decompress"
                    )
                    # Mark decompress as failed so .tensor returns None,
                    # triggering error in H2D -> retrieve failure path
                    obj.mark_decompress_failed()
                else:
                    logger.debug(
                        "StagingPool has no free buffer, skipping prefetch decompress"
                    )
                continue

            # Capture references for the closure
            _obj = obj
            _codec = self._codec
            _staging = obj._staging_tensor

            def _decompress_callback(_arg, _o=_obj, _c=_codec, _s=_staging):
                """Host callback: sync decompress raw_data → staging.

                Runs in CUDA driver's callback thread (GIL acquired by cupy).
                Args:
                    _arg: User data from cupy launch_host_func (unused).
                """
                try:
                    # codec.decompress(dst_np, dst_size, src_np, src_size)
                    # Pass full raw_data; codec self-describes block boundaries
                    dst_np = _s.numpy()
                    src_np = _o.raw_data.numpy()
                    result = _c.decompress(dst_np, dst_np.nbytes, src_np, src_np.nbytes)
                    if not result.success:
                        logger.error("ABO decompress failed: %s", result.error_msg)
                        _o.mark_decompress_failed()
                    else:
                        _o.mark_decompress_done()
                except Exception as e:
                    logger.error("Decompress callback failed: %s", e)
                    _o.mark_decompress_failed()

            # Register host callback on decompress_stream
            cupy_decompress_stream.launch_host_func(_decompress_callback, None)

            # Record decompress done event on decompress_stream
            decompress_event = torch.cuda.Event()
            decompress_event.record(decompress_stream)

            # Set decompress state on the obj
            obj.setup_decompress(decompress_event)

        return memory_objs

    def submit_per_chunk_release(
        self,
        h2d_event: torch.cuda.Event,
        compressed_obj: "CompressedMemoryObj",
    ) -> None:
        """Submit per-chunk staging release via launch_host_func.

        Called in H2D loop after each chunk H2D completes.
        The release callback is registered on the main stream (via h2d_event),
        so it executes after H2D is done.

        Args:
            h2d_event: CUDA event recorded after this chunk's H2D.
            compressed_obj: CompressedMemoryObj instance.
        """
        decompress_stream, cupy_decompress_stream = self._get_decompress_stream()

        # Use stream-level dependency: _decompress_stream waits for h2d_event
        # so the release callback only fires after H2D is done on the GPU.
        # This avoids calling h2d_event.synchronize() inside the callback
        # (which would hold GIL and deadlock with other host callbacks).
        decompress_stream.wait_event(h2d_event)

        def _release_callback(_arg):
            """Host callback: release staging buffer (H2D already done via stream dep)."""
            try:
                compressed_obj.release_staging()
            except Exception as e:
                logger.error("Per-chunk staging release failed: %s", e)

        cupy_decompress_stream.launch_host_func(_release_callback, None)

    def release_staging_batch(
        self,
        memory_objs: list["MemoryObj"],
    ) -> None:
        """Release staging buffers for a batch of MemoryObj (sync mode, legacy path).

        Called in finish_read_prefetched callback.
        If already released via submit_per_chunk_release, this is a no-op.

        Args:
            memory_objs: MemoryObj list.
        """
        for obj in memory_objs:
            if isinstance(obj, CompressedMemoryObj) and obj.has_staging:
                obj.release_staging()

    def close(self) -> None:
        """Clean up resources.

        NOTE: Do NOT call decompress_stream.synchronize() here — it would
        deadlock if any host callbacks are pending.
        """
        pass
