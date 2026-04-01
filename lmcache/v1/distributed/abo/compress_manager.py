# SPDX-License-Identifier: Apache-2.0

"""
ABO compress task manager (CUDA stream + launch_host_func mode).

Manages per-chunk async compression in the STORE path:
1. cupy_stream.launch_host_func(compress_fn) — register sync compress
   as host callback on compress_stream
2. compress_done_event.record(compress_stream) — mark compress done

The host callback runs in CUDA driver's callback thread (GIL acquired
by cupy). Inside the callback, it first does d2h_event.synchronize()
(CPU-side wait for D2H), then executes sync compress, marks done,
and releases staging. This way the D2H wait happens in the callback
thread, not blocking the main thread.

Timing design:
  main_stream:       D2H chunk0 → d2h_event0.record → D2H chunk1 → ...
  compress_stream:   launch_host_func(compress0) → compress_done_event0.record → ...
  (callback thread): d2h_event0.synchronize() → compress0() → mark_done + release staging
"""

# Standard
from typing import TYPE_CHECKING, Callable, Optional

# Third Party
import cupy
import torch

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    from lmcache.v1.distributed.abo.compressed_memory_obj import CompressedMemoryObj
    from lmcache.v1.distributed.api import ObjectKey

logger = init_logger(__name__)


class ABOCompressManager:
    """Manages async compression in the STORE path (stream mode).

    Uses a dedicated CUDA stream + cupy launch_host_func to schedule
    sync compress as a host callback. The callback executes in the
    CUDA driver's callback thread (which acquires GIL automatically).

    All operations (compress, staging release, mark done) are registered
    on the same compress_stream via launch_host_func, so FIFO ordering
    guarantees correct sequencing without extra synchronization primitives.
    """

    def __init__(
        self,
        codec,
        device: Optional[torch.device] = None,
        on_compress_failed: Optional[Callable[["ObjectKey"], None]] = None,
    ):
        """Initialize compress manager.

        Args:
            codec: abokvpress.HuffmanCodec instance (from ABOCodecFactory).
            device: CUDA device (None = current device).
            on_compress_failed: Callback on compress failure.
        """
        self._codec = codec
        self._abo_dtype: Optional[str] = None
        self._ratio: Optional[int] = None
        self._device = device
        self._on_compress_failed = on_compress_failed

        # Lazily created CUDA stream (needs CUDA context)
        self._compress_stream: Optional[torch.cuda.Stream] = None
        self._cupy_compress_stream: Optional[cupy.cuda.ExternalStream] = None

    def set_abo_params(self, abo_dtype: str, ratio: int) -> None:
        """Set ABO dtype and compression ratio.

        Called from StorageManager._ensure_staging_pool once the KV dtype
        is known from layout_desc.

        Args:
            abo_dtype: ABO dtype string (e.g. "bf16", "fp8e4m3").
            ratio: Compression ratio (20-32).
        """
        self._abo_dtype = abo_dtype
        self._ratio = ratio

    def _get_compress_stream(
        self,
    ) -> tuple[torch.cuda.Stream, cupy.cuda.ExternalStream]:
        """Get or create the dedicated compress CUDA stream + cupy wrapper."""
        if self._compress_stream is None:
            device = self._device or torch.cuda.current_device()
            self._compress_stream = torch.cuda.Stream(device=device)
            self._cupy_compress_stream = cupy.cuda.ExternalStream(
                self._compress_stream.cuda_stream
            )
        return self._compress_stream, self._cupy_compress_stream

    def submit_per_chunk_compress(
        self,
        d2h_event: torch.cuda.Event,
        obj_key: "ObjectKey",
        compressed_obj: "CompressedMemoryObj",
    ) -> torch.cuda.Event:
        """Submit per-chunk compression on the compress stream.

        Called in _cb_store_gpu_copy's for loop, after each chunk D2H.
        Does not block the main thread.

        Flow:
        1. cupy_stream.launch_host_func(compress_fn) — register sync compress
           as host callback on compress_stream
        2. compress_done_event.record(compress_stream) — mark compress done

        The host callback (compress_fn) runs in CUDA driver's callback thread:
        - d2h_event.synchronize() — CPU-side wait for D2H to finish
        - Executes sync compress (codec.compress with numpy buffers)
        - Marks compress done
        - Releases staging buffer

        The D2H wait is done inside the callback (CPU-side), not on the
        compress_stream (GPU-side), so the main thread is never blocked.

        Args:
            d2h_event: CUDA event recorded after this chunk's D2H.
            obj_key: Object key (for error logging).
            compressed_obj: CompressedMemoryObj instance.

        Returns:
            compress_done_event: CUDA event recorded after compress callback.
                Use current_stream.wait_event() for GPU-side sync.
        """
        assert self._abo_dtype is not None and self._ratio is not None, (
            "abo_dtype and ratio must be set via set_abo_params before compress"
        )
        assert compressed_obj._staging_tensor is not None, (
            "Compress requires staging_tensor != None"
        )
        assert compressed_obj.raw_data is not None, "Compress requires raw_data != None"

        compress_stream, cupy_compress_stream = self._get_compress_stream()

        # Capture references for the closure
        _d2h_event = d2h_event
        _obj_key = obj_key
        _compressed_obj = compressed_obj
        _on_compress_failed = self._on_compress_failed
        _codec = self._codec
        _abo_dtype = self._abo_dtype
        _ratio = self._ratio
        # Capture class-level staging pool reference for the closure
        # (avoid import inside callback)
        _cls_staging_pool = type(compressed_obj)._cls_staging_pool

        def _compress_callback(_arg):
            """Host callback: wait D2H → sync compress → mark done → release staging.

            Runs in CUDA driver's callback thread (GIL acquired by cupy).
            CPU-side d2h_event.synchronize() ensures D2H is done before compress.
            Args:
                _arg: User data from cupy launch_host_func (unused).
            """
            try:
                # CPU-side wait for D2H to finish (does not block main thread)
                # NOTE: Must use event.synchronize() (CPU-side blocking wait),
                # NOT stream.wait_event() — CUDA API calls are forbidden inside
                # host callbacks (CUDA driver restriction).
                _d2h_event.synchronize()

                staging_tensor = _compressed_obj._staging_tensor
                raw_data = _compressed_obj.raw_data

                if staging_tensor is None or raw_data is None:
                    logger.error(
                        "Compress aborted: staging_tensor or raw_data is None (key=%s)",
                        _obj_key,
                    )
                    _compressed_obj.mark_compress_failed()
                    if _on_compress_failed is not None:
                        try:
                            _on_compress_failed(_obj_key)
                        except Exception:
                            pass
                    return

                # Sync compress: codec.compress(dst, capacity, src, size, ratio, dtype)
                dst_np = raw_data.numpy()
                src_np = staging_tensor.numpy()
                result = _codec.compress(
                    dst_np,
                    dst_np.nbytes,
                    src_np,
                    src_np.nbytes,
                    _ratio,
                    _abo_dtype,
                )

                if not result.success:
                    logger.error(
                        "ABO compress failed (key=%s): %s",
                        _obj_key,
                        result.error_msg,
                    )
                    _compressed_obj.mark_compress_failed()
                    if _on_compress_failed is not None:
                        try:
                            _on_compress_failed(_obj_key)
                        except Exception:
                            pass
                    return

                # Mark compress done and restore meta.address
                _compressed_obj.mark_compress_done()
                _compressed_obj._restore_raw_data_meta()

                logger.debug(
                    "Compress done: key=%s, actual_compressed_size=%d bytes, "
                    "buffer_size=%d bytes",
                    _obj_key,
                    result.result_size,
                    _compressed_obj.raw_data.nbytes,
                )
            except Exception as e:
                logger.error("Compress callback failed (key=%s): %s", _obj_key, e)
                _compressed_obj.mark_compress_failed()
                if _on_compress_failed is not None:
                    try:
                        _on_compress_failed(_obj_key)
                    except Exception:
                        pass
            finally:
                # Always release staging buffer via mark_staging_released
                staging_to_release = _compressed_obj.mark_staging_released()
                if _cls_staging_pool is not None and staging_to_release is not None:
                    _cls_staging_pool.release(staging_to_release)

        # Register host callback on compress_stream
        cupy_compress_stream.launch_host_func(_compress_callback, None)

        # Record compress done event on compress_stream
        # (for GPU-side sync via current_stream.wait_event)
        compress_done_event = torch.cuda.Event()
        compress_done_event.record(compress_stream)

        # Set the CUDA event on compressed_obj for GPU-side sync
        compressed_obj.set_compress_event(compress_done_event)

        return compress_done_event

    def close(self) -> None:
        """Clean up resources.

        NOTE: Do NOT call compress_stream.synchronize() here — it would
        deadlock if any host callbacks are pending. The server process
        will terminate and clean up automatically.
        """
        pass
