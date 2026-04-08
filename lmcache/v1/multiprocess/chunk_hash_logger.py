# SPDX-License-Identifier: Apache-2.0

"""
ChunkHashLogger: Async JSONL writer for chunk hashes computed during lookup.

Records chunk hashes to rotating JSONL files for offline analysis.
Designed for the multiprocess server's lookup path where hashes are
already available as bytes from TokenHasher.
"""

# Standard
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import io
import json
import queue
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Pattern for discovering existing log files on disk.
_LOG_FILE_GLOB = "chunk_hashes_*.jsonl"


@dataclass
class ChunkHashLogConfig:
    """Configuration for chunk hash file logging.

    When ``output_dir`` is non-empty, chunk hashes computed during
    lookup are written to rotating JSONL files for offline analysis.
    """

    output_dir: str = ""
    """Directory to write chunk hash JSONL files.
    Empty string disables logging."""

    rotation_interval_sec: int = 6 * 3600
    """Time interval in seconds before rotating to a new file
    (default 6 hours)."""

    rotation_max_size: int = 100 * 1024 * 1024
    """Max file size in bytes before rotating even if the time
    interval has not elapsed (default 100MB)."""

    max_files: int = 100
    """Max number of log files to keep before deleting oldest."""

    @property
    def enabled(self) -> bool:
        """Whether chunk hash logging is enabled."""
        return bool(self.output_dir)


def _format_timestamp(ts: float) -> str:
    """Format a unix timestamp as a compact datetime string.

    Example: 20260401_143025
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")


class ChunkHashLogger:
    """Async JSONL writer for chunk hashes computed during lookup.

    Writes chunk hash data to rotating JSONL files in a background
    thread to avoid adding I/O latency to the lookup hot path.

    Files rotate when either the time interval or file size limit is
    reached, whichever comes first. File names include a
    human-readable timestamp, e.g.::

        chunk_hashes_20260401_143025_000003.jsonl

    Each JSONL line has the format::

        {"timestamp": 1711929600.123, "request_id": "req-abc",
         "model_name": "DeepSeek-V3",
         "chunk_hashes": ["0xab...", ...]}

    Args:
        config: ChunkHashLogConfig with output_dir, rotation
            settings, and max_files.
    """

    _QUEUE_CAPACITY = 100000

    def __init__(
        self,
        config: ChunkHashLogConfig,
    ):
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rotation_interval_sec = config.rotation_interval_sec
        self.rotation_max_size = config.rotation_max_size
        self.max_files = config.max_files

        # File state (only accessed by worker thread)
        self._current_file_size = 0
        self._current_file: Optional[Path] = None
        self._current_handle: Optional[io.TextIOWrapper] = None
        self._current_file_opened_at: float = 0.0

        # Discover existing log files so max_files limit accounts
        # for files from previous runs.
        self._file_list: list[Path] = sorted(
            self.output_dir.glob(_LOG_FILE_GLOB),
            key=lambda p: p.stat().st_mtime,
        )
        self._file_count = len(self._file_list)

        # Async queue and worker
        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_CAPACITY)
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="ChunkHashLoggerWorker",
        )
        self._worker.start()
        logger.info(
            "ChunkHashLogger started: output_dir=%s, "
            "rotation_interval=%ds, "
            "rotation_max_size=%d, max_files=%d, "
            "existing_files=%d",
            self.output_dir,
            self.rotation_interval_sec,
            self.rotation_max_size,
            self.max_files,
            len(self._file_list),
        )

    def log(
        self,
        request_id: str,
        chunk_hashes: list[bytes],
        model_name: str = "",
    ) -> None:
        """Record chunk hashes for a lookup request (non-blocking).

        Args:
            request_id: The request ID from the lookup.
            chunk_hashes: List of chunk hash bytes from TokenHasher.
            model_name: Model name associated with this lookup.
        """
        if self._shutdown:
            return
        entry = (time.time(), request_id, chunk_hashes, model_name)
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            logger.warning(
                "ChunkHashLogger queue full, dropping entry for %s",
                request_id,
            )

    def close(self) -> None:
        """Shutdown the background worker and close file handles."""
        self._shutdown = True
        # Sentinel to unblock the worker
        try:
            self._queue.put(None, block=False)
        except queue.Full:
            pass

        self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            logger.warning("ChunkHashLogger worker did not stop gracefully")

        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None
        logger.info("ChunkHashLogger closed")

    # ---- internals (worker thread only) ----

    def _worker_loop(self) -> None:
        """Background worker that consumes the queue."""
        while not self._shutdown:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if item is None:
                    break
                self._write_entry(item)
            except Exception:
                logger.exception("ChunkHashLogger worker error")
            finally:
                self._queue.task_done()

        # Drain remaining items
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                if item is not None:
                    self._write_entry(item)
            except Exception:
                logger.exception("ChunkHashLogger worker drain error")
            finally:
                self._queue.task_done()

    def _needs_rotation(self, now: float) -> bool:
        """Check if the current file needs rotation."""
        if self._current_handle is None:
            return True
        # Time-based rotation
        elapsed = now - self._current_file_opened_at
        if elapsed >= self.rotation_interval_sec:
            return True
        # Size-based rotation
        if self._current_file_size >= self.rotation_max_size:
            return True
        return False

    def _write_entry(self, entry: tuple) -> None:
        """Write a single entry to the current JSONL file."""
        timestamp, request_id, chunk_hashes, model_name = entry

        if self._needs_rotation(timestamp):
            self._rotate_file(timestamp)

        data = {
            "timestamp": timestamp,
            "request_id": request_id,
            "model_name": model_name,
            "chunk_hashes": [
                "0x" + h.hex() if isinstance(h, bytes) else hex(h) for h in chunk_hashes
            ],
        }
        line = json.dumps(data) + "\n"
        if self._current_handle is not None:
            self._current_handle.write(line)
            self._current_handle.flush()
            self._current_file_size += len(line)

    def _rotate_file(self, now: float) -> None:
        """Close current file and open a new one."""
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None

        time_str = _format_timestamp(now)
        self._current_file = (
            self.output_dir / f"chunk_hashes_{time_str}_{self._file_count:06d}.jsonl"
        )
        self._current_handle = open(self._current_file, "w", encoding="utf-8")
        self._current_file_opened_at = now
        self._current_file_size = 0
        self._file_count += 1
        self._file_list.append(self._current_file)

        # Enforce max file count
        while len(self._file_list) > self.max_files:
            oldest = self._file_list.pop(0)
            try:
                if oldest.exists():
                    oldest.unlink()
            except Exception as e:
                logger.error(
                    "Failed to delete old chunk hash file %s: %s",
                    oldest,
                    e,
                )
