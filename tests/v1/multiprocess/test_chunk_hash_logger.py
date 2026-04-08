# SPDX-License-Identifier: Apache-2.0
"""Tests for ChunkHashLogger and ChunkHashLogConfig."""

# Standard
from pathlib import Path
import json
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.chunk_hash_logger import (
    ChunkHashLogConfig,
    ChunkHashLogger,
    _format_timestamp,
)


class TestChunkHashLogConfig:
    def test_disabled_by_default(self) -> None:
        config = ChunkHashLogConfig()
        assert not config.enabled
        assert config.output_dir == ""

    def test_enabled_when_output_dir_set(self) -> None:
        config = ChunkHashLogConfig(output_dir="/tmp/test")
        assert config.enabled

    def test_defaults(self) -> None:
        config = ChunkHashLogConfig()
        assert config.rotation_interval_sec == 6 * 3600
        assert config.rotation_max_size == 100 * 1024 * 1024
        assert config.max_files == 100


class TestFormatTimestamp:
    def test_known_timestamp(self) -> None:
        # 2026-04-01 14:30:25 UTC
        ts = 1775053825.0
        result = _format_timestamp(ts)
        assert result == "20260401_143025"

    def test_returns_string(self) -> None:
        result = _format_timestamp(time.time())
        assert isinstance(result, str)
        assert len(result) == 15  # YYYYMMDD_HHMMSS


class TestChunkHashLogger:
    @pytest.fixture
    def log_dir(self, tmp_path: Path) -> Path:
        """Provide a temporary log directory."""
        return tmp_path / "chunk_hashes"

    @pytest.fixture
    def config(self, log_dir: Path) -> ChunkHashLogConfig:
        """Provide a config with small limits for testing."""
        return ChunkHashLogConfig(
            output_dir=str(log_dir),
            rotation_interval_sec=3600,
            rotation_max_size=100 * 1024 * 1024,
            max_files=100,
        )

    def test_creates_output_dir(self, config: ChunkHashLogConfig) -> None:
        logger = ChunkHashLogger(config)
        assert Path(config.output_dir).is_dir()
        logger.close()

    def test_log_and_close_writes_file(
        self, config: ChunkHashLogConfig, log_dir: Path
    ) -> None:
        logger = ChunkHashLogger(config)
        logger.log("req-001", [b"\xab\xcd", b"\x12\x34"], "test-model")
        logger.close()

        files = list(log_dir.glob("chunk_hashes_*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["request_id"] == "req-001"
        assert data["model_name"] == "test-model"
        assert data["chunk_hashes"] == ["0xabcd", "0x1234"]
        assert "timestamp" in data

    def test_multiple_entries(self, config: ChunkHashLogConfig, log_dir: Path) -> None:
        logger = ChunkHashLogger(config)
        for i in range(10):
            logger.log(f"req-{i:03d}", [b"\xaa"], f"model-{i}")
        logger.close()

        files = list(log_dir.glob("chunk_hashes_*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 10

    def test_disabled_logger_not_created(self) -> None:
        config = ChunkHashLogConfig()  # output_dir=""
        assert not config.enabled
        # Just verify the config property works; the server checks
        # config.enabled before creating the logger.

    def test_close_is_idempotent(self, config: ChunkHashLogConfig) -> None:
        logger = ChunkHashLogger(config)
        logger.log("req-001", [b"\xaa"], "model")
        logger.close()
        # Second close should not raise
        logger.close()

    def test_log_after_close_is_ignored(
        self, config: ChunkHashLogConfig, log_dir: Path
    ) -> None:
        logger = ChunkHashLogger(config)
        logger.log("req-001", [b"\xaa"], "model")
        logger.close()

        # This should be silently dropped
        logger.log("req-002", [b"\xbb"], "model")

        files = list(log_dir.glob("chunk_hashes_*.jsonl"))
        total_lines = 0
        for f in files:
            total_lines += len(f.read_text(encoding="utf-8").strip().split("\n"))
        assert total_lines == 1

    def test_size_based_rotation(self, log_dir: Path) -> None:
        config = ChunkHashLogConfig(
            output_dir=str(log_dir),
            rotation_interval_sec=999999,  # won't trigger
            rotation_max_size=200,  # very small, triggers quickly
            max_files=100,
        )
        logger = ChunkHashLogger(config)
        for i in range(20):
            logger.log(f"req-{i:03d}", [b"\xaa\xbb\xcc\xdd"], "model")
        logger.close()

        files = list(log_dir.glob("chunk_hashes_*.jsonl"))
        assert len(files) > 1, "Should have rotated due to size"

    def test_max_files_limit(self, log_dir: Path) -> None:
        config = ChunkHashLogConfig(
            output_dir=str(log_dir),
            rotation_interval_sec=999999,
            rotation_max_size=50,  # tiny, forces many rotations
            max_files=3,
        )
        logger = ChunkHashLogger(config)
        for i in range(30):
            logger.log(f"req-{i:03d}", [b"\xaa\xbb\xcc\xdd"], "model")
        logger.close()

        files = list(log_dir.glob("chunk_hashes_*.jsonl"))
        assert len(files) <= 3

    def test_existing_files_discovered_on_init(self, log_dir: Path) -> None:
        """Files from previous runs are counted toward max_files."""
        log_dir.mkdir(parents=True, exist_ok=True)
        # Create 3 pre-existing files
        for i in range(3):
            f = log_dir / f"chunk_hashes_20260101_000000_{i:06d}.jsonl"
            f.write_text("{}\n", encoding="utf-8")
            # Stagger mtime so sorting is deterministic
            time.sleep(0.01)

        config = ChunkHashLogConfig(
            output_dir=str(log_dir),
            rotation_interval_sec=999999,
            rotation_max_size=50,
            max_files=4,  # 3 existing + 1 new = at limit
        )
        logger = ChunkHashLogger(config)
        # Write enough to trigger at least 2 rotations
        for i in range(20):
            logger.log(f"req-{i:03d}", [b"\xaa\xbb\xcc\xdd"], "model")
        logger.close()

        files = list(log_dir.glob("chunk_hashes_*.jsonl"))
        assert len(files) <= 4

    def test_json_output_is_valid(
        self, config: ChunkHashLogConfig, log_dir: Path
    ) -> None:
        logger = ChunkHashLogger(config)
        logger.log("req-001", [b"\xff"], "model-a")
        logger.log("req-002", [b"\x00" * 16], "model-b")
        logger.close()

        files = list(log_dir.glob("chunk_hashes_*.jsonl"))
        for f in files:
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                data = json.loads(line)  # Should not raise
                assert isinstance(data["timestamp"], float)
                assert isinstance(data["request_id"], str)
                assert isinstance(data["model_name"], str)
                assert isinstance(data["chunk_hashes"], list)

    def test_integer_hashes_handled(
        self, config: ChunkHashLogConfig, log_dir: Path
    ) -> None:
        """Verify integer chunk hashes (not bytes) are also handled."""
        logger = ChunkHashLogger(config)
        logger.log("req-001", [255, 65536], "model")  # type: ignore[list-item]
        logger.close()

        files = list(log_dir.glob("chunk_hashes_*.jsonl"))
        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        data = json.loads(lines[0])
        assert data["chunk_hashes"] == ["0xff", "0x10000"]
