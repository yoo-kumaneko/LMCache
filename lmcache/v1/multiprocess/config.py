# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the multiprocess (ZMQ) server and HTTP frontend.
"""

# Standard
from dataclasses import dataclass, field
import argparse

# First Party
from lmcache.v1.multiprocess.chunk_hash_logger import ChunkHashLogConfig


@dataclass
class MPServerConfig:
    """Configuration for the ZMQ-based multiprocess cache server."""

    host: str = "localhost"
    """ZMQ server host."""

    port: int = 5555
    """ZMQ server port."""

    chunk_size: int = 256
    """Chunk size for KV cache operations."""

    max_workers: int = 1
    """Base number of worker threads. Sets default for both GPU and CPU pools."""

    max_gpu_workers: int = 1
    """Worker threads for the GPU affinity pool (STORE/RETRIEVE).
    Resolved from --max-gpu-workers or --max-workers."""

    max_cpu_workers: int = 1
    """Worker threads for the normal (CPU) pool (LOOKUP, END_SESSION, etc.).
    Resolved from --max-cpu-workers or --max-workers."""

    hash_algorithm: str = "blake3"
    """Hash algorithm for token-based operations (builtin, sha256_cbor, blake3)."""

    engine_type: str = "default"
    """Cache engine backend type
    ('default' for MPCacheEngine, 'blend' for BlendEngineV2).
    """

    runtime_plugin_locations: list[str] = field(default_factory=list)
    """Paths to runtime plugin scripts or directories."""

    chunk_hash_log: ChunkHashLogConfig = field(default_factory=ChunkHashLogConfig)
    """Configuration for chunk hash file logging."""


DEFAULT_MP_SERVER_CONFIG = MPServerConfig()


@dataclass
class HTTPFrontendConfig:
    """Configuration for the HTTP frontend (uvicorn/FastAPI)."""

    http_host: str = "0.0.0.0"
    """HTTP server host."""

    http_port: int = 8080
    """HTTP server port."""


DEFAULT_HTTP_FRONTEND_CONFIG = HTTPFrontendConfig()


def add_mp_server_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add MP server configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with MP server arguments added.
    """
    mp_group = parser.add_argument_group(
        "MP Server", "Configuration for the ZMQ multiprocess cache server"
    )
    mp_group.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the ZMQ server. Default is localhost.",
    )
    mp_group.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to bind the ZMQ server. Default is 5555.",
    )
    mp_group.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for KV cache operations. Default is 256.",
    )
    mp_group.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Base number of worker threads for both GPU and CPU pools. "
        "Default is 1. Can be overridden per-pool with "
        "--max-gpu-workers and --max-cpu-workers.",
    )
    mp_group.add_argument(
        "--max-gpu-workers",
        type=int,
        default=None,
        help="Worker threads for the GPU affinity pool (STORE/RETRIEVE). "
        "Defaults to --max-workers if not specified.",
    )
    mp_group.add_argument(
        "--max-cpu-workers",
        type=int,
        default=None,
        help="Worker threads for the normal CPU pool (LOOKUP, etc.). "
        "Defaults to --max-workers if not specified.",
    )
    mp_group.add_argument(
        "--hash-algorithm",
        type=str,
        default="blake3",
        help="Hash algorithm for token-based operations "
        "(builtin, sha256_cbor, blake3). Default is blake3.",
    )
    mp_group.add_argument(
        "--engine-type",
        type=str,
        default="default",
        choices=["default", "blend"],
        help="Cache engine backend type. 'default' uses MPCacheEngine, "
        "'blend' uses BlendEngineV2 for cross-request KV reuse. "
        "Default is 'default'.",
    )
    mp_group.add_argument(
        "--runtime-plugin-locations",
        type=str,
        nargs="*",
        default=[],
        help="Paths to runtime plugin scripts or "
        "directories to launch alongside the server.",
    )

    # Chunk hash logging config
    log_group = parser.add_argument_group(
        "Chunk Hash Logging",
        "Configuration for chunk hash file logging (offline analysis)",
    )
    log_group.add_argument(
        "--chunk-hash-log-dir",
        type=str,
        default="",
        help="Directory to write chunk hash JSONL files for offline analysis. "
        "Empty string (default) disables logging.",
    )
    log_group.add_argument(
        "--chunk-hash-log-rotation-interval",
        type=int,
        default=6 * 3600,
        help="Time interval in seconds before rotating to a new log file. "
        "Default is 21600 (6 hours).",
    )
    log_group.add_argument(
        "--chunk-hash-log-rotation-max-size",
        type=int,
        default=100 * 1024 * 1024,
        help="Max file size in bytes before rotating even if the time "
        "interval has not elapsed. Default is 100MB (104857600).",
    )
    log_group.add_argument(
        "--chunk-hash-log-max-files",
        type=int,
        default=100,
        help="Max number of chunk hash log files to keep. "
        "Oldest files are deleted when this limit is exceeded. Default is 100.",
    )
    return parser


def parse_args_to_mp_server_config(
    args: argparse.Namespace,
) -> MPServerConfig:
    """
    Convert parsed command line arguments to an MPServerConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        MPServerConfig: The configuration object.
    """
    base = args.max_workers
    max_gpu = args.max_gpu_workers if args.max_gpu_workers is not None else base
    max_cpu = args.max_cpu_workers if args.max_cpu_workers is not None else base
    return MPServerConfig(
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        max_workers=base,
        max_gpu_workers=max_gpu,
        max_cpu_workers=max_cpu,
        hash_algorithm=args.hash_algorithm,
        engine_type=args.engine_type,
        runtime_plugin_locations=(args.runtime_plugin_locations or []),
        chunk_hash_log=ChunkHashLogConfig(
            output_dir=args.chunk_hash_log_dir,
            rotation_interval_sec=args.chunk_hash_log_rotation_interval,
            rotation_max_size=args.chunk_hash_log_rotation_max_size,
            max_files=args.chunk_hash_log_max_files,
        ),
    )


def add_http_frontend_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add HTTP frontend configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with HTTP frontend arguments added.
    """
    http_group = parser.add_argument_group(
        "HTTP Frontend", "Configuration for the HTTP frontend server"
    )
    http_group.add_argument(
        "--http-host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the HTTP server. Default is 0.0.0.0.",
    )
    http_group.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port to bind the HTTP server. Default is 8080.",
    )
    return parser


def parse_args_to_http_frontend_config(
    args: argparse.Namespace,
) -> HTTPFrontendConfig:
    """
    Convert parsed command line arguments to an HTTPFrontendConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        HTTPFrontendConfig: The configuration object.
    """
    return HTTPFrontendConfig(
        http_host=args.http_host,
        http_port=args.http_port,
    )
