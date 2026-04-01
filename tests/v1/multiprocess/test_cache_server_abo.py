# SPDX-License-Identifier: Apache-2.0
"""
End-to-end tests for ABO compression in MP paged KV cache mode (server.py).

Tests cover:
1. STORE path: D2H + ABO compression overlap (single & multi-chunk)
2. LOOKUP + RETRIEVE path: ABO decompression + H2D overlap (single & multi-chunk)
3. Full STORE -> LOOKUP -> RETRIEVE roundtrip with data correctness
4. Multi-batch STORE + RETRIEVE: staging buffer release and reuse

ABO is a lossy compression scheme, so data correctness is verified with
a relaxed tolerance (atol) instead of strict equality.

Both bfloat16 and float8_e4m3fn KV dtypes are tested via parametrization.
"""

# Standard
from dataclasses import dataclass
from typing import Generator
import multiprocessing as mp_lib
import os
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.v1.distributed.config import (
    ABOConfig,
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_response_class,
)
from lmcache.v1.multiprocess.server import run_cache_server

# Skip the entire module if abokvpress is not installed
abokvpress = pytest.importorskip("abokvpress", reason="abokvpress not installed")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
SERVER_HOST = "localhost"
CHUNK_SIZE = 256
CPU_BUFFER_SIZE = 5.0  # GB
ABO_STAGING_SIZE_GB = 16.0
ABO_CODEC = "huffman"
ABO_NUM_THREADS = 32
DEFAULT_TIMEOUT = 10.0
BLOCKS_PER_KEY = 4  # page_size=64, chunk_size=256 => 256/64=4 blocks per chunk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def _has_working_new_shared_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        buf = torch.empty(1024, device="cuda")
        shared = buf.untyped_storage()._share_cuda_()
        return shared is not None
    except Exception:
        return False


if not _has_working_new_shared_cuda():
    pytest.skip(
        "new_shared_cuda is not available or not working on this system",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Per-dtype configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DTypeConfig:
    """Configuration for a specific KV dtype test variant."""

    dtype: torch.dtype
    abo_ratio: int
    port: int
    atol: float
    key_offset: int  # Unique key offset to avoid collision between dtypes


DTYPE_CONFIGS = {
    "bf16": DTypeConfig(
        dtype=torch.bfloat16,
        abo_ratio=22,
        port=5601,
        atol=0.1,
        key_offset=50000,
    ),
    "fp8": DTypeConfig(
        dtype=torch.float8_e4m3fn,
        abo_ratio=26,
        port=5602,
        atol=0.15,
        key_offset=60000,
    ),
}


# =============================================================================
# Helper classes and functions
# =============================================================================


def _approx_equal(a: torch.Tensor, b: torch.Tensor, atol: float) -> bool:
    """Compare two tensors with tolerance, casting to float if needed.

    fp8 tensors do not support torch.allclose directly, so we cast to float.
    """
    return torch.allclose(a.float(), b.float(), atol=atol)


def initialize_kv_cache(
    device: torch.device,
    num_pages: int = 1024,
    num_layers: int = 32,
    page_size: int = 64,
    num_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """Initialize KV cache tensors on GPU for testing.

    torch.rand does not support fp8 dtypes, so we generate in bf16 then cast.
    """
    torch.random.manual_seed(42)
    tensors = []
    for _ in range(num_layers):
        t = torch.rand(
            (2, num_pages, page_size, num_heads, head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        if dtype != torch.bfloat16:
            t = t.to(dtype)
        tensors.append(t)
    return tensors


class ClientContext:
    """Client context that manages GPU KV cache tensors."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        num_pages: int = 1024,
        num_layers: int = 32,
        page_size: int = 64,
        num_heads: int = 8,
        head_size: int = 128,
    ):
        self.device = device
        self.num_pages = num_pages
        self.num_layers = num_layers
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype
        self.gpu_kv_caches = initialize_kv_cache(
            device, num_pages, num_layers, page_size, num_heads, head_size, dtype
        )

    def get_kv_cache(self) -> KVCache:
        return [CudaIPCWrapper(tensor) for tensor in self.gpu_kv_caches]


def create_cache_key(index: int, model: str = "testmodel") -> IPCCacheEngineKey:
    """Create a cache key for testing."""
    token_ids = [index] * CHUNK_SIZE
    return IPCCacheEngineKey.from_token_ids(
        model,
        1,
        0,
        token_ids,
        start=0,
        end=CHUNK_SIZE,
        request_id=f"abo_test_request_{index}",
    )


def lookup_all(
    client: MessageQueueClient,
    keys: list[IPCCacheEngineKey],
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """Lookup all keys individually and return total found count."""
    total = 0
    for key in keys:
        lookup_key = key.no_worker_id_version()
        job_id = client.submit_request(
            RequestType.LOOKUP,
            [lookup_key, 1],
            get_response_class(RequestType.LOOKUP),
        ).result(timeout=timeout)
        while True:
            result = client.submit_request(
                RequestType.QUERY_PREFETCH_STATUS,
                [job_id],
                get_response_class(RequestType.QUERY_PREFETCH_STATUS),
            ).result(timeout=timeout)
            if result is not None:
                total += result
                break
    return total


def store_keys(
    client: MessageQueueClient,
    keys: list[IPCCacheEngineKey],
    instance_id: int,
    gpu_block_ids: list[int],
    event: torch.cuda.Event,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """Store keys one at a time using the single-key API."""
    for i, key in enumerate(keys):
        start = i * BLOCKS_PER_KEY
        end = start + BLOCKS_PER_KEY
        block_ids = gpu_block_ids[start:end]
        future = client.submit_request(
            RequestType.STORE,
            [key, instance_id, block_ids, event.ipc_handle()],
            get_response_class(RequestType.STORE),
        )
        result = future.to_cuda_future().result(timeout=timeout)
        assert result is True, f"Store should succeed for key {i}"
    # Allow async finish_write (host callback on cupy_stream) to complete.
    # The CUDA event returned by store completes before the host callback
    # runs, so a short sleep is needed to ensure keys are visible for lookup.
    time.sleep(0.5)


def retrieve_keys(
    client: MessageQueueClient,
    keys: list[IPCCacheEngineKey],
    instance_id: int,
    gpu_block_ids: list[int],
    event: torch.cuda.Event,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[bool]:
    """Retrieve keys one at a time using the single-key API."""
    results = []
    for i, key in enumerate(keys):
        start = i * BLOCKS_PER_KEY
        end = start + BLOCKS_PER_KEY
        block_ids = gpu_block_ids[start:end]
        future = client.submit_request(
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event.ipc_handle(), 0],
            get_response_class(RequestType.RETRIEVE),
        )
        result = future.to_cuda_future().result(timeout=timeout)
        results.append(result)
    return results


# =============================================================================
# Server process runner (paged KV cache mode with ABO enabled)
# =============================================================================


def server_process_runner_abo(
    host: str,
    port: int,
    chunk_size: int,
    cpu_buffer_size: float,
    abo_ratio: int,
):
    """Entry point for the server process running MPCacheEngine with ABO."""
    from lmcache.v1.multiprocess.config import MPServerConfig

    mp_config = MPServerConfig(host=host, port=port, chunk_size=chunk_size)
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=int(cpu_buffer_size * 1024**3),
                use_lazy=True,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        abo_config=ABOConfig(
            enable=True,
            staging_size_gb=ABO_STAGING_SIZE_GB,
            ratio=abo_ratio,
            codec=ABO_CODEC,
            num_threads=ABO_NUM_THREADS,
        ),
    )
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


def _start_server(cfg: DTypeConfig) -> mp_lib.Process:
    """Start a server process for the given dtype config."""
    process = mp_lib.Process(
        target=server_process_runner_abo,
        args=(
            SERVER_HOST,
            cfg.port,
            CHUNK_SIZE,
            CPU_BUFFER_SIZE,
            cfg.abo_ratio,
        ),
        daemon=True,
    )
    process.start()
    return process


def _stop_server(process: mp_lib.Process) -> None:
    """Gracefully stop a server process."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def server_processes() -> Generator[dict[str, mp_lib.Process], None, None]:
    """Start ABO-enabled servers for all dtype configs (one per dtype)."""
    mp_lib.set_start_method("spawn", force=True)
    procs: dict[str, mp_lib.Process] = {}
    for name, cfg in DTYPE_CONFIGS.items():
        procs[name] = _start_server(cfg)
    time.sleep(3)
    yield procs
    for proc in procs.values():
        _stop_server(proc)


@pytest.fixture(scope="module")
def zmq_context() -> Generator[zmq.Context, None, None]:
    context = zmq.Context.instance()
    yield context


@pytest.fixture(
    scope="function",
    params=list(DTYPE_CONFIGS.keys()),
    ids=list(DTYPE_CONFIGS.keys()),
)
def dtype_cfg(request) -> DTypeConfig:
    """Parametrized fixture that yields each DTypeConfig."""
    return DTYPE_CONFIGS[request.param]


@pytest.fixture(scope="function")
def client(
    server_processes: dict[str, mp_lib.Process],
    zmq_context: zmq.Context,
    dtype_cfg: DTypeConfig,
) -> Generator[MessageQueueClient, None, None]:
    url = f"tcp://{SERVER_HOST}:{dtype_cfg.port}"
    c = MessageQueueClient(server_url=url, context=zmq_context)
    yield c
    c.close()


@pytest.fixture(scope="function")
def client_context(dtype_cfg: DTypeConfig) -> Generator[ClientContext, None, None]:
    device = torch.device("cuda:0")
    ctx = ClientContext(device=device, dtype=dtype_cfg.dtype)
    yield ctx
    del ctx.gpu_kv_caches
    torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def registered_instance(
    client: MessageQueueClient, client_context: ClientContext
) -> Generator[int, None, None]:
    """Register a KV cache instance; unregister and clear after the test."""
    instance_id = os.getpid() + 6000  # Offset to avoid collision

    future = client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [instance_id, client_context.get_kv_cache(), "testmodel", 1, {}],
        get_response_class(RequestType.REGISTER_KV_CACHE),
    )
    assert future.result(timeout=DEFAULT_TIMEOUT) is None

    yield instance_id

    try:
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)
        client.submit_request(
            RequestType.UNREGISTER_KV_CACHE,
            [instance_id],
            get_response_class(RequestType.UNREGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        print(f"Error during unregister: {e}")


# =============================================================================
# Tests
# =============================================================================


def test_abo_server_running(
    server_processes: dict[str, mp_lib.Process],
):
    """ABO-enabled paged KV cache server processes should be alive."""
    for name, proc in server_processes.items():
        assert proc.is_alive(), f"Server for {name} should be alive"


def test_abo_store_single_chunk(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
    dtype_cfg: DTypeConfig,
):
    """
    Store one chunk with ABO enabled.
    D2H + ABO compression pipeline should succeed and the chunk
    should be discoverable via lookup.
    """
    key = create_cache_key(dtype_cfg.key_offset)
    gpu_block_ids = list(range(0, BLOCKS_PER_KEY))

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_keys(client, [key], registered_instance, gpu_block_ids, event)

    # Verify the stored chunk is discoverable via lookup
    found = lookup_all(client, [key])
    assert found == 1, "Stored chunk should be found via lookup"


def test_abo_store_multiple_chunks(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
    dtype_cfg: DTypeConfig,
):
    """
    Store 4 chunks with ABO enabled.
    D2H and ABO compression should overlap via CUDA stream scheduling.
    All chunks should be discoverable via lookup.
    """
    num_keys = 4
    base = dtype_cfg.key_offset + 1000
    keys = [create_cache_key(base + i) for i in range(num_keys)]
    gpu_block_ids = list(range(0, BLOCKS_PER_KEY * num_keys))

    event = torch.cuda.Event(interprocess=True)
    event.record()

    store_keys(client, keys, registered_instance, gpu_block_ids, event)

    found = lookup_all(client, keys)
    assert found == num_keys, "All stored chunks should be found via lookup"


def test_abo_store_lookup_retrieve_single_chunk(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
    dtype_cfg: DTypeConfig,
):
    """
    Store one chunk, lookup, then retrieve to a different GPU location.
    Verify data correctness with ABO lossy tolerance.

    This tests the full ABO decompress + H2D overlap path for a single chunk.
    """
    # Fill source pages with known value
    src_value = 0.25
    for layer in range(client_context.num_layers):
        client_context.gpu_kv_caches[layer][:, 0:BLOCKS_PER_KEY] = src_value

    base = dtype_cfg.key_offset + 2000
    key = create_cache_key(base)
    src_block_ids = list(range(0, BLOCKS_PER_KEY))

    event = torch.cuda.Event(interprocess=True)
    event.record()

    # STORE
    store_keys(client, [key], registered_instance, src_block_ids, event)

    # LOOKUP
    found = lookup_all(client, [key])
    assert found == 1

    # RETRIEVE to different location
    dst_offset = 40  # pages
    dst_block_ids = list(range(dst_offset, dst_offset + BLOCKS_PER_KEY))

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    results = retrieve_keys(client, [key], registered_instance, dst_block_ids, event2)
    assert results == [True], "Retrieve should succeed"

    torch.cuda.synchronize()

    # Verify data correctness with ABO lossy tolerance
    atol = dtype_cfg.atol
    for layer in range(client_context.num_layers):
        src_tensor = client_context.gpu_kv_caches[layer][:, 0:BLOCKS_PER_KEY]
        dst_tensor = client_context.gpu_kv_caches[layer][
            :, dst_offset : dst_offset + BLOCKS_PER_KEY
        ]
        assert _approx_equal(src_tensor, dst_tensor, atol=atol), (
            f"Layer {layer}: retrieved data should approximately match source "
            f"(atol={atol})"
        )


def test_abo_store_lookup_retrieve_multiple_chunks(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
    dtype_cfg: DTypeConfig,
):
    """
    Store 3 chunks with distinct fill values, lookup all, then retrieve all
    to different GPU locations. Verify per-chunk data correctness.

    This tests multi-chunk ABO decompress + H2D overlap, where decompress
    of chunk N+1 overlaps with H2D of chunk N.
    """
    num_keys = 3
    fill_values = [0.2, 0.5, 0.8]

    # Fill source pages with distinct values per chunk
    for i in range(num_keys):
        start_page = i * BLOCKS_PER_KEY
        end_page = start_page + BLOCKS_PER_KEY
        for layer in range(client_context.num_layers):
            client_context.gpu_kv_caches[layer][:, start_page:end_page] = fill_values[i]

    base = dtype_cfg.key_offset + 3000
    keys = [create_cache_key(base + i) for i in range(num_keys)]
    src_block_ids = list(range(0, BLOCKS_PER_KEY * num_keys))

    event = torch.cuda.Event(interprocess=True)
    event.record()

    # STORE
    store_keys(client, keys, registered_instance, src_block_ids, event)

    # LOOKUP
    found = lookup_all(client, keys)
    assert found == num_keys

    # RETRIEVE to different location
    dst_offset = 50  # pages
    dst_block_ids = list(range(dst_offset, dst_offset + BLOCKS_PER_KEY * num_keys))

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    results = retrieve_keys(client, keys, registered_instance, dst_block_ids, event2)
    assert all(results), "All retrieves should succeed"

    torch.cuda.synchronize()

    # Verify per-chunk data correctness
    atol = dtype_cfg.atol
    for i in range(num_keys):
        src_start = i * BLOCKS_PER_KEY
        dst_start = dst_offset + i * BLOCKS_PER_KEY
        for layer in range(client_context.num_layers):
            src_tensor = client_context.gpu_kv_caches[layer][
                :, src_start : src_start + BLOCKS_PER_KEY
            ]
            dst_tensor = client_context.gpu_kv_caches[layer][
                :, dst_start : dst_start + BLOCKS_PER_KEY
            ]
            assert _approx_equal(src_tensor, dst_tensor, atol=atol), (
                f"Chunk {i}, layer {layer}: retrieved data should approximately "
                f"match source (atol={atol})"
            )


def test_abo_full_roundtrip(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
    dtype_cfg: DTypeConfig,
):
    """
    Complete STORE -> LOOKUP -> RETRIEVE roundtrip with ABO compression.
    Verifies the entire ABO compress/decompress pipeline works end-to-end
    in MP paged KV cache mode, including CUDA stream synchronization.
    """
    src_value = 0.375

    # Fill source pages
    for layer in range(client_context.num_layers):
        client_context.gpu_kv_caches[layer][:, 0:BLOCKS_PER_KEY] = src_value

    base = dtype_cfg.key_offset + 4000
    key = create_cache_key(base)
    src_block_ids = list(range(0, BLOCKS_PER_KEY))

    # --- STORE ---
    event_store = torch.cuda.Event(interprocess=True)
    event_store.record()
    store_keys(client, [key], registered_instance, src_block_ids, event_store)

    # --- LOOKUP ---
    found = lookup_all(client, [key])
    assert found == 1, "LOOKUP should find exactly 1 chunk"

    # --- RETRIEVE ---
    dst_offset = 60
    dst_block_ids = list(range(dst_offset, dst_offset + BLOCKS_PER_KEY))

    # Zero out destination before retrieve
    for layer in range(client_context.num_layers):
        client_context.gpu_kv_caches[layer][
            :, dst_offset : dst_offset + BLOCKS_PER_KEY
        ] = 0.0

    event_retrieve = torch.cuda.Event(interprocess=True)
    event_retrieve.record()
    results = retrieve_keys(
        client, [key], registered_instance, dst_block_ids, event_retrieve
    )
    assert results == [True], "RETRIEVE should succeed"

    torch.cuda.synchronize()

    # Verify data correctness
    atol = dtype_cfg.atol
    for layer in range(client_context.num_layers):
        src_tensor = client_context.gpu_kv_caches[layer][:, 0:BLOCKS_PER_KEY]
        dst_tensor = client_context.gpu_kv_caches[layer][
            :, dst_offset : dst_offset + BLOCKS_PER_KEY
        ]
        assert _approx_equal(src_tensor, dst_tensor, atol=atol), (
            f"Roundtrip layer {layer}: data mismatch (atol={atol})"
        )

    # Verify no data corruption beyond the destination
    beyond_start = dst_offset + BLOCKS_PER_KEY
    for layer in range(client_context.num_layers):
        beyond = client_context.gpu_kv_caches[layer][
            :, beyond_start : beyond_start + BLOCKS_PER_KEY
        ]
        expected = torch.full_like(beyond, src_value)
        assert not _approx_equal(beyond, expected, atol=atol), (
            "Region beyond destination should not contain the stored value"
        )


def test_abo_multi_batch_store_retrieve(
    client: MessageQueueClient,
    client_context: ClientContext,
    registered_instance: int,
    dtype_cfg: DTypeConfig,
):
    """
    Store 4 batches of 4 keys each, then retrieve all 4 batches.
    Verifies that staging buffers are correctly released and reused
    across multiple store/retrieve cycles.

    This exercises the full compress/decompress pipeline repeatedly,
    ensuring no staging buffer leaks.
    """
    num_batches = 4
    keys_per_batch = 4
    base = dtype_cfg.key_offset + 5000

    # Fill source pages with distinct values per batch
    for batch_idx in range(num_batches):
        fill_value = (batch_idx + 1) / num_batches
        start_page = batch_idx * keys_per_batch * BLOCKS_PER_KEY
        end_page = start_page + keys_per_batch * BLOCKS_PER_KEY
        for layer in range(client_context.num_layers):
            client_context.gpu_kv_caches[layer][:, start_page:end_page] = fill_value

    # STORE in batches
    for batch_idx in range(num_batches):
        keys = [
            create_cache_key(base + batch_idx * keys_per_batch + i)
            for i in range(keys_per_batch)
        ]
        start_block = batch_idx * keys_per_batch * BLOCKS_PER_KEY
        block_ids = list(
            range(start_block, start_block + keys_per_batch * BLOCKS_PER_KEY)
        )
        event = torch.cuda.Event(interprocess=True)
        event.record()
        store_keys(client, keys, registered_instance, block_ids, event)

    # LOOKUP all
    all_keys = [
        create_cache_key(base + batch_idx * keys_per_batch + i)
        for batch_idx in range(num_batches)
        for i in range(keys_per_batch)
    ]
    found = lookup_all(client, all_keys)
    assert found == num_batches * keys_per_batch, "All stored keys should be found"

    # RETRIEVE in batches to a different GPU region
    retrieve_offset = 32  # chunks offset (in pages: 32 * BLOCKS_PER_KEY)
    retrieve_page_offset = retrieve_offset * BLOCKS_PER_KEY

    event2 = torch.cuda.Event(interprocess=True)
    event2.record()

    for batch_idx in range(num_batches):
        keys = [
            create_cache_key(base + batch_idx * keys_per_batch + i)
            for i in range(keys_per_batch)
        ]
        start_block = retrieve_page_offset + batch_idx * keys_per_batch * BLOCKS_PER_KEY
        block_ids = list(
            range(start_block, start_block + keys_per_batch * BLOCKS_PER_KEY)
        )
        results = retrieve_keys(client, keys, registered_instance, block_ids, event2)
        assert all(results), f"Batch {batch_idx}: all retrieves should succeed"

    torch.cuda.synchronize()

    # Verify per-batch data correctness
    atol = dtype_cfg.atol
    for batch_idx in range(num_batches):
        expected_value = (batch_idx + 1) / num_batches
        start_page = retrieve_page_offset + batch_idx * keys_per_batch * BLOCKS_PER_KEY
        end_page = start_page + keys_per_batch * BLOCKS_PER_KEY
        for layer in range(client_context.num_layers):
            retrieved = client_context.gpu_kv_caches[layer][:, start_page:end_page]
            expected = torch.full_like(retrieved, expected_value)
            assert _approx_equal(retrieved, expected, atol=atol), (
                f"Batch {batch_idx}, layer {layer}: data mismatch "
                f"(expected ~{expected_value}, atol={atol})"
            )
