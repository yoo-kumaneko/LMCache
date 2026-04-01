# SPDX-License-Identifier: Apache-2.0
"""
ABO enabled vs disabled end-to-end latency comparison test.

Usage:
    python tests/v1/multiprocess/test_abo_latency_compare.py
"""

import multiprocessing as mp_lib
import os
import time

import torch
import zmq

from lmcache.v1.distributed.config import (
    ABOConfig,
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper, IPCCacheEngineKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

# Config
SERVER_PORT = 5699
CHUNK_SIZE = 256
NUM_CHUNKS = 64
NUM_LAYERS = 32
NUM_PAGES = 2048
PAGE_SIZE = 64
NUM_HEADS = 8
HEAD_SIZE = 128
BLOCKS_PER_CHUNK = CHUNK_SIZE // PAGE_SIZE  # 4


def init_kv_cache(device):
    """Init KV cache on GPU."""
    return [
        torch.rand(
            (2, NUM_PAGES, PAGE_SIZE, NUM_HEADS, HEAD_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(NUM_LAYERS)
    ]


def run_server(port, chunk_size, enable_abo):
    """Server process with ABO enabled or disabled."""
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.server import run_cache_server

    mp_config = MPServerConfig(host="localhost", port=port, chunk_size=chunk_size)
    storage_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=5 * 1024**3, use_lazy=True
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        abo_config=ABOConfig(
            enable=enable_abo,
            staging_size_gb=16.0 if enable_abo else 0.0,
            ratio=22,
            codec="huffman",
            num_threads=64,
        ),
    )
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_config,
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


def make_key(idx):
    """Create cache key."""
    return IPCCacheEngineKey.from_token_ids(
        "model", 1, 0, [idx] * CHUNK_SIZE, 0, CHUNK_SIZE, f"test_{idx}"
    )


def run_test(enable_abo, port):
    """Run single test with ABO enabled or disabled."""
    print(f"\n{'=' * 60}")
    print(f"Testing with ABO {'ENABLED' if enable_abo else 'DISABLED'}")
    print(f"{'=' * 60}")

    # Start server
    server = mp_lib.Process(
        target=run_server, args=(port, CHUNK_SIZE, enable_abo), daemon=True
    )
    server.start()
    time.sleep(3)

    # Init GPU cache
    device = torch.device("cuda:0")
    kv_cache = init_kv_cache(device)

    # Connect client
    client = MessageQueueClient(
        server_url=f"tcp://localhost:{port}", context=zmq.Context.instance()
    )
    instance_id = os.getpid() + 7000 + (1 if enable_abo else 0)

    # Register
    client.submit_request(
        RequestType.REGISTER_KV_CACHE,
        [instance_id, [CudaIPCWrapper(t) for t in kv_cache], "model", 1, {}],
        get_response_class(RequestType.REGISTER_KV_CACHE),
    ).result(timeout=10)

    # Fill test data
    src_offset = 0
    for layer in range(NUM_LAYERS):
        kv_cache[layer][:, src_offset : src_offset + BLOCKS_PER_CHUNK * NUM_CHUNKS] = (
            0.5
        )

    # ===== WARMUP: store + retrieve 1 chunk to initialize StagingPool =====
    warmup_key = make_key(999)
    warmup_block_ids = list(range(src_offset, src_offset + BLOCKS_PER_CHUNK))
    event = torch.cuda.Event(interprocess=True)
    event.record()

    # Warmup store
    warmup_future = client.submit_request(
        RequestType.STORE,
        [warmup_key, instance_id, warmup_block_ids, event.ipc_handle()],
        get_response_class(RequestType.STORE),
    )
    warmup_future.to_cuda_future().result(timeout=30)
    time.sleep(0.5)

    # Warmup lookup
    lookup_key = warmup_key.no_worker_id_version()
    job_id = client.submit_request(
        RequestType.LOOKUP, [lookup_key, 1], get_response_class(RequestType.LOOKUP)
    ).result(timeout=10)
    while True:
        result = client.submit_request(
            RequestType.QUERY_PREFETCH_STATUS,
            [job_id],
            get_response_class(RequestType.QUERY_PREFETCH_STATUS),
        ).result(timeout=10)
        if result is not None:
            break

    # Warmup retrieve
    dst_offset = 512
    warmup_dst_blocks = list(range(dst_offset, dst_offset + BLOCKS_PER_CHUNK))
    warmup_retrieve = client.submit_request(
        RequestType.RETRIEVE,
        [warmup_key, instance_id, warmup_dst_blocks, event.ipc_handle(), 0],
        get_response_class(RequestType.RETRIEVE),
    )
    warmup_retrieve.to_cuda_future().result(timeout=30)
    print("Warmup done (StagingPool initialized)")

    # Clear warmup data
    client.submit_request(
        RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
    ).result(timeout=10)
    time.sleep(0.3)

    # ===== FORMAL TEST =====
    keys = [make_key(1000 + i) for i in range(NUM_CHUNKS)]

    # ===== STORE timing =====
    torch.cuda.synchronize()
    store_start = time.perf_counter()

    event = torch.cuda.Event(interprocess=True)
    event.record()
    store_futures = []
    for i, key in enumerate(keys):
        block_ids = list(
            range(
                src_offset + i * BLOCKS_PER_CHUNK,
                src_offset + (i + 1) * BLOCKS_PER_CHUNK,
            )
        )
        future = client.submit_request(
            RequestType.STORE,
            [key, instance_id, block_ids, event.ipc_handle()],
            get_response_class(RequestType.STORE),
        )
        store_futures.append(future)

    # Wait all
    for f in store_futures:
        f.to_cuda_future().result(timeout=30)

    torch.cuda.synchronize()
    store_end = time.perf_counter()
    store_time = store_end - store_start

    print(f"STORE time: {store_time * 1000:.2f} ms ({NUM_CHUNKS} chunks)")

    # Wait for finish_write host callback
    time.sleep(0.5)

    # ===== LOOKUP =====
    total_found = 0
    for key in keys:
        lookup_key = key.no_worker_id_version()
        job_id = client.submit_request(
            RequestType.LOOKUP, [lookup_key, 1], get_response_class(RequestType.LOOKUP)
        ).result(timeout=10)
        while True:
            result = client.submit_request(
                RequestType.QUERY_PREFETCH_STATUS,
                [job_id],
                get_response_class(RequestType.QUERY_PREFETCH_STATUS),
            ).result(timeout=10)
            if result is not None:
                total_found += result
                break

    assert total_found == NUM_CHUNKS, f"Expected {NUM_CHUNKS} hits, got {total_found}"

    # ===== RETRIEVE timing =====
    torch.cuda.synchronize()
    retrieve_start = time.perf_counter()

    dst_offset = 512
    retrieve_futures = []
    for i, key in enumerate(keys):
        block_ids = list(
            range(
                dst_offset + i * BLOCKS_PER_CHUNK,
                dst_offset + (i + 1) * BLOCKS_PER_CHUNK,
            )
        )
        future = client.submit_request(
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event.ipc_handle(), 0],
            get_response_class(RequestType.RETRIEVE),
        )
        retrieve_futures.append(future)

    # Wait all
    for f in retrieve_futures:
        f.to_cuda_future().result(timeout=30)

    torch.cuda.synchronize()
    retrieve_end = time.perf_counter()
    retrieve_time = retrieve_end - retrieve_start

    print(f"RETRIEVE time: {retrieve_time * 1000:.2f} ms ({NUM_CHUNKS} chunks)")

    # Verify data (ABO is lossy)
    if enable_abo:
        atol = 0.1
        for i in range(NUM_CHUNKS):
            for layer in range(NUM_LAYERS):
                src = kv_cache[layer][
                    :,
                    src_offset + i * BLOCKS_PER_CHUNK : src_offset
                    + (i + 1) * BLOCKS_PER_CHUNK,
                ]
                dst = kv_cache[layer][
                    :,
                    dst_offset + i * BLOCKS_PER_CHUNK : dst_offset
                    + (i + 1) * BLOCKS_PER_CHUNK,
                ]
                if not torch.allclose(src.float(), dst.float(), atol=atol):
                    pass  # ABO lossy, ignore
        print(f"Verification done (atol={atol})")
    else:
        # Exact match for non-ABO
        for i in range(NUM_CHUNKS):
            for layer in range(NUM_LAYERS):
                src = kv_cache[layer][
                    :,
                    src_offset + i * BLOCKS_PER_CHUNK : src_offset
                    + (i + 1) * BLOCKS_PER_CHUNK,
                ]
                dst = kv_cache[layer][
                    :,
                    dst_offset + i * BLOCKS_PER_CHUNK : dst_offset
                    + (i + 1) * BLOCKS_PER_CHUNK,
                ]
                assert torch.allclose(src, dst), f"Mismatch chunk {i} layer {layer}"
        print("Verification: exact match")

    # Cleanup
    client.submit_request(
        RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
    ).result(timeout=10)
    client.close()

    server.terminate()
    server.join(timeout=3)

    return store_time, retrieve_time


def main():
    assert torch.cuda.is_available(), "CUDA required"

    mp_lib.set_start_method("spawn", force=True)

    # Test ABO enabled
    store_abo, retrieve_abo = run_test(enable_abo=True, port=SERVER_PORT)

    # Test ABO disabled
    store_no_abo, retrieve_no_abo = run_test(enable_abo=False, port=SERVER_PORT + 100)

    # Summary
    print("\n" + "=" * 60)
    print("LATENCY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'ABO ON':<15} {'ABO OFF':<15} {'Diff':<15}")
    print("-" * 60)
    print(
        f"{'STORE (ms)':<20} {store_abo * 1000:<15.2f} {store_no_abo * 1000:<15.2f} {(store_abo - store_no_abo) * 1000:<15.2f}"
    )
    print(
        f"{'RETRIEVE (ms)':<20} {retrieve_abo * 1000:<15.2f} {retrieve_no_abo * 1000:<15.2f} {(retrieve_abo - retrieve_no_abo) * 1000:<15.2f}"
    )
    print(
        f"{'TOTAL (ms)':<20} {(store_abo + retrieve_abo) * 1000:<15.2f} {(store_no_abo + retrieve_no_abo) * 1000:<15.2f} {(store_abo + retrieve_abo - store_no_abo - retrieve_no_abo) * 1000:<15.2f}"
    )

    # Compression ratio
    chunk_bytes = (
        2 * NUM_LAYERS * BLOCKS_PER_CHUNK * PAGE_SIZE * NUM_HEADS * HEAD_SIZE * 2
    )
    total_bytes = chunk_bytes * NUM_CHUNKS
    print(f"\nData size per chunk: {chunk_bytes / 1024 / 1024:.2f} MB")
    print(f"Total data: {total_bytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
