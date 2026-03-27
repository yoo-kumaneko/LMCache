# Pure C++ LMCache Server — Developer Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Build System](#build-system)
4. [File Reference](#file-reference)
5. [Wire Protocol](#wire-protocol)
6. [CUDA IPC & GPU Context](#cuda-ipc--gpu-context)
7. [L1 Slab Storage](#l1-slab-storage)
8. [Token Hashing & Sessions](#token-hashing--sessions)
9. [ZMQ Message Queue Server](#zmq-message-queue-server)
10. [Cache Engine Orchestration](#cache-engine-orchestration)
11. [Request Handlers](#request-handlers)
12. [Initialization & Startup Ordering](#initialization--startup-ordering)
13. [Debugging History & Lessons Learned](#debugging-history--lessons-learned)
14. [Configuration & Deployment](#configuration--deployment)
15. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Overview

This is a complete pure C++ rewrite of the LMCache multiprocess server (originally `lmcache/v1/multiprocess/server.py`). It replaces the Python server process with a single C++ binary that is wire-compatible with existing Python vLLM clients. No client-side changes are needed.

**Tested configuration:** DeepSeek-V3.1 on 8× NVIDIA H20 GPUs, TP=8, fp8 KV cache, MLA attention, block_size=64, 61 layers, hidden_dim=576.

**Codebase stats:** 5,238 lines across 20 files (10 .cpp, 10 .h), plus reused `mem_kernels.cu`, `bitmap.cpp`, and `ttl_lock.cpp`.

---

## Architecture

```
vLLM Worker (Python)                     C++ LMCache Server
┌──────────────┐                      ┌──────────────────────────┐
│  TP Worker 0 │─── ZMQ DEALER ──────>│                          │
│  TP Worker 1 │─── ZMQ DEALER ──────>│  MessageQueueServer      │
│  ...         │                      │  (ZMQ ROUTER + eventfd)  │
│  TP Worker 7 │─── ZMQ DEALER ──────>│                          │
└──────────────┘                      ├──────────────────────────┤
                                      │  Main Loop (zmq_poll)    │
                                      │  ├─ SYNC handlers        │
                                      │  └─ Thread pools         │
                                      │     ├─ General (16 wkrs) │
                                      │     └─ SYNC_LOOKUP (8)   │
                                      ├──────────────────────────┤
                                      │  CacheEngine             │
                                      │  ├─ GPUContext[0..7]     │
                                      │  ├─ L1Store (mmap slab)  │
                                      │  ├─ TokenHasher (BLAKE3) │
                                      │  └─ SessionManager       │
                                      └──────────────────────────┘
```

### Data Flow: Store Operation

```
vLLM Worker (TP rank X, device X):
  1. Encode STORE request: [key, instance_id, gpu_block_ids, event_ipc_handle]
  2. Send via ZMQ DEALER to server

C++ Server (blocking handler thread):
  1. Decode msgpack frames → StorePayload
  2. Compute token chunk hashes (BLAKE3 rolling prefix)
  3. Map to ObjectKeys (model_name + chunk_hash + kv_rank)
  4. L1Store::reserve_write() → get slab write slots
  5. Set CUDA device + stream guards for this GPU
  6. Wait on vLLM's CUDA event (ensure GPU writes complete)
  7. For each chunk:
     a. Get slot_mapping slice for this chunk's block IDs
     b. multi_layer_kv_transfer(D2H): GPU KV cache → tmp_gpu_buffer
     c. cudaMemcpyAsync(D2H): tmp_gpu_buffer → L1 slab
  8. cudaStreamSynchronize
  9. L1Store::finish_write() → transition slots to Ready
  10. Create + record completion CUDA event
  11. Encode response: [event_handle, true]
```

### Data Flow: Retrieve Operation

```
Same as Store but reversed:
  - L1Store::reserve_read()
  - cudaMemcpyAsync(H2D): L1 slab → tmp_gpu_buffer
  - multi_layer_kv_transfer(H2D): tmp_gpu_buffer → GPU KV cache
  - L1Store::finish_read()
  - Uses high-priority CUDA stream
```

### Data Flow: SYNC_LOOKUP

```
1. Compute chunk hashes → ObjectKeys
2. L1Store::prefix_lookup() → count leading hits
3. If hits > 0: reserve_read() + save PendingLookupState
4. Return found_count (number of cached chunks)
```

---

## Build System

### CMakeLists.txt

The build uses CMake 3.20+ with C++17 and CUDA 17. Dependencies are resolved via:

| Dependency | Resolution |
|------------|-----------|
| **libtorch** | Auto-discovered via `python3 -c "import torch; print(torch.utils.cmake_prefix_path)"` |
| **CUDA Toolkit** | `find_package(CUDAToolkit)` — system install |
| **libzmq** | System `.so.5` + FetchContent for headers (avoids needing dev packages) |
| **cppzmq** | FetchContent header-only C++ binding |
| **msgpack-cxx** | FetchContent header-only, compiled with `MSGPACK_NO_BOOST` |
| **BLAKE3** | FetchContent, built as static library with SIMD assembly (SSE2/SSE4.1/AVX2/AVX512) |

### CUDA Architectures

```
CMAKE_CUDA_ARCHITECTURES: 80 (Ampere), 86 (Ampere), 89 (Ada), 90 (Hopper)
```

### Reused Sources

Three existing LMCache C++ files are compiled directly (not via pybind11):

```
csrc/mem_kernels.cu           — CUDA KV transfer kernels
csrc/storage_manager/bitmap.cpp — Bitmap bitwise operations
csrc/storage_manager/ttl_lock.cpp — TTL-based lock management
```

### Build Commands

```bash
cd LMCache-repo/csrc/server
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # or Debug
make -j$(nproc)
# Binary: ./lmcache-server
```

### Important Linker Flags

- `-rdynamic` — enables readable symbol names in `backtrace_symbols()` for crash handler
- RPATH includes `/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib` for `libnvJitLink.so`

---

## File Reference

### Headers (.h)

| File | Lines | Key Contents |
|------|-------|-------------|
| `types.h` | 280 | `RequestType` enum (1-21), `DType`, `ObjectKey`+hash, `IPCCacheEngineKey`, `CudaIpcTensorDesc`, `MemorySlabRef`, `PrefetchHandle`, `compute_extra_count()` |
| `wire_protocol.h` | 164 | `Encoder`/`Decoder` classes, payload structs (`StorePayload`, `RetrievePayload`, `LookupPayload`, etc.) |
| `cache_engine.h` | 165 | `CacheEngine` class, `PendingLookupState`, `PrefetchJob` |
| `gpu_context.h` | 125 | `GPUContext` class — per-GPU IPC handles, streams, slot mapping, tmp buffer |
| `l1_store.h` | 112 | `L1Store` abstract interface + `L1StoreConfig` |
| `l2_adapter.h` | 107 | `L2Adapter` abstract interface (placeholder) |
| `session_manager.h` | 95 | `Session` + `SessionManager` classes |
| `tensor_bridge.h` | 89 | `wrap_as_tensor()`, `open_ipc_tensor()`, `open_ipc_event()`, `create_ipc_event()` |
| `mq_server.h` | 88 | `IRequestHandler` + `MessageQueueServer` |
| `token_hasher.h` | 73 | `TokenHasher` class |

### Implementations (.cpp)

| File | Lines | Key Contents |
|------|-------|-------------|
| `wire_protocol.cpp` | 905 | msgpack encode/decode, Python pickle protocol 4 parser, `map_find()` helper |
| `cache_engine.cpp` | 602 | `store()`, `retrieve()`, `lookup()`, `sync_lookup()`, GPU transfer orchestration |
| `l1_store.cpp` | 580 | `SlabL1Store` implementation: mmap, cudaHostRegister, state machine, LRU eviction |
| `mq_server.cpp` | 467 | ZMQ ROUTER, eventfd, thread pools, `zmq_poll` main loop |
| `main.cpp` | 448 | 13 handler classes, CLI parsing, CUDA init, server wiring |
| `gpu_context.cpp` | 432 | IPC handle opening, format discovery, stream creation, slot mapping |
| `tensor_bridge.cpp` | 202 | `getIpcDevPtr()`, `from_blob()`, CUDA event IPC helpers |
| `session_manager.cpp` | 135 | Hash caching with TTL cleanup |
| `token_hasher.cpp` | 128 | BLAKE3 rolling prefix hash |
| `types.cpp` | 41 | `ipc_key_to_object_keys()` |

---

## Wire Protocol

### ZMQ Frame Layout

Each request is a multipart ZMQ message:

```
Frame 0: ZMQ routing identity (auto-managed by ROUTER)
Frame 1: Empty delimiter
Frame 2: RequestUID (int64, msgpack-encoded)
Frame 3: RequestType (int, msgpack-encoded)
Frame 4+: Payload frames (varies by request type)
```

Each response:

```
Frame 0: ZMQ routing identity (echoed back)
Frame 1: Empty delimiter
Frame 2: RequestUID (echoed back)
Frame 3: Encoded response (msgpack)
```

### Payload Formats by Request Type

| RequestType | Payload Frames | Response |
|-------------|---------------|----------|
| `REGISTER_KV_CACHE` | `[instance_id, kv_caches_list, model_name, world_size]` | None |
| `UNREGISTER_KV_CACHE` | `[instance_id]` | None |
| `STORE` | `[key, instance_id, gpu_block_ids, event_ipc_handle]` | `(bytes, bool)` |
| `RETRIEVE` | `[key, instance_id, gpu_block_ids, event_ipc_handle, skip_tokens]` | `(bytes, bool)` |
| `LOOKUP` | `[key, tp_size]` | `int` |
| `SYNC_LOOKUP` | `[key, tp_size]` | `int` |
| `QUERY_PREFETCH_STATUS` | `[prefetch_job_id]` | `int \| None` |
| `FREE_LOOKUP_LOCKS` | `[key, tp_size]` | None |
| `END_SESSION` | `[request_id]` | None |
| `CLEAR` | (none) | None |
| `GET_CHUNK_SIZE` | (none) | `int` |
| `PING` | (none) | `bool` |
| `NOOP` | (none) | `str` |

### IPCCacheEngineKey Encoding

The Python client encodes `IPCCacheEngineKey` as a **msgpack MAP** (dict with string keys), NOT an array. The C++ decoder handles both formats via `map_find()` helper:

```
{
  "model_name": str,
  "world_size": int,
  "worker_id": int | None,
  "token_ids": [int, ...],
  "start": int,
  "end": int,
  "request_id": str
}
```

### CudaIPCWrapper (Pickle Protocol 4)

CUDA IPC tensor descriptors are serialized by Python as pickle Ext type code 1. The C++ parser extracts:

1. **`ipc_handle_blob`**: First `BINBYTES`/`SHORT_BINBYTES` object ≥64 bytes (this is `handle[1]` from `_share_cuda_()`)
2. **`storage_size_bytes`**: `handle[2]` — extracted from int after the handle blob
3. **`device_uuid`**: `SHORT_BINUNICODE` string after the `'device_uuid'` field name in the pickle stream. The pickle contains a bare UUID; we prepend `"GPU-"` for CUDA device matching.
4. **`dtype`**: Extracted from `STACK_GLOBAL` opcode (e.g., `"uint8"` for fp8)
5. **`shape`/`stride`**: Extracted from `TUPLE2`/`TUPLE3` opcodes (small tuples) or `MARK`...`TUPLE` (large tuples)
6. **`storage_offset`**: Last `int` before final `TUPLE`

### DType Mapping

| Wire string | DType enum | at::ScalarType | Element size |
|-------------|-----------|---------------|-------------|
| `"uint8"` | `Int8` | `Byte` | 1 |
| `"float16"` | `Float16` | `Half` | 2 |
| `"bfloat16"` | `BFloat16` | `BFloat16` | 2 |
| `"float32"` | `Float32` | `Float` | 4 |
| `"float8_e4m3fn"` | `Float8E4M3FN` | `Float8_e4m3fn` | 1 |

Note: fp8 KV cache uses `torch.uint8` on the wire, which maps to `DType::Int8` → `at::ScalarType::Byte`.

---

## CUDA IPC & GPU Context

### GPUContext Class (`gpu_context.h`, `gpu_context.cpp`)

Each vLLM TP worker registers its KV cache via `REGISTER_KV_CACHE`. The server creates one `GPUContext` per instance_id (= per TP worker PID).

#### Construction Flow

```
1. Match device UUID from first tensor descriptor to local CUDA device index
   (iterate cudaGetDeviceProperties, format UUID as "GPU-%02x%02x...")
2. Open IPC tensor handles via c10::cuda::CUDACachingAllocator::getIpcDevPtr()
   → Creates at::Tensor objects that MUST stay alive (stored as class member)
3. Discover GPU KV format from tensor shapes:
   - 5D [2,NB,BS,NH,HS] → NL_X_TWO_NB_BS_NH_HS (flash attn)
   - 5D [NB,2,BS,NH,HS] → NL_X_NB_TWO_BS_NH_HS (flash infer)
   - 3D [NB,BS,HS] → NL_X_NB_BS_HS (MLA)
4. Extract shape parameters: num_blocks, block_size, hidden_dim_size
5. Upload KV cache pointers to GPU as int64 tensor [num_layers]
6. Pre-compute slot mapping: [num_blocks, block_size] where slot[b][s] = b*bs+s
7. Allocate temporary GPU buffer for transfers
8. Create normal + high-priority CUDA streams
```

#### Critical: IPC Tensor Lifetime

```cpp
// WRONG (original bug — dangling pointers after constructor):
std::vector<at::Tensor> kv_tensors;  // local variable, destroyed at end of ctor
for (int i = 0; i < num_layers_; ++i) {
    at::Tensor t = open_ipc_tensor(desc, device_idx);
    kv_cache_ptrs_[i] = t.data_ptr();  // saves raw pointer
    kv_tensors.push_back(std::move(t)); // tensor destroyed when kv_tensors goes out of scope!
}

// CORRECT (fixed):
kv_cache_ipc_tensors_.reserve(num_layers_);  // class member!
for (int i = 0; i < num_layers_; ++i) {
    at::Tensor t = open_ipc_tensor(desc, device_idx);
    kv_cache_ptrs_[i] = t.data_ptr();
    kv_cache_ipc_tensors_.push_back(std::move(t));  // kept alive for GPUContext lifetime
}
```

**Why this matters:** `getIpcDevPtr()` returns `shared_ptr<void>`. PyTorch's IPC cache stores only `weak_ptr`. When the tensor (which holds the `shared_ptr` in its storage) is destroyed, the IPC handle is closed and the GPU memory is unmapped. Any subsequent kernel access to the raw pointer is an illegal memory access.

### Tensor Bridge (`tensor_bridge.h`, `tensor_bridge.cpp`)

#### `open_ipc_tensor()`

```cpp
// 1. Open IPC handle via PyTorch's caching allocator
std::string handle_str(blob.data(), blob.size());
auto dev_ptr = c10::cuda::CUDACachingAllocator::getIpcDevPtr(std::move(handle_str));

// 2. Apply storage_offset
void* offset_ptr = static_cast<char*>(dev_ptr.get()) + storage_offset * elem_size;

// 3. Create tensor with custom deleter holding shared_ptr
auto ref = std::make_shared<std::shared_ptr<void>>(std::move(dev_ptr));
auto deleter = [ref](void*) { /* ref drops when tensor dies */ };
return at::from_blob(offset_ptr, shape, stride, deleter, options);
```

**Why `getIpcDevPtr()` instead of `cudaIpcOpenMemHandle()`:** PyTorch 2.10+ with CUDA 12.x uses an expanded 66-byte handle format (not the raw 64-byte `cudaIpcMemHandle_t`). `getIpcDevPtr()` handles both formats.

#### `wrap_as_tensor()`

Non-owning tensor from raw pointer + shape + dtype:

```cpp
return at::from_blob(ptr, shape, options);
```

#### CUDA Event IPC

```cpp
cudaEvent_t open_ipc_event(const uint8_t* handle_bytes);   // Opens 64-byte IPC event handle
std::vector<uint8_t> create_ipc_event(cudaEvent_t& event); // Creates event + exports handle
```

### CUDA Device & Stream Scoping

Every GPU operation in `store()` and `retrieve()` uses ATen guards:

```cpp
// Set the correct CUDA device
c10::cuda::CUDAGuard device_guard(gpu_ctx->device_index());

// Direct all ATen + CUDA operations to this GPU's stream
at::cuda::CUDAStream torch_stream =
    at::cuda::getStreamFromExternal(gpu_ctx->stream(), gpu_ctx->device_index());
at::cuda::CUDAStreamGuard stream_guard(torch_stream);
```

This matches the Python server's `with torch.cuda.device(dev), torch.cuda.stream(stream):` pattern.

---

## L1 Slab Storage

### Architecture

```
┌─────────────────────────────────────────┐
│           L1 Slab (mmap'd)              │
│  ┌──────┬──────┬──────┬──────┬────────┐ │
│  │Slot 0│Slot 1│Slot 2│ ...  │Slot N-1│ │
│  │ FREE │READY │WRITE │      │READING │ │
│  └──────┴──────┴──────┴──────┴────────┘ │
│  cudaHostRegister'd → zero-copy DMA     │
├─────────────────────────────────────────┤
│  Metadata (hash map):                   │
│    ObjectKey → { slot_index, state,     │
│                  lock_count, lru_pos }  │
│  TTLLock for read lock management       │
│  LRU eviction when capacity exhausted   │
└─────────────────────────────────────────┘
```

### State Machine

```
 ┌──── Free ◄─── evict()/delete_key()
 │       │
 │  reserve_write()
 │       │
 │       ▼
 │    Writing
 │       │
 │  finish_write()
 │       │
 │       ▼
 └──── Ready ◄─── finish_read()
          │
     reserve_read()
          │
          ▼
       Reading ───► Ready (when lock_count → 0)
```

### Memory Layout

- Slab is a single contiguous `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` allocation
- Optional `MAP_HUGETLB` for 2MB huge pages
- `cudaHostRegister(ptr, size, cudaHostRegisterDefault)` enables zero-copy DMA:
  - GPU can DMA directly to/from the slab without staging buffers
  - `cudaMemcpyAsync(D2H/H2D)` between GPU and slab is truly async
- Each slot has a fixed size determined by the first `reserve_write()` layout

### MLA Multi-Reader Locking

For MLA models, all TP workers share the same KV cache object (since there's only one KV head). The `extra_count` parameter in `reserve_read()`/`finish_read()` handles this:

```cpp
int extra_count = compute_extra_count(tp_size, world_size);
// Non-MLA: extra_count = 0 (each worker has distinct KV shard)
// MLA: extra_count = tp_size - 1 (all workers share same object)
```

---

## Token Hashing & Sessions

### TokenHasher (`token_hasher.h`, `token_hasher.cpp`)

Computes rolling prefix hashes using BLAKE3:

```
Chunk 0 hash = BLAKE3(none_hash || tokens[0:chunk_size])
Chunk 1 hash = BLAKE3(chunk_0_hash || tokens[chunk_size:2*chunk_size])
Chunk N hash = BLAKE3(chunk_{N-1}_hash || tokens[N*chunk_size:(N+1)*chunk_size])
```

Where `none_hash` is `BLAKE3(b"None")` — the initial prefix context.

Only complete chunks are hashed (trailing tokens < chunk_size are ignored).

### SessionManager (`session_manager.h`, `session_manager.cpp`)

- `Session` stores per-request state: token_ids, cached hashes
- `get_hashes(start, end)` returns chunk hashes for the range, using cached values when possible
- Thread-safe (mutex per session)
- `cleanup_expired()` removes sessions older than TTL

---

## ZMQ Message Queue Server

### Architecture

```
                          ┌──────────────┐
  ZMQ ROUTER socket ─────┤  Main Loop   │
  (tcp://host:port)       │  zmq_poll()  │
                          │  [socket_fd, │
                          │   eventfd]   │
                          └──────┬───────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
     ┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
     │ SYNC handler  │  │ General pool  │  │ Dedicated pool│
     │ (main thread) │  │ (16 workers)  │  │ SYNC_LOOKUP   │
     │               │  │ STORE,RETRIEVE│  │ (8 workers)   │
     │ GET_CHUNK_SIZE│  │ LOOKUP, etc.  │  │               │
     │ REGISTER, etc.│  │               │  │               │
     └───────────────┘  └───────┬───────┘  └───────┬───────┘
                                │                   │
                         eventfd write ◄────────────┘
                         (wakes main loop to send ZMQ response)
```

### Key Design Points

1. **SYNC handlers** run in the main ZMQ poll loop (fast, non-blocking)
2. **BLOCKING handlers** are dispatched to a thread pool
3. **eventfd** bridges thread pool → main loop: worker writes 1 to eventfd, main loop wakes up and sends the ZMQ response
4. **Dedicated thread pool** for `SYNC_LOOKUP` prevents starvation by heavy STORE/RETRIEVE I/O
5. Responses are always sent from the main loop (ZMQ sockets are not thread-safe)

### Handler Classification

| Handler Type | Request Types |
|-------------|--------------|
| **SYNC** | `REGISTER_KV_CACHE`, `UNREGISTER_KV_CACHE`, `QUERY_PREFETCH_STATUS`, `GET_CHUNK_SIZE`, `NOOP` |
| **BLOCKING** (general) | `STORE`, `RETRIEVE`, `LOOKUP`, `FREE_LOOKUP_LOCKS`, `END_SESSION`, `CLEAR`, `PING` |
| **BLOCKING** (dedicated) | `SYNC_LOOKUP` |

---

## Cache Engine Orchestration

### Store Flow (Detailed)

```cpp
std::pair<std::vector<uint8_t>, bool> CacheEngine::store(
    const IPCCacheEngineKey& key,
    int instance_id,
    const std::vector<int32_t>& gpu_block_ids,
    const std::vector<uint8_t>& event_ipc_handle)
{
  // 1. Session management + hash computation
  auto session = session_manager_.get_or_create(key.request_id);
  session->set_tokens(key.token_ids);
  auto chunk_hashes = session->get_hashes(key.start, key.end);
  auto obj_keys = ipc_key_to_object_keys(model_name, world_size, worker_id, chunk_hashes);

  // 2. GPU context lookup
  auto& gpu_ctx = gpu_contexts_[instance_id];

  // 3. CUDA device + stream guards
  c10::cuda::CUDAGuard device_guard(gpu_ctx->device_index());
  at::cuda::CUDAStream torch_stream =
      at::cuda::getStreamFromExternal(gpu_ctx->stream(), gpu_ctx->device_index());
  at::cuda::CUDAStreamGuard stream_guard(torch_stream);

  // 4. Wait for vLLM to finish writing KV cache
  cudaEvent_t vllm_event = open_ipc_event(event_ipc_handle.data());
  cudaStreamWaitEvent(gpu_ctx->stream(), vllm_event, 0);

  // 5. Reserve L1 write slots
  auto reserved = l1_store_->reserve_write(obj_keys, layout, "new");

  // 6. Transfer each chunk: GPU → tmp_buffer → L1
  {
    std::lock_guard<std::mutex> lk(gpu_ctx->transfer_lock());
    at::Tensor slot_mapping = gpu_ctx->get_slot_mapping_tensor(gpu_block_ids);

    for (size_t idx = 0; idx < obj_keys.size(); ++idx) {
      // Slice slot mapping for this chunk
      at::Tensor slot_slice = slot_mapping.slice(0, start_tok, end_tok);
      at::Tensor tmp_buf = gpu_ctx->get_tmp_gpu_buffer(chunk_size_);

      // GPU KV cache → tmp_buffer (CUDA kernel)
      multi_layer_kv_transfer(tmp_buf, gpu_ctx->kv_pointers(), slot_slice,
                               device, page_buf_size, D2H, format, block_size, 0);

      // tmp_buffer → L1 slab (async memcpy)
      cudaMemcpyAsync(slab_ref.data, tmp_buf.data_ptr(),
                       slab_ref.size_bytes, cudaMemcpyDeviceToHost, stream);
    }
  }

  // 7. Sync and finish
  cudaStreamSynchronize(gpu_ctx->stream());
  l1_store_->finish_write(written_keys);
  return {done_event_bytes, true};
}
```

### Retrieve Flow

Same structure as Store but:
- Uses **high-priority stream** (for latency-sensitive path)
- Checks `PendingLookupState` for async prefetch completion
- Applies `skip_first_n_tokens` (skips tokens already in GPU cache)
- Transfer direction is H2D (host → device)

### SYNC_LOOKUP Flow

```cpp
int CacheEngine::sync_lookup(const IPCCacheEngineKey& key, int tp_size) {
  // 1. Compute chunk hashes
  auto chunk_hashes = token_hasher_.compute_chunk_hashes(key.token_ids);
  auto obj_keys = ipc_key_to_object_keys(...);

  // 2. L1 prefix lookup (find longest prefix of existing keys)
  int64_t hit_count = l1_store_->prefix_lookup(obj_keys);
  int found_count = hit_count / key.world_size;

  // 3. If hits: reserve read locks + save pending state for RETRIEVE
  if (found_count > 0) {
    l1_store_->reserve_read(hit_keys, extra_count);
    pending_lookups_[key.request_id] = PendingLookupState{
        key.world_size, key.world_size, {}, false};
  }

  return found_count;
}
```

---

## Request Handlers

All 13 handlers are defined in `main.cpp` as concrete `IRequestHandler` implementations:

| Class | Request Type | Handler Type | Action |
|-------|-------------|-------------|--------|
| `RegisterHandler` | `REGISTER_KV_CACHE` | SYNC | `engine.register_kv_cache()` |
| `UnregisterHandler` | `UNREGISTER_KV_CACHE` | SYNC | `engine.unregister_kv_cache()` |
| `StoreHandler` | `STORE` | BLOCKING | `engine.store()` |
| `RetrieveHandler` | `RETRIEVE` | BLOCKING | `engine.retrieve()` |
| `LookupHandler` | `LOOKUP` | BLOCKING | `engine.lookup()` |
| `SyncLookupHandler` | `SYNC_LOOKUP` | BLOCKING | `engine.sync_lookup()` |
| `QueryPrefetchStatusHandler` | `QUERY_PREFETCH_STATUS` | SYNC | `engine.query_prefetch_status()` |
| `FreeLookupLocksHandler` | `FREE_LOOKUP_LOCKS` | BLOCKING | `engine.free_lookup_locks()` |
| `EndSessionHandler` | `END_SESSION` | BLOCKING | `engine.end_session()` |
| `ClearHandler` | `CLEAR` | BLOCKING | `engine.clear()` |
| `GetChunkSizeHandler` | `GET_CHUNK_SIZE` | SYNC | `engine.get_chunk_size()` |
| `PingHandler` | `PING` | BLOCKING | `engine.ping()` |
| `NoopHandler` | `NOOP` | SYNC | Returns "OK" |

---

## Initialization & Startup Ordering

**Critical ordering in `main()`:**

```cpp
// 1. Signal handlers (SIGINT, SIGTERM → clean shutdown; SIGSEGV, SIGABRT → backtrace)
std::signal(SIGSEGV, crash_handler);

// 2. Parse CLI arguments
auto cfg = parse_args(argc, argv);

// 3. Initialize CUDA runtime BEFORE CacheEngine
//    This MUST happen before cudaHostRegister (in L1Store) or it silently fails!
for (int i = 0; i < device_count; ++i) {
    cudaSetDevice(i);
    cudaFree(nullptr);  // Force lazy CUDA init
}
cudaSetDevice(0);
c10::cuda::CUDACachingAllocator::init(device_count);  // Required for getIpcDevPtr()

// 4. Create CacheEngine (L1 slab mmap + cudaHostRegister happens here)
CacheEngine engine(cfg.chunk_size, l1_config, nullptr);

// 5. Create ZMQ server + register all handlers
MessageQueueServer server(bind_url, cfg.max_workers);
server.add_handler(RequestType::STORE, std::make_unique<StoreHandler>(engine));
// ... all 13 handlers ...

// 6. Add dedicated thread pool for SYNC_LOOKUP (8 workers)
server.add_dedicated_thread_pool({RequestType::SYNC_LOOKUP}, 8);

// 7. Start server (spawns main loop thread)
server.start();

// 8. Sleep loop until shutdown signal
while (!g_shutdown) sleep(1);
```

**Why CUDA init must come first:**
- `cudaHostRegister()` requires a CUDA context to exist. Without `cudaFree(nullptr)` first, the register call silently fails.
- Later `cudaMemcpyAsync` from the slab hits illegal memory access because the slab wasn't actually pinned.
- `c10::cuda::CUDACachingAllocator::init()` is required for `getIpcDevPtr()` to work.

---

## Debugging History & Lessons Learned

### Bug 1: PyTorch 2.10+ IPC Handle Format

**Symptom:** `cudaIpcOpenMemHandle` returned `CUDA_ERROR_INVALID_VALUE`
**Root Cause:** PyTorch 2.10+ with CUDA 12.x uses a 66-byte expanded handle format, not the raw 64-byte `cudaIpcMemHandle_t`.
**Fix:** Use `c10::cuda::CUDACachingAllocator::getIpcDevPtr()` which handles both formats.

### Bug 2: CUDA Caching Allocator Not Initialized

**Symptom:** Segfault inside `getIpcDevPtr()`
**Root Cause:** `c10::cuda::CUDACachingAllocator` was not initialized before the first `getIpcDevPtr()` call.
**Fix:** Call `c10::cuda::CUDACachingAllocator::init(device_count)` in `main()` before creating `CacheEngine`.

### Bug 3: Dangling IPC Pointers (The Final Critical Bug)

**Symptom:** First store on one GPU succeeded, then `multi_layer_kv_transfer` crashed with illegal memory access on subsequent stores. CUDA error became "sticky" and broke all subsequent operations (including `cudaIpcOpenEventHandle`).
**Root Cause:** IPC tensors were stored in a **local variable** in the `GPUContext` constructor. When the constructor returned, the tensors were destroyed, releasing the `shared_ptr` from `getIpcDevPtr()`, which closed the IPC handle and unmapped the GPU memory. `kv_cache_ptrs_` then contained dangling pointers.
**Fix:** Store IPC tensors as a **class member** `kv_cache_ipc_tensors_` so they live as long as the `GPUContext`.

### Bug 4: IPCCacheEngineKey MAP vs ARRAY

**Symptom:** Key decode failed — fields were all zeros/empty.
**Root Cause:** Python `msgspec.msgpack.encode()` for `@msgspec.Struct` classes produces msgpack MAP (dict with string keys), not ARRAY. The decoder assumed ARRAY format.
**Fix:** Support both formats with a `map_find()` helper function.

### Bug 5: Pickle Parser Issues

**Symptom:** Empty `device_uuid`, wrong IPC handle extraction, shape parsing failures.
**Root Cause (multiple):**
1. `device_uuid`: Pickle contains bare UUID, not "GPU-" prefixed → prepend "GPU-" after extraction
2. IPC handle: Originally took last 64-byte blob (handle[6] = event handle) → take FIRST ≥64-byte `BINBYTES` (handle[1] = storage handle)
3. Shape: Pickle uses `TUPLE2`/`TUPLE3` opcodes for small tuples, not `MARK`...`TUPLE` → handle both formats

### Bug 6: uint8 DType Not Recognized

**Symptom:** fp8 KV cache dtype failed to parse.
**Root Cause:** fp8 uses `torch.uint8` on the wire, which wasn't in the dtype mapping.
**Fix:** Add `"uint8"` → `DType::Int8` → `at::ScalarType::Byte` mapping.

### Bug 7: ATen Header Namespace Conflicts

**Symptom:** Compilation error — `c10::ScalarType` vs `at::ScalarType` mismatch under NVCC.
**Root Cause:** Per-operator ATen includes (`ATen/ops/*.h`) caused namespace aliasing to break in PyTorch 2.10+.
**Fix:** Use `#include <torch/all.h>` everywhere (sets up proper namespace aliasing).

### Bug 8: CUDA Init Before L1 Slab

**Symptom:** `cudaMemcpyAsync(D2H)` to the L1 slab caused illegal memory access.
**Root Cause:** `cudaHostRegister()` was called before CUDA was initialized. It returned `cudaSuccess` but silently did nothing.
**Fix:** Move CUDA initialization (`cudaFree(nullptr)` per device) before `CacheEngine` construction.

---

## Configuration & Deployment

### Launch Scripts

**C++ LMCache server** (`launch_lmc_cpp_server.sh`):
```bash
LMCache-repo/csrc/server/build/lmcache-server \
    --host 0.0.0.0 \
    --port 15555 \
    --chunk-size 256 \
    --l1-capacity-gib 64 \
    --max-workers 16
```

**vLLM with C++ LMCache** (`launch_vllm_server.sh`):
```bash
vllm serve "$MODEL" \
    -tp 8 --kv-cache-dtype fp8 --block-size 64 \
    --max-model-len 16384 --enforce-eager \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnectorDynamic",
        "kv_connector_extra_config":{"lmcache.mp.port":15555,
        "lmcache.mp.use_two_stage": true}}'
```

### Log Locations

When using nohup:
- LMCache server: `/tmp/lmc_out.log` (or wherever redirected)
- vLLM server: `/disc/data1/riggins/hover/vllm_server.log`

### Process Management

```bash
# Kill all related processes
kill -9 $(pgrep -f lmcache-server) $(pgrep -f "vllm serve") 2>/dev/null

# Verify GPU memory is freed
nvidia-smi
```

---

## Known Limitations & Future Work

### Not Implemented

1. **L2 Adapter** — `l2_adapter.h` is an abstract interface only. No Redis, filesystem, or NitroFS backend.
   - The `run_prefetch_load()` method is a stub returning 0
   - L2 lookup/lock/load paths in SYNC_LOOKUP are commented out

2. **Telemetry** — No equivalent of the Python server's telemetry system (START/END events, span correlation, JSONL output).

3. **Blend Operations** — `CB_*` request types (14-21) are defined in `types.h` but no handlers are registered.

4. **Hot Reload** — Server must be restarted for any configuration changes.

5. **Graceful Shutdown** — CUDA cleanup on shutdown is best-effort. IPC handles are leaked (cleaned up by OS on exit).

### Potential Improvements

1. **Memory Pool for slot mapping** — `get_slot_mapping_tensor()` currently does `cudaMalloc`/`cudaMemcpy`/`cudaFree` per call. A pre-allocated ring buffer would avoid allocation overhead.

2. **Concurrent stores to different GPUs** — Currently, the transfer lock serializes all transfers per GPU. Multiple GPUs can transfer concurrently but each GPU is serial.

3. **L1 eviction policy** — Current LRU eviction is simple. Could add frequency-based or cost-aware eviction.

4. **Error recovery** — A CUDA error on one device currently corrupts all CUDA state. Could add per-device error isolation.

5. **Metrics endpoint** — No Prometheus/HTTP metrics. Could add a lightweight HTTP server or ZMQ stats channel.

---

## Testing

### Quick Smoke Test

```bash
# Terminal 1: Start C++ LMCache server
bash launch_lmc_cpp_server.sh

# Terminal 2: Start vLLM
bash launch_vllm_server.sh 2>&1 | tee vllm_server.log

# Terminal 3: Run benchmark (wait for vLLM to be ready)
bash easy_bench.sh

# Check logs
cat /tmp/lmc_out.log   # Should show REGISTER, SYNC_LOOKUP, STORE operations
```

### Expected Log Output (Success)

```
lmcache-server (pure C++)
  bind: tcp://0.0.0.0:15555
  chunk_size: 256
  L1 capacity: 64 GiB
  max_workers: 16
CUDA initialized: 8 device(s)
Starting server...
LMCache C++ server is running on tcp://0.0.0.0:15555
CacheEngine: registered KV cache for instance XXXXX (61 layers, model=..., ws=1)
  [repeated 8 times, one per TP worker]
SYNC_LOOKUP[req-id]: hit_count=0, found_count=0, ws=1
CacheEngine: stored 23 chunks (5888 tokens)
  [repeated 8 times, one per TP worker]
```
