# LMCache Pure C++ Server — Implementation Complete

## Status: WORKING

The pure C++ LMCache server is **implemented and operational**. It has been tested with DeepSeek-V3.1 on 8x NVIDIA H20 GPUs with TP=8, fp8 KV cache, MLA attention, and successfully handles store/retrieve/lookup operations from vLLM clients. Tested with 20 concurrent requests sent in two rounds.

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│           lmcache-server binary              │
│        (LMCache-repo/csrc/server/)           │
├─────────────────────────────────────────────┤
│  main.cpp — CLI, signal handlers, wiring    │
│  ├─ 13 IRequestHandler classes              │
│  ├─ CacheEngine orchestrator                │
│  └─ MessageQueueServer (ZMQ ROUTER)         │
├─────────────────────────────────────────────┤
│  CacheEngine (cache_engine.cpp)             │
│  ├─ GPUContext — per-GPU IPC + streams      │
│  ├─ L1Store — mmap slab + cudaHostRegister  │
│  ├─ TokenHasher — BLAKE3 rolling hashes     │
│  └─ SessionManager — per-request state      │
├─────────────────────────────────────────────┤
│  Wire Protocol (wire_protocol.cpp)          │
│  ├─ msgpack encode/decode                   │
│  └─ Python pickle parser (CudaIPCWrapper)   │
├─────────────────────────────────────────────┤
│  Reused C++ Sources                         │
│  ├─ mem_kernels.cu — CUDA transfer kernels  │
│  ├─ bitmap.cpp — bitwise operations         │
│  └─ ttl_lock.cpp — TTL-based locking        │
├─────────────────────────────────────────────┤
│  Dependencies                               │
│  ├─ libtorch (ATen tensors, CUDA guards)    │
│  ├─ libzmq (ZMQ ROUTER socket)             │
│  ├─ msgpack-cxx (wire serialization)        │
│  ├─ BLAKE3 (token hashing)                  │
│  └─ CUDA runtime + driver APIs              │
└─────────────────────────────────────────────┘
```

---

## File Inventory (5,238 lines total)

| File | Lines | Purpose |
|------|-------|---------|
| `wire_protocol.cpp` | 905 | msgpack encode/decode + Python pickle parser |
| `cache_engine.cpp` | 602 | Store/retrieve/lookup orchestration |
| `l1_store.cpp` | 580 | Mmap slab storage with state machine |
| `mq_server.cpp` | 467 | ZMQ ROUTER + eventfd + thread pool |
| `main.cpp` | 448 | Entry point, handlers, CLI |
| `gpu_context.cpp` | 432 | Per-GPU IPC, streams, slot mapping |
| `types.h` | 280 | All shared type definitions |
| `tensor_bridge.cpp` | 202 | ATen ↔ raw CUDA bridge |
| `wire_protocol.h` | 164 | Encoder/Decoder declarations |
| `cache_engine.h` | 165 | CacheEngine class |
| `session_manager.cpp` | 135 | Per-request hash cache |
| `token_hasher.cpp` | 128 | BLAKE3 rolling prefix hash |
| `gpu_context.h` | 125 | GPUContext class |
| `l1_store.h` | 112 | L1Store abstract interface |
| `l2_adapter.h` | 107 | L2 interface (placeholder) |
| `session_manager.h` | 95 | SessionManager class |
| `tensor_bridge.h` | 89 | Tensor bridge declarations |
| `mq_server.h` | 88 | MessageQueueServer class |
| `token_hasher.h` | 73 | TokenHasher class |
| `types.cpp` | 41 | ipc_key_to_object_keys() |

---

## Key Technical Details

### 1. Wire Protocol Compatibility

The C++ server is **wire-compatible** with existing Python vLLM clients. No client changes needed.

- ZMQ ROUTER socket (same as Python `MessageQueueServer`)
- msgpack serialization matching `msgspec.msgpack` format
- CudaIPCWrapper parsed from Python pickle protocol 4 (Ext type code 1)
- IPCCacheEngineKey decoded as msgpack MAP (dict with string keys)

### 2. CUDA IPC Handle Management

- Uses `c10::cuda::CUDACachingAllocator::getIpcDevPtr()` (not raw `cudaIpcOpenMemHandle`)
- Supports PyTorch 2.10+ expanded 66-byte handle format
- IPC tensors stored as class members to keep shared_ptr alive (critical for handle lifetime)
- `c10::cuda::CUDAGuard` + `at::cuda::CUDAStreamGuard` for device/stream scoping

### 3. GPU KV Transfer

Reuses the existing `multi_layer_kv_transfer` CUDA kernel from `csrc/mem_kernels.cu`:
- Store: GPU KV cache → tmp_buffer → L1 slab (D2H)
- Retrieve: L1 slab → tmp_buffer → GPU KV cache (H2D)
- Supports MLA format (format=3, `NL_X_NB_BS_HS`) and all 5 other vLLM/SGLang formats

### 4. L1 Slab Storage

- mmap'd memory region + `cudaHostRegister` for zero-copy DMA
- State machine: Free → Writing → Ready → Reading → Ready
- LRU eviction when capacity exhausted
- TTL-based read lock management via existing C++ `TTLLock`

### 5. Token Hashing

- BLAKE3 rolling prefix hash (same algorithm as Python server)
- Session-based hash caching for incremental updates

---

## Build & Run

```bash
# Build
cd LMCache-repo/csrc/server
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
./lmcache-server \
    --host 0.0.0.0 \
    --port 15555 \
    --chunk-size 256 \
    --l1-capacity-gib 64 \
    --max-workers 16
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8001` | Bind port |
| `--chunk-size` | `256` | Tokens per chunk |
| `--l1-capacity-gib` | `8` | L1 slab capacity in GiB |
| `--max-workers` | `4` | Thread pool workers |
| `--hugepages` | off | Use huge pages for L1 slab |
| `--no-cuda-host-register` | off | Disable cudaHostRegister |

---

## Dependencies

| Dependency | Version | How Resolved |
|------------|---------|-------------|
| libtorch | from pip PyTorch | `torch.utils.cmake_prefix_path` |
| CUDA Toolkit | 12.x | system install |
| libzmq | 4.3.5 | system .so + FetchContent headers |
| cppzmq | 4.10.0 | FetchContent (header-only) |
| msgpack-cxx | 6.1.1 | FetchContent (header-only, no Boost) |
| BLAKE3 | 1.8.2 | FetchContent (static lib with SIMD) |
| c10_cuda | from libtorch | for `getIpcDevPtr()` |

---

## Known Limitations

1. **L2 adapter not implemented** — only L1 (host memory) caching; L2 (Redis/NitroFS) is a placeholder
2. **No telemetry** — the Python server's telemetry system is not ported
3. **Blend operations not implemented** — CB_* request types are defined but not handled
4. **No hot-reload** — server must be restarted for config changes

---

## Bugs Fixed During Development

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `cudaIpcOpenMemHandle` invalid argument | PyTorch 2.10+ uses 66-byte expanded handle | Use `getIpcDevPtr()` instead |
| Segfault in `getIpcDevPtr` | CUDA caching allocator not initialized | Call `CUDACachingAllocator::init()` before engine |
| CUDA illegal memory access in kernel | IPC tensor pointers dangled after constructor | Store IPC tensors as class members |
| IPCCacheEngineKey decode failure | msgspec encodes as MAP, not array | Handle both MAP and ARRAY formats |
| `uint8` dtype unknown | fp8 KV cache uses `torch.uint8` | Map `"uint8"` → `DType::Int8` → `at::ScalarType::Byte` |
| Empty `device_uuid` | Pickle doesn't contain "GPU-" prefix | Parse bare UUID, prepend "GPU-" |
| `c10::ScalarType` namespace mismatch | Per-operator ATen includes | Use `#include <torch/all.h>` |
