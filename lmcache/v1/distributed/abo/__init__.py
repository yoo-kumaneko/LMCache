# SPDX-License-Identifier: Apache-2.0

"""
ABO KV compression module.

Core components:
- StagingPool: pinned memory buffer pool for GPU↔CPU data transfer
- ABOCodecFactory: creates abokvpress.HuffmanCodec directly (no wrapper)
- CompressedMemoryObj: MemoryObj subclass with compression support
"""
