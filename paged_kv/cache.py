from __future__ import annotations

import torch

from .allocator import BlockAllocator
from .block_table import BlockTable


class PagedKVCache:
    """Reference paged KV store. Physical layout: [layer, block, kv_head, offset, head_dim]."""
    def __init__(self, num_layers, num_blocks, block_size, batch_size, num_kv_heads, head_dim, device=None, dtype=torch.float32):
        if batch_size != 1:
            # The phase-5 memory primitive is request-scoped; continuous batching maps
            # each request to its own table. Keeping this explicit avoids a false API.
            raise ValueError("PagedKVCache is request-scoped; use one cache pool for many request tables")
        self.num_layers, self.num_blocks, self.block_size = num_layers, num_blocks, block_size
        self.num_kv_heads, self.head_dim = num_kv_heads, head_dim
        self.device, self.dtype = device, dtype
        self.allocator = BlockAllocator(num_blocks)
        shape = (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        self.key_cache = torch.empty(shape, device=device, dtype=dtype)
        self.value_cache = torch.empty(shape, device=device, dtype=dtype)
        self.block_tables = {}

    def create_request(self, request_id):
        if request_id in self.block_tables: raise ValueError(f"request already exists: {request_id}")
        self.block_tables[request_id] = BlockTable(self.block_size)

    def delete_request(self, request_id):
        table = self._get(request_id)
        self.allocator.free(table.physical_blocks())
        del self.block_tables[request_id]

    def _get(self, request_id):
        if request_id not in self.block_tables: raise KeyError(f"unknown request: {request_id}")
        return self.block_tables[request_id]

    def ensure_capacity(self, request_id, num_tokens):
        table = self._get(request_id)
        additional = table.required_blocks(num_tokens) - table.logical_block_count()
        if additional > 0: table.add_blocks(self.allocator.allocate(additional))

    def write(self, request_id, layer_idx, key, value, start_position):
        if key.ndim != 4 or value.shape != key.shape: raise ValueError("key/value must be [B, KV, T, D]")
        if key.shape[0] != 1 or key.shape[1] != self.num_kv_heads or key.shape[3] != self.head_dim: raise ValueError("KV shape mismatch")
        table = self._get(request_id)
        end = start_position + key.shape[2]
        self.ensure_capacity(request_id, end)
        for i in range(key.shape[2]):
            pos = start_position + i
            block = table.blocks[pos // self.block_size]
            offset = pos % self.block_size
            self.key_cache[layer_idx, block, :, offset, :].copy_(key[0, :, i, :])
            self.value_cache[layer_idx, block, :, offset, :].copy_(value[0, :, i, :])
        table.update_token_count(max(table.num_tokens, end))

    def read(self, request_id, layer_idx, num_tokens=None):
        table = self._get(request_id)
        n = table.num_tokens if num_tokens is None else num_tokens
        if n < 0 or n > table.num_tokens: raise ValueError("invalid num_tokens")
        keys, values = [], []
        for pos in range(n):
            block = table.blocks[pos // self.block_size]; offset = pos % self.block_size
            keys.append(self.key_cache[layer_idx, block, :, offset, :])
            values.append(self.value_cache[layer_idx, block, :, offset, :])
        if not keys:
            shape = (1, self.num_kv_heads, 0, self.head_dim)
            return torch.empty(shape, device=self.device, dtype=self.dtype), torch.empty(shape, device=self.device, dtype=self.dtype)
        return torch.stack(keys, dim=1).unsqueeze(0), torch.stack(values, dim=1).unsqueeze(0)

    def sequence_length(self, request_id): return self._get(request_id).num_tokens
    def block_count(self, request_id): return self._get(request_id).logical_block_count()
    def free_blocks(self): return self.allocator.free_count()
    def allocated_blocks(self): return self.allocator.allocated_count()
