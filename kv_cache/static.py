from __future__ import annotations

import torch

from .base import KVCache


class StaticKVCache(KVCache):
    """Contiguous cache for one request or a homogeneous batch."""
    def __init__(self, batch_size, num_layers, num_kv_heads, max_seq_len, head_dim, device=None, dtype=torch.float32):
        if min(batch_size, num_layers, num_kv_heads, max_seq_len, head_dim) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.key_cache = torch.empty(shape, device=device, dtype=dtype)
        self.value_cache = torch.empty(shape, device=device, dtype=dtype)
        self.seq_len = 0
        self.pending_end = 0

    def update(self, layer_idx, key, value):
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("key/value must be [B, KV, T, D] with equal shapes")
        if key.shape[0] != self.batch_size or key.shape[1] != self.num_kv_heads or key.shape[3] != self.head_dim:
            raise ValueError("key/value shape does not match cache")
        t = key.shape[2]
        end = self.seq_len + t
        if self.pending_end not in (self.seq_len, end):
            raise RuntimeError("cache update length changed within one forward pass")
        if end > self.max_seq_len:
            raise RuntimeError(f"KV cache overflow: {end} > {self.max_seq_len}")
        self.key_cache[layer_idx, :, :, self.seq_len:end, :].copy_(key)
        self.value_cache[layer_idx, :, :, self.seq_len:end, :].copy_(value)
        self.pending_end = end

    def get(self, layer_idx):
        end = max(self.seq_len, self.pending_end)
        return self.key_cache[layer_idx, :, :, :end, :], self.value_cache[layer_idx, :, :, :end, :]

    def advance(self, tokens: int):
        if tokens < 0 or self.seq_len + tokens > self.max_seq_len:
            raise ValueError("invalid cache advance")
        expected = self.seq_len + tokens
        if self.pending_end != expected:
            raise ValueError(f"advance({tokens}) does not match pending cache length {self.pending_end}")
        self.seq_len = expected
        self.pending_end = self.seq_len

    def get_seq_length(self):
        return self.seq_len

    def reset(self):
        self.seq_len = 0
        self.pending_end = 0
