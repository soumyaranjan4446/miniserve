from __future__ import annotations

import torch
import torch.nn as nn

from .config import QwenConfig
from .layers import RMSNorm, RotaryEmbedding, TransformerBlock


class MiniQwen(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, kv_cache=None, position_ids=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if position_ids is None:
            start = kv_cache.get_seq_length() if kv_cache is not None else 0
            position_ids = torch.arange(start, start + input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        x = self.embed_tokens(input_ids)
        for idx, layer in enumerate(self.layers):
            x = layer(x, position_ids, self.rotary_emb, kv_cache, idx)
        if kv_cache is not None:
            kv_cache.advance(input_ids.shape[1])
        return self.lm_head(self.norm(x))
