from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QwenConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must divide KV heads")
        return self.num_attention_heads // self.num_key_value_heads


def load_qwen_config(model_name: str = "Qwen/Qwen2.5-0.5B") -> QwenConfig:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required") from exc
    cfg = AutoConfig.from_pretrained(model_name)
    return QwenConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=getattr(cfg, "max_position_embeddings", 32768),
        rope_theta=getattr(cfg, "rope_theta", 1_000_000.0),
        rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
    )
