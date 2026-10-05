from __future__ import annotations

import json


def inspect_model(model_name: str = "Qwen/Qwen2.5-0.5B") -> dict:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required") from exc
    cfg = AutoConfig.from_pretrained(model_name)
    fields = [
        "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "vocab_size", "rope_theta", "rms_norm_eps", "max_position_embeddings",
    ]
    out = {k: getattr(cfg, k, None) for k in fields}
    if out["hidden_size"] and out["num_attention_heads"]:
        out["head_dim"] = out["hidden_size"] // out["num_attention_heads"]
    return out


if __name__ == "__main__":
    print(json.dumps(inspect_model(), indent=2))
