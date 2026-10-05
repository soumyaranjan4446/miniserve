from __future__ import annotations

import torch


def load_hf_weights(model, model_name: str = "Qwen/Qwen2.5-0.5B"):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required") from exc
    hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    raw_source = hf.state_dict()
    source = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in raw_source.items()
    }
    target = model.state_dict()
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    shape_mismatch = []
    compatible = {}
    for key in target:
        if key in source:
            if target[key].shape != source[key].shape:
                shape_mismatch.append((key, tuple(target[key].shape), tuple(source[key].shape)))
            else:
                compatible[key] = source[key]
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(f"HF parity mapping failed: missing={missing[:10]}, unexpected={unexpected[:10]}, shape_mismatch={shape_mismatch[:10]}")
    model.load_state_dict(compatible, strict=True)
    return model
