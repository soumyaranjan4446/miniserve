from __future__ import annotations

import time
from pathlib import Path

import torch

from .hardware import get_hardware_info


def run_baseline(model_name: str = "Qwen/Qwen2.5-0.5B", prompt: str = "Explain transformer inference in simple terms.", max_new_tokens: int = 32) -> dict:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for the HF baseline") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = int(output.shape[1] - inputs["input_ids"].shape[1])
    return {
        "hardware": get_hardware_info(),
        "model": model_name,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": generated,
        "e2e_seconds": elapsed,
        "tokens_per_second": generated / elapsed if elapsed > 0 else 0.0,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
    }
