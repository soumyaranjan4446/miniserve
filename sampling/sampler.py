from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    stop_token_id: Optional[int] = None
    stop_strings: Optional[list[str]] = None
    max_new_tokens: int = 32

    def validate(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")


def create_generator(seed, device):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def greedy_sample(logits):
    return logits.argmax(dim=-1)


def apply_temperature(logits, temperature):
    if temperature == 0:
        return logits
    return logits / temperature


def apply_repetition_penalty(logits, generated_tokens, penalty):
    if penalty == 1.0 or not generated_tokens:
        return logits
    out = logits.clone()
    tokens = torch.as_tensor(generated_tokens, device=logits.device, dtype=torch.long)
    if out.ndim == 1:
        selected = out[tokens]
        out[tokens] = torch.where(selected < 0, selected * penalty, selected / penalty)
    else:
        selected = out[:, tokens]
        out[:, tokens] = torch.where(selected < 0, selected * penalty, selected / penalty)
    return out


def apply_top_k(logits, top_k):
    if top_k is None or top_k >= logits.shape[-1]:
        return logits
    values = torch.topk(logits, top_k, dim=-1).values
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, filtered)


def sample(logits, params: SamplingParams, generated_tokens=None, generator=None):
    params.validate()
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if params.temperature == 0:
        result = greedy_sample(logits)
        return result.squeeze(0) if squeeze else result
    x = apply_repetition_penalty(logits, generated_tokens or [], params.repetition_penalty)
    x = apply_temperature(x, params.temperature)
    x = apply_top_k(x, params.top_k)
    x = apply_top_p(x, params.top_p)
    probs = torch.softmax(x, dim=-1)
    result = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return result.squeeze(0) if squeeze else result
