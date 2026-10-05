from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import torch

from kv_cache import StaticKVCache
from sampling import SamplingParams, create_generator, sample


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    e2e_ms: float


class MiniServeRuntime:
    """Single-request serving adapter around MiniQwen + StaticKVCache."""

    def __init__(self, model, tokenizer, device=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device or next(model.parameters()).device)
        self._dtype = next(model.parameters()).dtype

    def _cache(self, prompt_tokens: int, max_new_tokens: int) -> StaticKVCache:
        c = self.model.config
        return StaticKVCache(
            batch_size=1,
            num_layers=c.num_hidden_layers,
            num_kv_heads=c.num_key_value_heads,
            max_seq_len=prompt_tokens + max_new_tokens,
            head_dim=c.head_dim,
            device=self.device,
            dtype=self._dtype,
        )

    def _token_stopped(self, token_id: int, generated: list[int], params: SamplingParams) -> bool:
        if params.stop_token_id is not None and token_id == params.stop_token_id:
            return True
        if params.stop_strings:
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return any(stop in text for stop in params.stop_strings)
        return False

    @torch.no_grad()
    def _generate(self, prompt: str, params: SamplingParams, on_token=None) -> GenerationResult:
        params.validate()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        prompt_tokens = int(input_ids.shape[1])
        cache = self._cache(prompt_tokens, params.max_new_tokens)

        start = time.perf_counter()
        logits = self.model(input_ids, kv_cache=cache)
        generated: list[int] = []
        generator = create_generator(params.seed, self.device)
        first_token_time = None

        for _ in range(params.max_new_tokens):
            next_token = sample(logits[:, -1, :], params, generated_tokens=generated, generator=generator)
            token_id = int(next_token[0].item())
            generated.append(token_id)
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if on_token is not None:
                on_token(token_id)
            if self._token_stopped(token_id, generated, params):
                break
            next_input = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
            logits = self.model(next_input, kv_cache=cache)

        end = time.perf_counter()
        first_token_time = first_token_time or end
        return GenerationResult(
            token_ids=generated,
            text=self.tokenizer.decode(generated, skip_special_tokens=True),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(generated),
            ttft_ms=(first_token_time - start) * 1000.0,
            e2e_ms=(end - start) * 1000.0,
        )

    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        return self._generate(prompt, params)

    @torch.no_grad()
    def stream(self, prompt: str, params: SamplingParams) -> Iterator[dict]:
        """Yield each token immediately after its decode step."""
        params.validate()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        prompt_tokens = int(input_ids.shape[1])
        cache = self._cache(prompt_tokens, params.max_new_tokens)
        logits = self.model(input_ids, kv_cache=cache)
        generated: list[int] = []
        generator = create_generator(params.seed, self.device)

        for _ in range(params.max_new_tokens):
            next_token = sample(logits[:, -1, :], params, generated_tokens=generated, generator=generator)
            token_id = int(next_token[0].item())
            generated.append(token_id)
            yield {
                "token_id": token_id,
                "text": self.tokenizer.decode([token_id], skip_special_tokens=True),
            }
            if self._token_stopped(token_id, generated, params):
                break
            next_input = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
            logits = self.model(next_input, kv_cache=cache)
