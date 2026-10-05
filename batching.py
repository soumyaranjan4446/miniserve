from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BatchRequest:
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int = 32

    @property
    def prompt_length(self):
        return int(self.input_ids.shape[-1])


class StaticBatch:
    def __init__(self, requests, pad_token_id: int):
        if not requests:
            raise ValueError("requests must not be empty")
        self.requests = list(requests)
        self.pad_token_id = int(pad_token_id)

    @property
    def batch_size(self): return len(self.requests)
    @property
    def max_prompt_length(self): return max(r.prompt_length for r in self.requests)
    @property
    def total_prompt_tokens(self): return sum(r.prompt_length for r in self.requests)
    @property
    def padded_tokens(self): return self.batch_size * self.max_prompt_length
    @property
    def padding_tokens(self): return self.padded_tokens - self.total_prompt_tokens
    @property
    def padding_waste_ratio(self): return self.padding_tokens / self.padded_tokens if self.padded_tokens else 0.0

    def prepare(self, device=None):
        batch = torch.full((self.batch_size, self.max_prompt_length), self.pad_token_id, dtype=self.requests[0].input_ids.dtype, device=device)
        mask = torch.zeros_like(batch, dtype=torch.long)
        for row, req in enumerate(self.requests):
            ids = req.input_ids.reshape(-1)
            batch[row, :ids.numel()] = ids.to(device)
            mask[row, :ids.numel()] = 1
        return batch, mask

    def statistics(self):
        return {"batch_size": self.batch_size, "actual_tokens": self.total_prompt_tokens, "padded_tokens": self.padded_tokens, "padding_tokens": self.padding_tokens, "padding_waste_ratio": self.padding_waste_ratio}


class StaticBatchRunner:
    """Runs the model only for equal-length prompts.

    MiniQwen intentionally does not accept an attention mask, so silently running
    mixed-length right-padded inputs would be incorrect. Mixed lengths remain valid
    for padding/waste measurement; execution rejects them until a mask-aware model
    path is added.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def run(self, batch: StaticBatch):
        lengths = {r.prompt_length for r in batch.requests}
        if len(lengths) != 1:
            raise ValueError("StaticBatchRunner requires equal-length prompts")
        ids, mask = batch.prepare(next(self.model.parameters()).device)
        logits = self.model(ids)
        return logits, mask
