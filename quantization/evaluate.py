from __future__ import annotations
import math
import torch

def perplexity(model, batches):
    total_nll=0.0; total_tokens=0
    model.eval()
    with torch.no_grad():
        for input_ids in batches:
            logits=model(input_ids)
            shift_logits=logits[:,:-1,:].float(); shift_labels=input_ids[:,1:]
            loss=torch.nn.functional.cross_entropy(shift_logits.reshape(-1,shift_logits.shape[-1]),shift_labels.reshape(-1),reduction="sum")
            total_nll += float(loss); total_tokens += shift_labels.numel()
    return math.exp(total_nll/total_tokens) if total_tokens else float("inf")

def greedy_token_match(reference, candidate):
    if reference.shape != candidate.shape: return 0.0
    return float((reference==candidate).float().mean().item())
