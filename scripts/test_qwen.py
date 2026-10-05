from __future__ import annotations
import torch
from model import MiniQwen, load_qwen_config
from weights import load_hf_weights

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name="Qwen/Qwen2.5-0.5B"; cfg=load_qwen_config(name); ours=MiniQwen(cfg).eval(); load_hf_weights(ours,name)
    hf=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float32).eval(); tok=AutoTokenizer.from_pretrained(name)
    ids=tok("Explain KV caching.",return_tensors="pt")["input_ids"]
    with torch.no_grad(): a=ours(ids); b=hf(ids).logits
    diff=(a-b).abs().max().item(); print("max_abs_diff:",diff); assert diff<1e-3
if __name__=="__main__": main()
