from __future__ import annotations
import argparse
from pathlib import Path
import torch
from bench.harness import summarize
from model import MiniQwen, load_qwen_config
from weights import load_hf_weights
from sampling import SamplingParams
from engine import MiniServeRuntime

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="Qwen/Qwen2.5-0.5B"); p.add_argument("--steps",type=int,default=16); p.add_argument("--output",default="results/phase9_trace.json"); args=p.parse_args()
    from transformers import AutoTokenizer
    device="cuda" if torch.cuda.is_available() else "cpu"; cfg=load_qwen_config(args.model); model=MiniQwen(cfg); load_hf_weights(model,args.model); model.to(device).eval(); tok=AutoTokenizer.from_pretrained(args.model)
    enc=tok("Explain transformer inference.",return_tensors="pt")["input_ids"].to(device)
    activities=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if device=="cuda" else [])
    with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:
        with torch.no_grad():
            cache=__import__("kv_cache").StaticKVCache(1,cfg.num_hidden_layers,cfg.num_key_value_heads,enc.shape[1]+args.steps,cfg.head_dim,device,next(model.parameters()).dtype)
            logits=model(enc,kv_cache=cache)
            for _ in range(args.steps):
                token=logits[:,-1,:].argmax(-1,keepdim=True); logits=model(token,kv_cache=cache); prof.step()
    print(prof.key_averages().table(sort_by="self_cpu_time_total",row_limit=40)); Path(args.output).parent.mkdir(parents=True,exist_ok=True); prof.export_chrome_trace(args.output); print("Trace:",args.output)

if __name__=="__main__": main()
