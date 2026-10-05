from __future__ import annotations
import time
import torch

def benchmark(fn,warmup=2,steps=10):
    for _ in range(warmup): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    times=[]
    for _ in range(steps):
        start=time.perf_counter(); fn();
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    return {"mean_ms":sum(times)/len(times)*1000,"tokens_per_second":1/(sum(times)/len(times))}

def compare(before,after):
    b=benchmark(before); a=benchmark(after); return {"before":b,"after":a,"speedup":b["mean_ms"]/a["mean_ms"]}
