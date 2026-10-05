import torch
from profiling.benchmark_before_after import benchmark,compare

def test_benchmark():
    x=torch.tensor(1.); r=benchmark(lambda: x+1,warmup=1,steps=2); assert r["mean_ms"]>=0 and r["tokens_per_second"]>0
