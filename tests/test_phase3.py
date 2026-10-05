import torch
from sampling import SamplingParams,create_generator,sample

def test_greedy(): assert sample(torch.tensor([[1.,5.,2.]]),SamplingParams(temperature=0)).item()==1
def test_seed():
    l=torch.randn(1,10); p=SamplingParams(seed=5); assert sample(l,p,generator=create_generator(5,"cpu")).item()==sample(l,p,generator=create_generator(5,"cpu")).item()
def test_topk1(): assert sample(torch.tensor([[1.,5.,2.]]),SamplingParams(top_k=1)).item()==1
