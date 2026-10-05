import torch
import pytest
from batching import BatchRequest,StaticBatch

def test_padding_stats():
    b=StaticBatch([BatchRequest(str(i),torch.zeros(1,n,dtype=torch.long)) for i,n in enumerate([128,512,64,256])],0); assert b.total_prompt_tokens==960; assert b.padded_tokens==2048; assert b.padding_tokens==1088

def test_prepare():
    b=StaticBatch([BatchRequest("a",torch.tensor([[1,2]])),BatchRequest("b",torch.tensor([[3,4]]))],0); x,m=b.prepare(); assert x.tolist()==[[1,2],[3,4]]; assert m.tolist()==[[1,1],[1,1]]
