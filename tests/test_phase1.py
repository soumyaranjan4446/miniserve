import torch
from model import MiniQwen,QwenConfig

def test_model_shape():
    c=QwenConfig(32,16,2,4,2,32,max_position_embeddings=64); m=MiniQwen(c); y=m(torch.randint(0,32,(2,5))); assert y.shape==(2,5,32)

def test_tied_head():
    c=QwenConfig(32,16,1,4,2,32); m=MiniQwen(c); assert m.lm_head.weight.data_ptr()==m.embed_tokens.weight.data_ptr()
