import torch
from model import MiniQwen,QwenConfig
from kv_cache import StaticKVCache

def make(): return MiniQwen(QwenConfig(32,16,2,4,2,32,max_position_embeddings=64)).eval()
def test_cache_advances_and_matches():
    m=make(); ids=torch.randint(0,32,(1,6)); c=StaticKVCache(1,2,2,8,4)
    full=m(ids); pre=m(ids[:,:5],kv_cache=c); dec=m(ids[:,5:],kv_cache=c)
    assert c.get_seq_length()==6; assert torch.allclose(full[:,-1],dec[:,-1],atol=1e-5,rtol=1e-5)
