import torch,pytest
from paged_kv import BlockAllocator,BlockTable,PagedKVCache
from scheduler import Request,ContinuousBatchingScheduler

def test_allocator():
    a=BlockAllocator(4); b=a.allocate(2); assert a.allocated_count()==2; a.free(b); assert a.free_count()==4
    with pytest.raises(RuntimeError): a.free(b)
def test_table():
    t=BlockTable(4); t.add_blocks([7,2]); assert t.block_for_token(0)==7; assert t.block_for_token(4)==2; assert t.required_blocks(5)==2
def test_cache_roundtrip():
    c=PagedKVCache(1,8,4,1,2,4); c.create_request("a"); k=torch.randn(1,2,5,4); v=torch.randn_like(k); c.write("a",0,k,v,0); rk,rv=c.read("a",0); assert torch.allclose(k,rk); assert torch.allclose(v,rv); c.delete_request("a"); assert c.allocated_blocks()==0
def test_scheduler_recycles():
    c=PagedKVCache(1,8,4,1,2,4); s=ContinuousBatchingScheduler(1,c); s.add_request(Request("a",torch.zeros(1,5,dtype=torch.long),2)); s.add_request(Request("b",torch.zeros(1,5,dtype=torch.long),2)); s.admit_requests(); assert s.running_count()==1 and s.waiting_count()==1; s.finish_request("a"); assert s.running_count()==1 and s.waiting_count()==0
