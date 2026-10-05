import random
import torch
from paged_kv import PagedKVCache
from scheduler import Request, ContinuousBatchingScheduler

def test_500_request_scheduler_stress():
    rng=random.Random(7)
    cache=PagedKVCache(1,64,8,1,1,4)
    scheduler=ContinuousBatchingScheduler(8,cache)
    requests=[Request(str(i),torch.zeros(1,rng.randint(1,32),dtype=torch.long),rng.randint(1,8)) for i in range(500)]
    for r in requests: scheduler.add_request(r)
    completed=0
    while not scheduler.is_empty():
        active=scheduler.step()
        for r in list(active):
            # Prompt KV blocks were allocated at admission.
            if r.output_length < r.max_new_tokens:
                r.add_token(0)
            if r.output_length >= r.max_new_tokens:
                scheduler.finish_request(r.request_id); completed += 1
    assert completed==500
    assert scheduler.waiting_count()==0
    assert scheduler.running_count()==0
    assert cache.allocated_blocks()==0
    assert cache.free_blocks()==64
