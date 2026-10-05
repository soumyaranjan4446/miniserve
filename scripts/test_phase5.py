import torch
from paged_kv import PagedKVCache
from scheduler import Request,ContinuousBatchingScheduler

def main():
    cache=PagedKVCache(1,16,4,1,2,4); s=ContinuousBatchingScheduler(2,cache)
    for i,n in enumerate([5,7,3]): s.add_request(Request(str(i),torch.zeros(1,n,dtype=torch.long),4))
    s.admit_requests(); print("running",s.running_count(),"waiting",s.waiting_count(),"allocated",cache.allocated_blocks()); s.finish_request("0"); print("running",s.running_count(),"waiting",s.waiting_count(),"free",cache.free_blocks())
if __name__=="__main__": main()
