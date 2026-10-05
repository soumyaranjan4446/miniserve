import torch
from paged_kv import PagedKVCache

def main():
    for block_size in (8,16,32):
        cache=PagedKVCache(1,256,block_size,1,2,64)
        cache.create_request("r")
        cache.ensure_capacity("r",100)
        allocated=cache.block_count("r")
        waste=(allocated*block_size-100)/(allocated*block_size)
        print({"block_size":block_size,"blocks":allocated,"fragmentation":waste})

if __name__=="__main__": main()
