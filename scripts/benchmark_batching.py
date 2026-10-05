import time
import torch
from batching import BatchRequest, StaticBatch

def main():
    print("Static batching / padding benchmark")
    workloads={"batch_1":[128],"batch_2":[128,128],"batch_4":[128]*4,"mixed_4":[128,512,64,256]}
    for name,lengths in workloads.items():
        batch=StaticBatch([BatchRequest(str(i),torch.zeros(1,n,dtype=torch.long)) for i,n in enumerate(lengths)],0)
        start=time.perf_counter(); batch.prepare(); elapsed=(time.perf_counter()-start)*1000
        print(name,{**batch.statistics(),"prepare_ms":elapsed})

if __name__=="__main__": main()
