import torch
from sampling import SamplingParams,create_generator,sample

def main():
    logits=torch.tensor([[1.,2.,3.]])
    assert sample(logits,SamplingParams(temperature=0)) .item()==2
    p=SamplingParams(temperature=1.,seed=7); a=sample(logits,p,generator=create_generator(7,"cpu")); b=sample(logits,p,generator=create_generator(7,"cpu")); assert a.item()==b.item(); print("sampling ok")
if __name__=="__main__": main()
