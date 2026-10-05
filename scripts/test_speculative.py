import torch
from speculative import SpeculativeDecoder
class Toy(torch.nn.Module):
    def __init__(self,vocab=8): super().__init__(); self.vocab=vocab; self.anchor=torch.nn.Parameter(torch.zeros(1))
    def forward(self,ids):
        b,t=ids.shape; logits=torch.full((b,t,self.vocab),-100.,device=ids.device); nxt=(ids+1)%self.vocab; logits.scatter_(2,nxt.unsqueeze(-1),10.); return logits

def main():
    x=torch.tensor([[0,1,2]]); d=SpeculativeDecoder(Toy(),Toy()); out,stats=d.generate(x,8,3); assert torch.equal(out,torch.tensor([[0,1,2,3,4,5,6,7,0,1,2]])); print(stats)
if __name__=="__main__": main()
