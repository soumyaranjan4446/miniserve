import torch
from speculative import SpeculativeDecoder
class Toy(torch.nn.Module):
    def __init__(self,vocab=8,step=1): super().__init__(); self.vocab=vocab; self.step=step; self.anchor=torch.nn.Parameter(torch.zeros(1))
    def forward(self,ids):
        b,t=ids.shape; logits=torch.full((b,t,self.vocab),-100.,device=ids.device); nxt=(ids+self.step)%self.vocab; logits.scatter_(2,nxt.unsqueeze(-1),10.); return logits

def test_exact_greedy_when_models_match():
    d=SpeculativeDecoder(Toy(),Toy()); inp=torch.tensor([[0,1]]); out,stats=d.generate(inp,5,k=3); expected=torch.tensor([[0,1,2,3,4,5,6]]); assert torch.equal(out,expected); assert stats.acceptance_rate==1.
def test_correction():
    d=SpeculativeDecoder(Toy(step=2),Toy(step=1)); inp=torch.tensor([[0,1]]); out,_=d.generate(inp,4,k=2); assert torch.equal(out,torch.tensor([[0,1,2,3,4,5]]))
