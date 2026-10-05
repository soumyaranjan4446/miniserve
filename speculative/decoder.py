from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class SpeculativeStats:
    draft_tokens:int=0
    accepted_tokens:int=0
    target_corrections:int=0
    target_forwards:int=0
    @property
    def acceptance_rate(self): return self.accepted_tokens/self.draft_tokens if self.draft_tokens else 0.0

class SpeculativeDecoder:
    """Reference greedy speculative decoding.

    Models are callables returning logits [B,T,V]. This version intentionally
    favors transparent correctness over KV-cache/kernel optimization.
    """
    def __init__(self,draft_model,target_model,device=None):
        self.draft_model=draft_model.eval(); self.target_model=target_model.eval(); self.device=device

    @torch.no_grad()
    def generate(self,input_ids,max_new_tokens,k=4,stop_token_id=None):
        if k<=0 or max_new_tokens<0: raise ValueError("invalid k/max_new_tokens")
        seq=input_ids.to(self.device or next(self.target_model.parameters()).device)
        stats=SpeculativeStats()
        while seq.shape[1]-input_ids.shape[1] < max_new_tokens:
            remaining=max_new_tokens-(seq.shape[1]-input_ids.shape[1]); kk=min(k,remaining)
            draft=seq
            proposals=[]
            for _ in range(kk):
                logits=self.draft_model(draft)[:,-1,:]
                tok=logits.argmax(-1,keepdim=True); proposals.append(tok); draft=torch.cat([draft,tok],1)
            proposed=torch.cat(proposals,1); stats.draft_tokens += kk
            verified=torch.cat([seq,proposed],1)
            target_logits=self.target_model(verified); stats.target_forwards += 1
            preds=target_logits[:,seq.shape[1]-1:seq.shape[1]-1+kk,:].argmax(-1)
            accepted=0
            for i in range(kk):
                if preds[:,i].item()!=proposed[:,i].item(): break
                accepted+=1
            if accepted:
                seq=torch.cat([seq,proposed[:,:accepted]],1); stats.accepted_tokens += accepted
                if stop_token_id is not None and int(proposed[0,accepted-1])==stop_token_id: break
            if accepted<kk:
                replacement=preds[:,accepted:accepted+1]
                seq=torch.cat([seq,replacement],1); stats.target_corrections += 1
                if stop_token_id is not None and int(replacement[0,0])==stop_token_id: break
            else:
                if seq.shape[1]-input_ids.shape[1] >= max_new_tokens: break
                extra=target_logits[:,-1,:].argmax(-1,keepdim=True)
                seq=torch.cat([seq,extra],1)
                if stop_token_id is not None and int(extra[0,0])==stop_token_id: break
        return seq,stats
