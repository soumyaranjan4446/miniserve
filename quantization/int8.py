from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

def quantize_per_output_channel(weight):
    if weight.ndim != 2: raise ValueError("weight must be 2-D")
    w=weight.detach().float(); absmax=w.abs().amax(dim=1); scale=(absmax/127).clamp_min(1e-8)
    q=torch.round(w/scale[:,None]).clamp(-127,127).to(torch.int8)
    return q, scale

class QLinear(nn.Module):
    def __init__(self,in_features,out_features,bias=True,dtype=torch.float32,device=None):
        super().__init__(); self.in_features=in_features; self.out_features=out_features
        self.register_buffer("weight_q",torch.empty(out_features,in_features,dtype=torch.int8,device=device))
        self.register_buffer("scale",torch.empty(out_features,dtype=torch.float32,device=device))
        self.bias=nn.Parameter(torch.empty(out_features,dtype=dtype,device=device)) if bias else None

    @classmethod
    def from_linear(cls,linear):
        q=cls(linear.in_features,linear.out_features,linear.bias is not None,linear.weight.dtype,linear.weight.device)
        with torch.no_grad():
            w,s=quantize_per_output_channel(linear.weight); q.weight_q.copy_(w); q.scale.copy_(s)
            if linear.bias is not None: q.bias.copy_(linear.bias)
        return q

    def dequantized_weight(self): return self.weight_q.float()*self.scale[:,None]
    def forward(self,x):
        out=F.linear(x.float(),self.weight_q.float(),None)*self.scale
        if self.bias is not None: out=out+self.bias.float()
        return out.to(x.dtype)

def quantize_model_int8(model: nn.Module, exclude: Iterable[str]=()):
    exclude=set(exclude)
    def visit(parent,prefix=""):
        for name,child in list(parent.named_children()):
            full=f"{prefix}.{name}" if prefix else name
            if full in exclude: continue
            if isinstance(child,nn.Linear): setattr(parent,name,QLinear.from_linear(child))
            else: visit(child,full)
    visit(model); return model

def model_storage_bytes(model):
    return sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers()))
