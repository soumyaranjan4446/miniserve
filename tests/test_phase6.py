import torch
from quantization import QLinear,quantize_per_output_channel

def test_quantization_shape():
    w=torch.randn(4,6); q,s=quantize_per_output_channel(w); assert q.dtype==torch.int8 and q.shape==w.shape and s.shape==(4,)
def test_qlinear_close():
    torch.manual_seed(0); l=torch.nn.Linear(8,4); q=QLinear.from_linear(l); x=torch.randn(3,8); assert torch.allclose(l(x),q(x),atol=0.08,rtol=0.08)
def test_storage_reduction():
    l=torch.nn.Linear(8,4); q=QLinear.from_linear(l); assert q.weight_q.numel()*q.weight_q.element_size() < l.weight.numel()*l.weight.element_size()
