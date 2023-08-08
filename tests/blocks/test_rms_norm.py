from pyblocks.blocks.norm.rms_norm import RMSNORMBlock
from pyblocks.native.blocks.norm.rms_norm import RMSNorm
import torch
import pytest

@pytest.mark.parametrize("val",[-1,1,3,123.4])
def test_rms_constant(val):
    dim = 21
    batch_size = 127
    lay1 = RMSNORMBlock(dim,4)
    lay2 = RMSNorm(dim)
    lay2.from_weights(lay1.params.data)
    x1 = val * torch.ones((dim,batch_size),dtype=torch.float32)
    x1.requires_grad = True
    x2 = x1.detach().clone()
    x2.requires_grad = True
    out1 = lay1(x1)
    out2 = lay2(x2)
    assert torch.allclose(out1,out2)
    out1.sum().backward()
    out2.sum().backward()
    assert torch.allclose(lay1.params.grad,lay2.params.grad)
    assert torch.allclose(x1.grad,x2.grad,atol=10**-6,rtol=10**-6)

@pytest.mark.parametrize("val",[-11,-3,1,3,123.4])
def test_rms_variable(val):
    dim = 21
    batch_size = 127
    lay1 = RMSNORMBlock(dim,4)
    lay2 = RMSNorm(dim)
    lay2.from_weights(lay1.params.data)
    shifts = torch.arange(0,dim,dtype=torch.float32).reshape(-1,1)
    x1 = (val * torch.ones((dim,batch_size),dtype=torch.float32) + shifts).detach().clone()
    x1.requires_grad = True
    x2 = x1.detach().clone()
    x2.requires_grad = True
    out1 = lay1(x1)
    out2 = lay2(x2)
    assert torch.allclose(out1,out2)
    out1.sum().backward()
    out2.sum().backward()
    assert torch.allclose(x1.grad,x2.grad)