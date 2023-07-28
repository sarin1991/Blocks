from pyblocks.blocks.dyad import DyadBlock
from pyblocks.native.blocks.dyad import Dyad as PyDyad
import torch
import pytest

@pytest.mark.parametrize("has_bias",[True, False])
@pytest.mark.parametrize("val",[-1,1,3,123.4])
def test_simple_dyad(has_bias,val):
    dim_in = 64
    dim_out = 64
    dyad_dim = 64
    batch_size = 9
    lay1 = DyadBlock(dyad_dim,dim_in,dim_out,has_bias)
    lay2 = PyDyad(dyad_dim,dim_in,dim_out,has_bias)
    lay1.w_upper = torch.nn.Parameter(torch.ones_like(lay1.w_upper.data))
    lay1.w_lower = torch.nn.Parameter(torch.ones_like(lay1.w_lower.data))
    if has_bias:
        lay1.bias = torch.nn.Parameter(torch.ones_like(lay1.bias.data))
        b = lay1.bias.data
    else:
        lay1.bias = None
        b = None
    lay2.from_weights(lay1.w_upper.data,lay1.w_lower.data,b)
    x1 = val * torch.ones((dyad_dim*dim_in,batch_size),dtype=torch.float32)
    x1.requires_grad = True
    x2 = x1.detach().clone()
    x2.requires_grad = True
    out1 = lay1(x1)
    out2 = lay2(x2)
    assert torch.allclose(out1,out2)
    out1.sum().backward()
    out2.sum().backward()
    assert torch.allclose(lay1.w_upper.grad,lay2.w_upper.grad)
    assert torch.allclose(lay1.w_lower.grad,lay2.w_lower.grad)
    if has_bias:
        assert torch.allclose(lay1.bias.grad,lay2.bias.grad)
    assert torch.allclose(x1.grad,x2.grad)

@pytest.mark.parametrize("has_bias",[True, False])
@pytest.mark.parametrize("val",[-132.1,1,3,123.4])
def test_rand_dyad(has_bias,val):
    dim_in = 2
    dim_out = 3
    dyad_dim = 2
    batch_size = 1
    lay1 = DyadBlock(dyad_dim,dim_in,dim_out,has_bias)
    lay2 = PyDyad(dyad_dim,dim_in,dim_out,has_bias)
    lay1.w_upper = torch.nn.Parameter((val+12)*torch.ones_like(lay1.w_upper.data))
    lay1.w_lower = torch.nn.Parameter(val*torch.ones_like(lay1.w_lower.data))
    if has_bias:
        lay1.bias = torch.nn.Parameter(torch.ones_like(lay1.bias.data))
        b = lay1.bias.data
    else:
        lay1.bias = None
        b = None
    lay2.from_weights(lay1.w_upper.data,lay1.w_lower.data,b)
    x1 = torch.rand((dyad_dim*dim_in,batch_size),dtype=torch.float32)
    x1.requires_grad = True
    x2 = x1.detach().clone()
    x2.requires_grad = True
    out1 = lay1(x1)
    out2 = lay2(x2)
    print(out1,out2)
    assert torch.allclose(out1,out2)
    out1.sum().backward()
    out2.sum().backward()
    assert torch.allclose(lay1.w_upper.grad,lay2.w_upper.grad)
    assert torch.allclose(lay1.w_lower.grad,lay2.w_lower.grad)
    if has_bias:
        assert torch.allclose(lay1.bias.grad,lay2.bias.grad)
    assert torch.allclose(x1.grad,x2.grad)

@pytest.mark.parametrize("has_bias",[True, False])
@pytest.mark.parametrize("val",[-132.1,1,3,123.4])
def test_non_symmetric_out_dyad(has_bias,val):
    dim_in = 2
    dim_out = 3
    dyad_dim = 2
    batch_size = 1
    lay1 = DyadBlock(dyad_dim,dim_in,dim_out,has_bias)
    lay2 = PyDyad(dyad_dim,dim_in,dim_out,has_bias)
    lay1.w_upper = torch.nn.Parameter((val+12)*torch.ones_like(lay1.w_upper.data))
    lay1.w_lower = torch.nn.Parameter(val*torch.ones_like(lay1.w_lower.data))
    expected = torch.arange(0,6,dtype=torch.float32)
    if has_bias:
        lay1.bias = torch.nn.Parameter(torch.ones_like(lay1.bias.data))
        b = lay1.bias.data
    else:
        lay1.bias = None
        b = None
    lay2.from_weights(lay1.w_upper.data,lay1.w_lower.data,b)
    x1 = torch.rand((dyad_dim*dim_in,batch_size),dtype=torch.float32)
    x1.requires_grad = True
    x2 = x1.detach().clone()
    x2.requires_grad = True
    out1 = lay1(x1)
    out2 = lay2(x2)
    print(out1,out2)
    assert torch.allclose(out1,out2)
    (out1-expected).sum().backward()
    (out2-expected).sum().backward()
    assert torch.allclose(lay1.w_upper.grad,lay2.w_upper.grad)
    assert torch.allclose(lay1.w_lower.grad,lay2.w_lower.grad)
    if has_bias:
        assert torch.allclose(lay1.bias.grad,lay2.bias.grad)
    assert torch.allclose(x1.grad,x2.grad)