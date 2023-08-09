from pyblocks.blocks.non_linear.rms_dyad_relu import RMSDyadReLUBlock
from pyblocks.native.blocks.non_linear.rms_dyad_relu import RMSDyadReLU
import torch
import pytest

@pytest.mark.parametrize("has_bias",[True, False])
@pytest.mark.parametrize("val",[-1,1,3,123.4])
def test_simple_dyad(has_bias,val):
    dim_in = 4
    dim_out = 4
    dyad_dim = 4
    batch_size = 1
    rms_norm_chunk_size = 16
    lay1 = RMSDyadReLUBlock(dyad_dim,dim_in,dim_out,rms_norm_chunk_size,has_bias)
    lay2 = RMSDyadReLU(dyad_dim,dim_in,dim_out,has_bias)
    rms_params = torch.nn.Parameter(torch.ones(dyad_dim*dim_in,dtype=torch.float32))
    w_upper = torch.nn.Parameter(torch.ones((dyad_dim,dim_out,dim_in),dtype=torch.float32))
    w_lower = torch.nn.Parameter(torch.ones((dyad_dim,dim_out,dim_in),dtype=torch.float32))
    if has_bias:
        bias = torch.nn.Parameter(torch.ones(dyad_dim*dim_out,dtype=torch.float32))
    else:
        bias = None
    lay1.from_weights(rms_params,w_upper,w_lower,bias)
    lay2.from_weights(rms_params,w_upper,w_lower,bias)
    x1 = val * torch.ones((dyad_dim*dim_in,batch_size),dtype=torch.float32)
    x1.requires_grad = True
    x2 = x1.detach().clone()
    x2.requires_grad = True
    out1 = lay1(x1)
    out2 = lay2(x2)
    assert torch.allclose(out1,out2)
    out1.sum().backward()
    out2.sum().backward()
    print(lay1.rms_params.grad,lay2.model[0].params.grad)
    assert torch.allclose(lay1.rms_params.grad,lay2.model[0].params.grad)
    assert torch.allclose(lay1.w_upper.grad,lay2.model[1].w_upper.grad)
    assert torch.allclose(lay1.w_lower.grad,lay2.model[1].w_lower.grad)
    if has_bias:
        assert torch.allclose(lay1.bias.grad,lay2.model[1].bias.grad)
    assert torch.allclose(x1.grad,x2.grad,atol=10**-6,rtol=10**-6)

@pytest.mark.parametrize("has_bias",[True, False])
@pytest.mark.parametrize("val",[-132.1,1,3,123.4])
def test_rand_dyad(has_bias,val):
    dim_in = 2
    dim_out = 3
    dyad_dim = 2
    batch_size = 1
    rms_norm_chunk_size = 16
    lay1 = RMSDyadReLUBlock(dyad_dim,dim_in,dim_out,rms_norm_chunk_size,has_bias)
    lay2 = RMSDyadReLU(dyad_dim,dim_in,dim_out,has_bias)
    rms_params = torch.nn.Parameter(torch.ones(dyad_dim*dim_in,dtype=torch.float32))
    w_upper = torch.nn.Parameter((val+12)*torch.ones((dyad_dim,dim_out,dim_in),dtype=torch.float32))
    w_lower = torch.nn.Parameter(val*torch.ones((dyad_dim,dim_out,dim_in),dtype=torch.float32))
    if has_bias:
        bias = torch.nn.Parameter(torch.ones(dyad_dim*dim_out,dtype=torch.float32))
    else:
        bias = None
    lay1.from_weights(rms_params,w_upper,w_lower,bias)
    lay2.from_weights(rms_params,w_upper,w_lower,bias)
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
    assert torch.allclose(lay1.rms_params.grad,lay2.model[0].params.grad)
    assert torch.allclose(lay1.w_upper.grad,lay2.model[1].w_upper.grad)
    assert torch.allclose(lay1.w_lower.grad,lay2.model[1].w_lower.grad)
    if has_bias:
        assert torch.allclose(lay1.bias.grad,lay2.model[1].bias.grad)
    print(x1.grad,x2.grad)
    assert torch.allclose(x1.grad,x2.grad,atol=10**-3,rtol=10**-3)