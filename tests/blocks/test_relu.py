from pyblocks.blocks.activations.relu import ReLUBlock
import torch
import pytest

@pytest.mark.parametrize("val",[-1,1,3,123.4])
def test_relu_constant(val):
    dim = 21
    batch_size = 127
    lay1 = ReLUBlock()
    lay2 = torch.nn.ReLU()
    x1 = val * torch.ones((dim,batch_size),dtype=torch.float32)
    x1.requires_grad = True
    x2 = x1.detach().clone()
    x2.requires_grad = True
    out1 = lay1(x1)
    out2 = lay2(x2)
    assert torch.allclose(out1,out2)
    out1.sum().backward()
    out2.sum().backward()
    assert torch.allclose(x1.grad,x2.grad)