import numpy as np
import torch
import blocks_extension
PyReLUBlock = blocks_extension.blocks.PyReLUBlock

class ReLUOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,layer):
        out = torch.clone(x,memory_format=torch.contiguous_format).detach()
        ctx.layer = layer
        layer.forward(out.numpy())
        ctx.save_for_backward(out)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0]
        in_grad = torch.clone(grad_output,memory_format=torch.contiguous_format).detach()
        ctx.layer.backward(in_grad.numpy(),
                           out.numpy())
        return in_grad, None
    
relu_op = ReLUOp.apply

class ReLUBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = PyReLUBlock()
    def forward(self,x):
        out = relu_op(x,self.layer)
        return out