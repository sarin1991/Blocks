import numpy as np
import torch
import blocks_extension
PyRMSNormBlock = blocks_extension.blocks.PyRMSNormBlock

class RMSNormOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,params,layer):
        bs = x.size(dim=1)
        out = torch.clone(x,memory_format=torch.contiguous_format).detach()
        forward_context = torch.zeros(bs,dtype=torch.float32)
        ctx.layer = layer
        layer.forward(params.numpy(),out.numpy(),forward_context.numpy())
        ctx.save_for_backward(out,params,forward_context)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        out,params,forward_context = ctx.saved_tensors
        in_grad = torch.clone(grad_output,memory_format=torch.contiguous_format).detach()
        param_grad = torch.zeros_like(params)
        ctx.layer.backward(param_grad.numpy(),in_grad.numpy(),
                           out.numpy(),forward_context.numpy(),params.numpy())
        return in_grad, param_grad, None
    
rmsnorm_op = RMSNormOp.apply

class RMSNORMBlock(torch.nn.Module):
    def __init__(self,dim: int, chunk_size: int):
        super().__init__()
        self.layer = PyRMSNormBlock(dim,chunk_size)
        self.params = torch.nn.Parameter(torch.ones(dim,dtype=torch.float32))
    def forward(self,x):
        out = rmsnorm_op(x,self.params,self.layer)
        return out