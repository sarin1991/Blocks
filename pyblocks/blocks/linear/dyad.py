import numpy as np
import torch
import blocks_extension
PyDyadBlock = blocks_extension.blocks.PyDyadBlock

def get_dyad_zero_params(dy,do,di,has_bias):
    if has_bias:
        bias_params = torch.zeros(dy*do,dtype=torch.float32)
    else:
        bias_params = None
    w_upper = torch.zeros((dy,do,di),dtype=torch.float32)
    w_lower = torch.zeros((dy,do,di),dtype=torch.float32)
    return (w_upper,w_lower,bias_params)

class DyadOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w_upper,w_lower,bias,has_bias,layer):
        x = x.contiguous()
        dy,do,di = w_upper.shape
        ctx.layer = layer
        ctx.has_bias = has_bias
        out = torch.zeros((dy*do,x.size(dim=1)),device=x.device)
        if has_bias:
            bias_numpy = bias.numpy()
        else:
            bias_numpy = None
        params = (w_upper.numpy(),w_lower.numpy(),bias_numpy)
        layer.forward(params,x.numpy(),out.numpy())
        if ctx.has_bias:
            ctx.save_for_backward(x,out,w_upper,w_lower,bias)
        else:
            ctx.save_for_backward(x,out,w_upper,w_lower)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x, out, w_upper, w_lower, bias = ctx.saved_tensors
            bias_numpy = bias.numpy()
        else:
            x, out, w_upper, w_lower = ctx.saved_tensors
            bias_numpy = None
        dy,do,di = w_upper.shape
        input_gradients = torch.zeros((dy*di,x.size(dim=1)),device=x.device)
        output_gradients = grad_output.contiguous()
        (w_upper_grad,w_lower_grad,bias_grad) = get_dyad_zero_params(dy,do,di,ctx.has_bias)
        if ctx.has_bias:
            parameter_gradients = (w_upper_grad.numpy(),w_lower_grad.numpy(),bias_grad.numpy())
        else:
            parameter_gradients = (w_upper_grad.numpy(),w_lower_grad.numpy(),None)
        parameters = (w_upper.numpy(),w_lower.numpy(),bias_numpy)
        ctx.layer.backward(parameter_gradients, input_gradients.numpy(),
                           output_gradients.numpy(), out.numpy(), x.numpy(),
                           parameters)
        return input_gradients, w_upper_grad, w_lower_grad, bias_grad, None, None
    
dyad_op = DyadOp.apply

class DyadBlock(torch.nn.Module):
    def __init__(self,dyad_dim,dim_in,dim_out,has_bias=False):
        super().__init__()
        self.dyad_dim = dyad_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.has_bias = has_bias
        k = 1.0/float(np.sqrt(dyad_dim*dim_in))
        self.layer = PyDyadBlock(dyad_dim,dim_in,dim_out,has_bias)
        self.w_upper = torch.nn.Parameter(torch.empty((dyad_dim,dim_out,dim_in),dtype=torch.float32))
        torch.nn.init.uniform_(self.w_upper,-k,k)
        self.w_lower = torch.nn.Parameter(torch.empty((dyad_dim,dim_out,dim_in),dtype=torch.float32))
        torch.nn.init.uniform_(self.w_lower,-k,k)
        if self.has_bias:
            self.bias = torch.nn.Parameter(torch.empty((dyad_dim*dim_out),dtype=torch.float32))
            torch.nn.init.uniform_(self.bias,-k,k)
        else:
            self.bias=None
    def forward(self,x):
        out = dyad_op(x,self.w_upper,self.w_lower,self.bias,
                      self.has_bias,self.layer)
        return out
    