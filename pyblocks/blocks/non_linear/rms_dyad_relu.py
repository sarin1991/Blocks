import numpy as np
import torch
import blocks_extension
from ..linear.dyad import get_dyad_zero_params
PyRMSDyadReLUBlock = blocks_extension.blocks.PyRMSDyadReLUBlock

def get_rms_dyad_zero_params(dy,do,di,has_bias):
    rms_params = torch.zeros(dy*di,dtype=torch.float32)
    dyad_params = get_dyad_zero_params(dy,do,di,has_bias)
    return (rms_params,dyad_params)

class RMSDyadReLUOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,rms_params,w_upper,w_lower,bias,has_bias,layer):
        x = x.contiguous()
        bs = x.size(dim=1)
        dy,do,di = w_upper.shape
        ctx.layer = layer
        ctx.has_bias = has_bias
        out = torch.zeros((dy*do,bs),device=x.device)
        if has_bias:
            bias_numpy = bias.numpy()
        else:
            bias_numpy = None
        dyad_params = (w_upper.numpy(),w_lower.numpy(),bias_numpy)
        params = (rms_params.numpy(),dyad_params)
        fc = torch.zeros(bs,dtype=torch.float32)
        layer.forward(params,x.numpy(),out.numpy(),fc.numpy())
        if ctx.has_bias:
            ctx.save_for_backward(x,out,rms_params,w_upper,w_lower,bias,fc)
        else:
            ctx.save_for_backward(x,out,rms_params,w_upper,w_lower,fc)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x, out, rms_params, w_upper, w_lower, bias, fc = ctx.saved_tensors
            bias_numpy = bias.numpy()
        else:
            x, out, rms_params, w_upper, w_lower, fc = ctx.saved_tensors
            bias_numpy = None
        bs = x.size(dim=1)
        dy,do,di = w_upper.shape
        (rms_param_grad,(w_upper_grad,w_lower_grad,bias_grad)) = get_rms_dyad_zero_params(dy,do,di,ctx.has_bias)
        if ctx.has_bias:
            parameter_gradients = (rms_param_grad.numpy(),(w_upper_grad.numpy(),w_lower_grad.numpy(),bias_grad.numpy()))
        else:
            parameter_gradients = (rms_param_grad.numpy(),(w_upper_grad.numpy(),w_lower_grad.numpy(),None))
        input_gradients = torch.zeros((dy*di,bs),device=x.device)
        output_gradients = grad_output.contiguous()
        dyad_params = (w_upper.numpy(),w_lower.numpy(),bias_numpy)
        params = (rms_params.numpy(),dyad_params)
        ctx.layer.backward(parameter_gradients, input_gradients.numpy(),
                           output_gradients.numpy(), out.numpy(), x.numpy(),
                           params,fc.numpy())
        return input_gradients, rms_param_grad, w_upper_grad, w_lower_grad, bias_grad, None, None
    
rms_dyad_relu_op = RMSDyadReLUOp.apply

class RMSDyadReLUBlock(torch.nn.Module):
    def __init__(self,dyad_dim,dim_in,dim_out,rms_norm_chunk_size,has_bias=False):
        super().__init__()
        self.dyad_dim = dyad_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.has_bias = has_bias
        k = 1.0/float(np.sqrt(dyad_dim*dim_in))
        self.layer = PyRMSDyadReLUBlock(dyad_dim,dim_in,dim_out,has_bias,rms_norm_chunk_size)
        self.rms_params = torch.nn.Parameter(torch.ones(dyad_dim*dim_in,dtype=torch.float32))
        self.w_upper = torch.nn.Parameter(torch.empty((dyad_dim,dim_out,dim_in),dtype=torch.float32))
        torch.nn.init.uniform_(self.w_upper,-k,k)
        self.w_lower = torch.nn.Parameter(torch.empty((dyad_dim,dim_out,dim_in),dtype=torch.float32))
        torch.nn.init.uniform_(self.w_lower,-k,k)
        if self.has_bias:
            self.bias = torch.nn.Parameter(torch.empty((dyad_dim*dim_out),dtype=torch.float32))
            torch.nn.init.uniform_(self.bias,-k,k)
        else:
            self.bias=None
    def from_weights(self,rms_params,w_upper,w_lower,bias):
        self.rms_params = torch.nn.Parameter(rms_params)
        self.w_upper = torch.nn.Parameter(w_upper)
        self.w_lower = torch.nn.Parameter(w_lower)
        if self.has_bias:
            self.bias = torch.nn.Parameter(bias)
    def forward(self,x):
        out = rms_dyad_relu_op(x,self.rms_params,self.w_upper,self.w_lower,self.bias,
                      self.has_bias,self.layer)
        return out
    