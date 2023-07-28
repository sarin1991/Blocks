import torch
import numpy as np

class Dyad(torch.nn.Module):
    def __init__(self,dyad_dim,dim_in,dim_out,has_bias=False):
        super().__init__()
        self.dyad_dim = dyad_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.has_bias = has_bias
        k = 1.0/float(np.sqrt(dyad_dim*dim_in))
        self.w_upper = torch.nn.Parameter(torch.empty((dyad_dim,dim_out,dim_in),dtype=torch.float32))
        torch.nn.init.uniform_(self.w_upper,-k,k)
        self.w_lower = torch.nn.Parameter(torch.empty((dyad_dim,dim_out,dim_in),dtype=torch.float32))
        torch.nn.init.uniform_(self.w_lower,-k,k)
        if self.has_bias:
            self.bias = torch.nn.Parameter(torch.empty((dyad_dim*dim_out),dtype=torch.float32))
            torch.nn.init.uniform_(self.bias,-k,k)
        else:
            self.bias=None
    def from_weights(self,w_upper,w_lower,bias):
        self.w_upper = torch.nn.Parameter(w_upper)
        self.w_lower = torch.nn.Parameter(w_lower)
        if self.has_bias:
            self.bias = torch.nn.Parameter(bias)
    def forward(self,x):
        print("self.w_lower ",self.w_lower)
        print("self.w_upper ",self.w_upper)
        x1 = self.w_lower.bmm(x.reshape(self.dyad_dim,self.dim_in,-1)).reshape(self.dyad_dim*self.dim_out,-1)
        print("x1 ",x1)
        x2 = self.w_upper.bmm(x.reshape(self.dim_in,self.dyad_dim,-1).transpose(0,1)).transpose(0,1).reshape(self.dyad_dim*self.dim_out,-1)
        print("x1 ",x2)
        out = x1 + x2
        if self.has_bias:
            out = out + self.bias.reshape(-1,1)
        return out