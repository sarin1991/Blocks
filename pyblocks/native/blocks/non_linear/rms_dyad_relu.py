import torch
import numpy as np
from ..linear.dyad import Dyad
from ..norm.rms_norm import RMSNorm

class RMSDyadReLU(torch.nn.Module):
    def __init__(self,dyad_dim,dim_in,dim_out,has_bias=False):
        super().__init__()
        self.dyad_dim = dyad_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.has_bias = has_bias
        dim = self.dyad_dim*self.dim_in
        self.model = torch.nn.Sequential(
            RMSNorm(dim),
            Dyad(dyad_dim,dim_in,dim_out,has_bias),
            torch.nn.ReLU(),
        )
    def from_weights(self,rms_params,w_upper,w_lower,bias):
        self.model[0].from_weights(rms_params)
        self.model[1].from_weights(w_upper,w_lower,bias)
    def forward(self,x):
        return self.model(x)