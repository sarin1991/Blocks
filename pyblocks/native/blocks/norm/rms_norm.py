import torch

eps = torch.tensor(0.000001,dtype=torch.float32)

class RMSNorm(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.params = torch.nn.Parameter(torch.ones(dim,dtype=torch.float32))
    def from_weights(self,params):
        self.params = torch.nn.Parameter(params)
    def forward(self,x):
        ms = torch.max((x**2).mean(axis=0),eps)
        rms = 1/torch.sqrt(ms)
        x_norm = x*rms.reshape(1,-1)*self.params.reshape(-1,1)
        return x_norm