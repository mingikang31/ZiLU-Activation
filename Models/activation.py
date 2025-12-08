import torch 
import torch.nn as nn 
import torch.nn.functional as F

"""
F.relu()
F.gelu()
F.silu()
"""

class GELU_a(nn.Module):
    def __init__(self, a, inplace=False, max_val=1000):
        super(GELU_a, self).__init__()

        self.a = a
        self.max_val = max_val
        self.kAlpha = 0.70710678118654752440
        self.relu = nn.ReLU(inplace=inplace) 

    def forward(self, x):
        if self.a >= self.max_val:
            return self.relu(x) 
        else: 
            return x * 0.5 * (1 + torch.erf(self.a * x * self.kAlpha))

class SiLU_a(nn.Module):
    def __init__(self, a, inplace=False, max_val=1000):
        super(SiLU_a, self).__init__()

        self.a = a
        self.max_val = max_val
        self.relu = nn.ReLU(inplace=inplace)
        
    def forward(self, x):
        if self.a >= self.max_val:
            return self.relu(x)
        else:
            return x * torch.sigmoid(self.a * x)

class ZiLU(nn.Module):
    def __init__(self, s, inplace=False, max_val=1000):
        super(ZiLU, self).__init__()

        self.s = s
        self.max_val = max_val
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        if self.s >= self.max_val:
            return self.relu(x)
        else:
            return x * (2 * (1/4 + 1/(2 * torch.pi) * torch.arctan(self.s * x)))
        

"""Zai's old code """
def gelu_a_zai(x, a=1):
    if a >= 1000:
        return F.relu(x)
    kAlpha = 0.70710678118654752440
    return x * 0.5 * (1 + torch.erf(a * x * kAlpha))

def silu_a_zai(x, a=1):
    if a >= 1000:
        return F.relu(x)
    return x * torch.sigmoid(a * x)

def zilu_zai(x, s=1):
    if s >= 1000:
        return F.relu(x)
    return x * (2 * (1/4 + 1/(2 * torch.pi) * torch.arctan(s * x)))


