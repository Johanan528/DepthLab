import torch
import torch.nn as nn

class My_proj(nn.Module):
    def __init__(self, input=1024, dtype=torch.float32):
        super(My_proj, self).__init__()
        self.mapping_layer = nn.Linear(input, 1024)
        self.dtype = dtype
    
    def forward(self, x):
        x = self.mapping_layer(x)
        return x