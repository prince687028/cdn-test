import torch
from torch import nn

a = torch.zeros(1000, 2, 256)

a = a.permute(2, 1, 0)

print(a.shape)

# pool = nn.MaxPool1d(a.shape[-1])
pool = nn.AdaptiveMaxPool1d(1)

b = pool(a).permute(2, 1, 0)

print(b.shape)