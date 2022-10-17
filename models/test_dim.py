import torch

a = torch.ones(1, 2, 3)

b = torch.rand(4, 3)

print(b)

c = torch.unsqueeze(b, 1)

d = c.repeat(1, a.shape[1], 1)

print(a.shape, b.shape, c.shape, d.shape)

e = torch.matmul(a.permute(1,0,2), d.permute(1,2,0))

# expect 2, 6

e = torch.squeeze(e)

print(e.shape)

print(e)

