"""
Day 1
We can learn pytorch with very very simple example!
https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py


import torch

x = torch.randn(1, 10)
prev_h = torch.randn(1, 20)
W_h = torch.randn(20, 20)
W_x = torch.randn(20, 10)

i2h = torch.mm(W_x, x.t())
h2h = torch.mm(W_h, prev_h.t())
next_h = i2h + h2h
next_h = next_h.tanh()

loss = next_h.sum()
loss.backward() #compute gradient



"""

from __future__ import print_function
import torch

x = torch.empty(5,3)  # uninitialized
print(x)

x = torch.randn(5,3)
print(x)  # initialized

x = torch.zeros(5,3, dtype=torch.long)
print(x)  # zero and long type

x = torch.tensor([5.5, 3])
print(x) # construct directly from data

x = x.new_ones(5,3,dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

y = torch.randn(5,3)
print(x+y)


print(torch.add(x, y))

# adds x to y
y.add_(x)  # Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
print(y)

print(x[:, 1])

x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions  --> reshape??
print(x.size(), y.size(), z.size())


x = torch.randn(1)
print(x)
print(x.item())


#  https://pytorch.org/docs/stable/torch.html



"""
NumPy Bridge
Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

The Torch Tensor and NumPy array will share their underlying memory locations, and changing one will change the other.
"""
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)  # ---> here, it also changed itself followed by a


if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.dobule))

