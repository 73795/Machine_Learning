from engine import Value
import torch


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

y = ((a * b + c) ** 2).relu().sum()

y.backward()

print("Custom Engine")
print("a grad =", a.grad)
print("b grad =", b.grad)
print("c grad =", c.grad)

print()

a_t = torch.tensor(2.0, requires_grad=True)
b_t = torch.tensor(-3.0, requires_grad=True)
c_t = torch.tensor(10.0, requires_grad=True)

y_t = ((a_t * b_t + c_t) ** 2)
y_t = torch.relu(y_t).sum()

y_t.backward()

print("PyTorch")
print("a grad =", a_t.grad.item())
print("b grad =", b_t.grad.item())
print("c grad =", c_t.grad.item())