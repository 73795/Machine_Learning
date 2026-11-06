import torch
import torch.nn as nn
import time

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1)
)

x = torch.randn(1000, 2)

print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Contiguous:", x.is_contiguous())

start = time.perf_counter()

for i in range(1000):
    model(x)

end = time.perf_counter()

print("Eager time:", end - start)

try:

    compiled_model = torch.compile(model)

    for i in range(10):
        compiled_model(x)

    start = time.perf_counter()

    for i in range(1000):
        compiled_model(x)

    end = time.perf_counter()

    print("Compiled time:", end - start)

except:

    traced_model = torch.jit.trace(model, x)

    start = time.perf_counter()

    for i in range(1000):
        traced_model(x)

    end = time.perf_counter()

    print("Traced time:", end - start)

a = torch.randn(1000, 1000)

b = a.t()

print("Is b contiguous:", b.is_contiguous())

start = time.perf_counter()

for i in range(100):
    c = a * 2

end = time.perf_counter()

print("Contiguous tensor time:", end - start)

start = time.perf_counter()

for i in range(100):
    c = b * 2

end = time.perf_counter()

print("Non-contiguous tensor time:", end - start)