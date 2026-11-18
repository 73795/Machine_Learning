import torch

from torch.profiler import profile

from models import CNN

model = CNN()

x = torch.randn(
    128,
    1,
    28,
    28
)

with profile() as prof:

    output = model(x)

print(prof.key_averages().table(sort_by="cpu_time_total"))
