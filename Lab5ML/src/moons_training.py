import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons
import time
import statistics

times = []

for i in range(5):

    X, y = make_moons(
        n_samples=200,
        noise=0.1,
        random_state=42
    )

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    loss_fn = nn.BCELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1
    )

    start = time.perf_counter()

    for epoch in range(300):

        optimizer.zero_grad()

        output = model(X)

        loss = loss_fn(output, y)

        loss.backward()

        optimizer.step()

    end = time.perf_counter()

    times.append(end - start)

print("Times:")
print(times)

print("Median:")
print(statistics.median(times))