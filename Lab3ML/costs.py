import statistics
import time

import torch
from torchinfo import summary
from thop import profile


def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )


def measure_latency(model, batch_size, device, runs):
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, 1, 28, 28, device=device)

    with torch.inference_mode():
        for _ in range(10):
            model(x)

        times = []

        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            times.append((time.perf_counter() - start) * 1000)

    return statistics.median(times)


def analyze_model(model, device):
    model = model.to(device)
    params = count_parameters(model)

    summary(
        model,
        input_size=(1, 1, 28, 28),
        device=str(device)
    )

    x = torch.randn(1, 1, 28, 28, device=device)

    model.eval()
    with torch.inference_mode():
        macs, _ = profile(model, inputs=(x,), verbose=False)

    latency_1 = measure_latency(model, 1, device, 100)
    latency_128 = measure_latency(model, 128, device, 20)

    return {
        "parameters": params,
        "macs": macs,
        "thop_flops_approx": 2 * macs,
        "latency_batch1_ms": latency_1,
        "latency_batch128_ms": latency_128
    }


def manual_flops():
    return {
        "MLP": 2 * 784 * 256,
        "CNN": 2 * 28 * 28 * 3 * 3 * 1 * 16,
        "Attention": 2 * 16 * 49 * 64
    }
