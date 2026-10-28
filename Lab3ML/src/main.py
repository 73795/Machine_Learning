
import csv
import os
import torch

from data import get_data
from models import MLP, CNN, AttentionNet
from costs import analyze_model, manual_flops
from train import train_model, evaluate
from plots import create_plots


BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

train_loader, val_loader, test_loader = get_data(BATCH_SIZE)

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionNet()
}

flops_estimates = manual_flops()
results = {}

for name, model in models.items():
    print(f"\n--- {name}: model cost ---")

    stats = analyze_model(model, device)
    stats["manual_flops"] = flops_estimates[name]

    print("Parameters:", stats["parameters"])
    print("MACs:", round(stats["macs"]))
    print("Approximate FLOPs:", round(stats["thop_flops_approx"]))
    print("Manual FLOPs estimate:", stats["manual_flops"])
    print("Latency, batch 1:", round(stats["latency_batch1_ms"], 3), "ms")
    print("Latency, batch 128:", round(stats["latency_batch128_ms"], 3), "ms")

    print(f"\n--- {name}: training ---")

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    test_accuracy = evaluate(model, test_loader, device)

    stats["test_accuracy"] = test_accuracy
    stats["final_val_accuracy"] = history[-1]["val_accuracy"]
    results[name] = stats

    print(f"{name} test accuracy: {test_accuracy:.4f}")


os.makedirs("results", exist_ok=True)

with open(
    "results/model_comparison.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    columns = [
        "model",
        "parameters",
        "macs",
        "thop_flops_approx",
        "manual_flops",
        "latency_batch1_ms",
        "latency_batch128_ms",
        "final_val_accuracy",
        "test_accuracy"
    ]

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for name, stats in results.items():
        row = {"model": name}
        row.update(stats)
        writer.writerow(row)

create_plots(results)

print("\nAll experiments completed!")
print("Check the results folder.")
