from data import get_data
from models import CNN
from train import train_model, evaluate

import torch
import time

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

batch_sizes = [32, 64, 128, 256, 512]


for batch_size in batch_sizes:

    print("\n===================")
    print("Batch Size:", batch_size)
    train_loader, val_loader, test_loader = get_data(
        batch_size=batch_size,
        num_workers=0
    )

    model = CNN()
    start = time.time()
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=5,
        learning_rate=0.001,
        checkpoint_path=f"batch_{batch_size}.pt"
    )

    elapsed = time.time() - start
    accuracy = evaluate(
        model,
        test_loader,
        device
    )

    throughput = 60000 / elapsed
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time: {elapsed:.2f}s")

    print(f"Throughput: {throughput:.2f} samples/s")