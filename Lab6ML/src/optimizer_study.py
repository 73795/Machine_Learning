from data import get_data
from models import CNN
from train import train_model, evaluate

import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
train_loader, val_loader, test_loader = get_data(
    batch_size=128,
    num_workers=0
)
optimizers = [
    "Adam",
    "SGD"
]
schedulers = [
    "StepLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau"
]


for optimizer in optimizers:
    for scheduler in schedulers:
        print("\n===================")
        print(optimizer, "+", scheduler)
        model = CNN()
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=10,
            learning_rate=0.001,
            checkpoint_path=f"{optimizer}_{scheduler}.pt"
        )
        accuracy = evaluate(
            model,
            test_loader,
            device
        )
        print(
            "Accuracy:",
            round(accuracy, 4)
        )

