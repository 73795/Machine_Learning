import os

import mlflow
import torch
import torch.nn as nn


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (
                predictions == labels
            ).sum().item()
            total += labels.size(0)
    return correct / total

def train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        learning_rate,
        checkpoint_path,
        start_epoch=0
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = (
            total_loss / len(train_loader)
        )
        val_accuracy = evaluate(
            model,
            val_loader,
            device
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": average_loss,
                "val_accuracy": val_accuracy
            }
        )
        mlflow.log_metric(
            "loss",
            average_loss,
            step=epoch
        )
        mlflow.log_metric(
            "val_accuracy",
            val_accuracy,
            step=epoch
        )
        os.makedirs(
            "checkpoints",
            exist_ok=True
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state":
                    model.state_dict(),

                "optimizer_state":
                    optimizer.state_dict()
            },
            checkpoint_path
        )
        print(f"Epoch {epoch+1}/{epochs} | " f"Loss: {average_loss:.4f} | "f"Val Acc: {val_accuracy:.4f}")
    return model, history