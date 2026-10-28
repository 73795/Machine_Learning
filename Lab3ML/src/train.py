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

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_model(
    model, train_loader, val_loader, device,
    epochs=5, learning_rate=1e-3
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "loss": average_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.4f}"
        )

    return model, history
