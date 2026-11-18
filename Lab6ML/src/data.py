
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data(batch_size=128, num_workers=2):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    train_data, validation_data = random_split(
        train_data,
        [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, validation_loader, test_loader
