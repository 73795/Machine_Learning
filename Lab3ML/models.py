import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = nn.Linear(49, 64)
        self.pos = nn.Parameter(torch.zeros(1, 16, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.head = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.reshape(batch_size, 1, 4, 7, 4, 7)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(batch_size, 16, 49)

        x = self.projection(x) + self.pos
        x = self.encoder(x)
        x = x.mean(dim=1)

        return self.head(x)
