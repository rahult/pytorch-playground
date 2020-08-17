# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimiser
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer import Trainer
import matplotlib.pyplot as plt

# hyper parameters
input_size = 784  # 28x28
hidden_size = 100
num_classes = 10  # 0-9 numbers
num_epochs = 2
batch_size = 100
learning_rate = 0.001


class LitMNIST(LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitMNIST, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # forward
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12
        )

        return train_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        tensorboard_logs = {"val_loss": loss}
        return {"val_loss": loss, "log": tensorboard_logs}

    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor()
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=12
        )

        return val_loader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}


if __name__ == "__main__":
    trainer = Trainer(max_epochs=num_epochs, fast_dev_run=False)
    model = LitMNIST(input_size, hidden_size, num_classes)
    trainer.fit(model)
