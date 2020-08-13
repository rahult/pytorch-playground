# Fetch data
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimiser
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# Transforms
transform = transforms.Compose([transforms.ToTensor()])

# DATASET
train_dataset = torchvision.datasets.???(
    root="./data", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.???(
    root="./data", train=False, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

    def forward(self, x):
        out = x
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (samples, labels) in enumerate(train_loader):

        # forward
        outputs = model(samples)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss:.8f}"
            )

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for samples, labels in test_loader:

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"accuracy = {acc}")
