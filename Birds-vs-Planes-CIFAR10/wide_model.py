import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import processed_data
from utils import training_loop, validate, device

torch.manual_seed(123)

device = device()

class NetWidth(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.nchans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2,
                               kernel_size=3, padding=1)

        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * self.nchans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


model = NetWidth().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

n_parameters = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters are: {n_parameters}")

print("Training started...")
training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    device=device
)
print("Training finished.")

validate(model, device=device)


# This script gives the following result:
# Accuracy train: 0.97
# Accuracy valid: 0.90