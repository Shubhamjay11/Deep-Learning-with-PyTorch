import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from data import processed_data
from utils import device, training_loop


torch.manual_seed(123)
cifar2, cifar2_val = processed_data()


class NetBatchNorm(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


def validate(model, device):
    model.eval()
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=False)
    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

    for name, loader in [("train", train_loader), ("valid", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
            
        print("Accuracy {}: {:.2f}".format(name, correct / total))


device = device()

model_batchnorm = NetBatchNorm(n_chans1=32).to(device=device)

optimizer = optim.SGD(model_batchnorm.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model=model_batchnorm,
    loss_fn=loss_fn,
    device=device
)


validate(model_batchnorm, device)

# This model gives the following result:
# Accuracy train: 1.00
# Accuracy valid: 0.88