import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from data import processed_data
from utils import device, validate
from wide_model import NetWidth

torch.manual_seed(123)
cifar2, cifar2_val = processed_data()
device = device()
model = NetWidth().to(device=device)


def training_loop_l2reg(n_epochs, optimizer, model, loss_fn, device):
    train_loader = torch.utils.data.DataLoader(
        cifar2, batch_size=64, shuffle=True)
    for epoch in range(n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training Loss {}'.format(
                datetime.datetime.now(), epoch, loss_train / len(train_loader)))



optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()


training_loop_l2reg(
    n_epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    device=device
)
