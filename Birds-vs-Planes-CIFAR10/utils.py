# This script contains all the functions need to experiment with different neural network
# architechtures

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from data import processed_data

torch.manual_seed(123)
cifar2, cifar2_val = processed_data()


def device():
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    return device


def training_loop(n_epochs, optimizer, model, loss_fn, device):
    train_loader = torch.utils.data.DataLoader(
        cifar2, batch_size=64, shuffle=True)

    for epoch in range(n_epochs+1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print("{} Epoch: {}, Training Loss: {}".format(
                datetime.datetime.now(), epoch, loss_train / len(train_loader)))


# validate model

def validate(model, device):
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
