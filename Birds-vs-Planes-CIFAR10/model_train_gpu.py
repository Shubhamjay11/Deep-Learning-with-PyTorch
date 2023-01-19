import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from birds_vs_planes_cnn import processed_data
import datetime
torch.manual_seed(123)

# initializing the data
cifar2, cifar2_val = processed_data()


# Using functional counterparts of pooling and activation
# since they do not have any parameters during training


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


# device
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

# training loop


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(n_epochs):
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
            print("{} Epoch {}, Training loss {}".format(
                datetime.datetime.now(), epoch, loss_train / len(train_loader)))


train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

# moving the model to GPU (if available)
model = Net().to(device=device)

# number of parameters
numel_list = [p.numel() for p in model.parameters()]
print(
    f"Total number of parameters (in functional counterparts model): {sum(numel_list)}")
print(
    f"Parameters in each layer (in functional counterparts model): {numel_list}")

optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# starting the training
training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader
)

# checking the accuracy

train_loader = torch.utils.data.DataLoader(
    cifar2, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(
    cifar2_val, batch_size=64, shuffle=False)


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
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

# validating the train and test set


validate(model, train_loader, val_loader)

# saving the model
data_path = '/content/Deep-Learning-with-PyTorch/Birds-vs-Planes-CIFAR10'
torch.save(model.state_dict(), data_path + 'birds_vs_airplanes.pt')

# loading the model
loaded_model = Net()
loaded_model.load_state_dict(torch.load(data_path + 'birds_vs_airplanes.pt'))
