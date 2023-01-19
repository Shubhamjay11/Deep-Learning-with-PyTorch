import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
torch.manual_seed(123)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
data_path = '/'

    
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ])
)

cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ])
)

# extracting birds and airplanes data
def processed_data():
    label_map={0: 0, 2: 1}
    class_names=['airplane', 'bird']
    cifar2=[(img, label_map[label]) for img, label in cifar10
                if label in [0, 2]]
    cifar2_val=[(img, label_map[label])
                for img, label in cifar10_val if label in [0, 2]]

    return cifar2, cifar2_val
# defining the model
model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Linear(8 * 8 * 8, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
)

numel_list = [p.numel() for p in model.parameters()]

print(f"The total parameters are: {sum(numel_list)}")
print(f"Parameters in each layer: {numel_list}")