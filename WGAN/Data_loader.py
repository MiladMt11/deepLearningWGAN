import sys
sys.path.append('../')
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import torchvision.transforms as transforms

# Define the train and test sets
dataset_cifar = CIFAR10(root="../Dataset/", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                           ]))
dataset_mnist = MNIST(root="../Dataset/", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
dataset_fashionmnist = FashionMNIST(root="../Dataset/", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))

batch_size = 64
num_workers = 2
# The loaders perform the actual work
train_loader_cifar = torch.utils.data.DataLoader(dataset_cifar, batch_size=batch_size,
                                         shuffle=True, num_workers= num_workers, pin_memory=True)
train_loader_mnist = torch.utils.data.DataLoader(dataset_mnist, batch_size=batch_size,
                                         shuffle=True, num_workers= num_workers, pin_memory=True)
train_loader_fashionmnist = torch.utils.data.DataLoader(dataset_fashionmnist, batch_size=batch_size,
                                         shuffle=True, num_workers= num_workers, pin_memory=True)
