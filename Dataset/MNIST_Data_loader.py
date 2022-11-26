import sys
sys.path.append('../')
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np

# Define the train and test sets
dataset = MNIST(root="../Dataset/", download=True, 
                           transform=transforms.Compose([
                               transforms.Resize(28),
                               transforms.CenterCrop(28),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                           ]))

batch_size = 64
num_workers = 4
# The loaders perform the actual work
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                         shuffle=True, pin_memory=True)