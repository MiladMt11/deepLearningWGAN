import torch
import sys 
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
sys.path.append('../')
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from functools import reduce
import numpy as np
import torchvision 
classes = np.arange(10)

def one_hot(labels):
    y = torch.eye(len(classes))
    return y[labels]

# Define the train and test sets
dset_train = FashionMNIST("../Dataset/", train=True, download=True,
				transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 [0.5], [0.5])
                             ]))
dset_test  = FashionMNIST("../Dataset/", download=True,train=False,
		transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 [0.5], [0.5])
                             ]))

def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

batch_size = 64
num_workers = 4
# The loaders perform the actual work
train_loader = DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.train_labels), num_workers= num_workers, pin_memory=cuda)
test_loader = DataLoader(dset_test, batch_size=batch_size,
                          sampler=stratified_sampler(dset_test.test_labels), num_workers= num_workers, pin_memory=cuda)
