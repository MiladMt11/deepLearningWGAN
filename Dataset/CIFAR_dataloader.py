import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from functools import reduce
import numpy as np

# The digit classes to use, these need to be in order because
# we are using one-hot representation
classes = np.arange(10)

def one_hot(labels):
    y = torch.eye(len(classes))
    return y[labels]

# Define the train and test sets
dset_train = CIFAR10("H:/Courses_files/Master/02456_Deep_learning/deepLearningWGAN/Dataset/", train=True, download=True, transform=ToTensor(), target_transform=one_hot)
dset_test  = CIFAR10("H:/Courses_files/Master/02456_Deep_learning/deepLearningWGAN/Dataset/", train=False, transform=ToTensor(), target_transform=one_hot)

def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

batch_size = 64
# The loaders perform the actual work
train_loader = DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.targets), pin_memory=cuda)
test_loader  = DataLoader(dset_test, batch_size=batch_size,
                          sampler=stratified_sampler(dset_test.targets), pin_memory=cuda)