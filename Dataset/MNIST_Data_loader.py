import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
import numpy as np
classes = np.arange(10)

def one_hot(labels):
    y = torch.eye(len(classes))
    return y[labels]

# Define the train and test sets
dset_train = MNIST("~/文档/Master/02456_Deep_learning/deepLearningWGAN/Dataset/", train=True, download=True, transform=ToTensor(), target_transform=one_hot)
dset_test  = MNIST("~/文档/Master/02456_Deep_learning/deepLearningWGAN/Dataset/", train=False, transform=ToTensor(), target_transform=one_hot)

def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

batch_size = 4096
# The loaders perform the actual work
train_loader = DataLoader(dset_train, batch_size=batch_size,
                          sampler=stratified_sampler(dset_train.train_labels), pin_memory=cuda)
test_loader = DataLoader(dset_test, batch_size=batch_size,
                          sampler=stratified_sampler(dset_test.test_labels), pin_memory=cuda)
