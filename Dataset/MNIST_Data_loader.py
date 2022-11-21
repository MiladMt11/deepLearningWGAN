import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce
import numpy as np
import torchvision 
classes = np.arange(10)

def one_hot(labels):
    y = torch.eye(len(classes))
    return y[labels]

# Define the train and test sets
dset_train = MNIST("/zhome/5f/d/136189/deepLearningWGAN/Dataset/", train=True, download=True, target_transform=one_hot, 
				transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
dset_test  = MNIST("/zhome/5f/d/136189/deepLearningWGAN/Dataset/", train=False, target_transform=one_hot,
		transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

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
