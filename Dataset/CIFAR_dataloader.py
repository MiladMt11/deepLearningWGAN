import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from functools import reduce
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

# The digit classes to use, these need to be in order because
# we are using one-hot representation
classes = np.arange(10)

def one_hot(labels):
    y = torch.eye(len(classes))
    return y[labels]

# Define the train and test sets
dataset = dset.CIFAR10(root="H:/Courses_files/Master/02456_Deep_learning/deepLearningWGAN/Dataset/",
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                           ]))
# dset_train = CIFAR10("H:/Courses_files/Master/02456_Deep_learning/deepLearningWGAN/Dataset/", train=True, download=True, transform=ToTensor(), target_transform=one_hot)
# dset_test  = CIFAR10("H:/Courses_files/Master/02456_Deep_learning/deepLearningWGAN/Dataset/", train=False, transform=ToTensor(), target_transform=one_hot)

def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

batch_size = 1024
workers = 2
# The loaders perform the actual work
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers, pin_memory=True)
# train_loader = DataLoader(dset_train, batch_size=batch_size,
#                           sampler=stratified_sampler(dset_train.targets), pin_memory=cuda)
# test_loader  = DataLoader(dset_test, batch_size=batch_size,
#                           sampler=stratified_sampler(dset_test.targets), pin_memory=cuda)
if __name__ == '__main__':
    real_batch = next(iter(train_loader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()