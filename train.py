from Data_loader import *
from WGAN.WGAN import WGAN
from DCGAN.DCGAN import DCGAN
import os

if __name__ == '__main__':
    train_set = 'CIFAR'
    _WGAN = WGAN(ResNet=True, gradient_penalty=False, spectral_norm=True, train_set=train_set, iter=0)
    _WGAN.train(train_loader_cifar)
    _DCGAN = DCGAN(ResNet=False, train_set='CIFAR', iter=0)
    _DCGAN.train(train_loader_cifar)
