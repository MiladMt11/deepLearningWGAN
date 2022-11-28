import sys
sys.path.append('../')
from WGAN_FashionMNIST import WGAN
from WGAN_FashionMNIST_GP import WGAN as WGAN_GP
from SN_WGAN_FashionMNIST import WGAN as SN_WGAN
from FASHION_MNIST_DCGAN import DCGAN
from Dataset.FASHION_MNIST_Data_loader import train_loader
import os

if __name__ == '__main__':
    try:
        os.mkdir('../Results/WGAN_FashionMNIST/')
    except:
        pass
    try:
        os.mkdir('../Results/WGAN_FashionMNIST_GP/')
    except:
        pass
    try:
        os.mkdir('../Results/SN_WGAN_FashionMNIST/')
    except:
        pass
    try:
        os.mkdir('../Results/DCGAN_FashionMNIST/')
    except:
        pass
    for i in range(3):
        _WGAN = WGAN()
        _WGAN.path = "WGAN_FashionMNIST_{}/".format(i)
        _WGAN.train(train_loader)

        _WGAN_GP = WGAN_GP()
        _WGAN_GP.path = 'WGAN_FashionMNIST_GP_{}/'.format(i)
        _WGAN_GP.train(train_loader)

        _SN_WGAN = SN_WGAN()
        _SN_WGAN.path = 'SN_WGAN_FashionMNIST_{}/'.format(i)
        _SN_WGAN.train(train_loader)

        _DCGAN = DCGAN()
        _DCGAN.path = 'DCGAN_FashionMNIST_{}/'.format(i)
        _DCGAN.train(train_loader)