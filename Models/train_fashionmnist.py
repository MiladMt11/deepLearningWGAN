import sys
sys.path.append('../')
from WGAN_FashionMNIST import WGAN
from WGAN_FashionMNIST_GP import WGAN as WGAN_GP
from SN_WGAN_FashionMNIST import WGAN as SN_WGAN
from FASHION_MNIST_DCGAN import DCGAN
from Dataset.FASHION_MNIST_Data_loader import train_loader
import os

if __name__ == '__main__':
    for i in range(3):
        WGAN = WGAN()
        path = "WGAN_FashionMNIST_{}/".format(i)
        try:
            os.mkdir('../Results/WGAN_FashionMNIST/')
        except:
            pass
        WGAN.path = path
        WGAN.train(train_loader)

        WGAN_GP = WGAN_GP()
        path = 'WGAN_FashionMNIST_GP_{}/'.format(i)
        try:
            os.mkdir('../Results/WGAN_FashionMNIST_GP/')
        except:
            pass
        WGAN_GP.path = path
        WGAN_GP.train(train_loader)

        SN_WGAN = SN_WGAN()
        path = 'SN_WGAN_FashionMNIST_{}/'.format(i)
        try:
            os.mkdir('../Results/SN_WGAN_FashionMNIST/')
        except:
            pass
        SN_WGAN.path = path
        SN_WGAN.train(train_loader)

        DCGAN = DCGAN()
        path = 'DCGAN_FashionMNIST_{}/'.format(i)
        try:
            os.mkdir('../Results/DCGAN_FashionMNIST/')
        except:
            pass
        DCGAN.path = path
        DCGAN.train(train_loader)