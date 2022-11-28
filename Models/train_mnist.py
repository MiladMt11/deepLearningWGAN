import sys
sys.path.append('../')
from WGAN_MNIST import WGAN
from WGAN_MNIST_GP import WGAN as WGAN_GP
from SN_WGAN_MNIST import WGAN as SN_WGAN
from Dataset.MNIST_Data_loader import train_loader
import os

if __name__ == '__main__':
    for i in range(3):
        _WGAN = WGAN()
        _WGAN.path = "WGAN_MNIST_{}/".format(i)
        try:
            os.mkdir('../Results/WGAN_MNIST/')
        except:
            pass
        _WGAN.train(train_loader)

        _WGAN_GP = WGAN_GP()
        _WGAN_GP.path = 'WGAN_MNIST_GP_{}/'.format(i)
        try:
            os.mkdir('../Results/WGAN_MNIST_GP/')
        except:
            pass
        _WGAN_GP.train(train_loader)

        _SN_WGAN = SN_WGAN()
        _SN_WGAN.path = '_SN_WGAN_MNIST_{}/'.format(i)
        try:
            os.mkdir('../Results/_SN_WGAN_MNIST/')
        except:
            pass
        _SN_WGAN.train(train_loader)