import sys
sys.path.append('../')
from WGAN_CIFAR import WGAN
from WGAN_CIFAR_GP import WGAN as WGAN_GP
from SN_WGAN_CIFAR import WGAN as SN_WGAN
from DCGAN_CIFAR import DCGAN
from Dataset.CIFAR_dataloader import train_loader
import os

if __name__ == '__main__':
    try:
        os.mkdir('../Results/WGAN_CIFAR')
    except:
        pass
    try:
        os.mkdir('../Results/WGAN_CIFAR_GP/')
    except:
        pass
    try:
        os.mkdir('../Results/DCGAN_CIFAR/')
    except:
        pass
    try:
        os.mkdir('../Results/SN_WGAN_CIFAR/')
    except:
        pass

    for i in range(3):
        _WGAN = WGAN()
        _WGAN.path = "WGAN_CIFAR_{}/".format(i)
        _WGAN.train(train_loader)

        _WGAN_GP = WGAN_GP()
        _WGAN_GP.path = 'WGAN_CIFAR_GP_{}/'.format(i)
        _WGAN_GP.train(train_loader)

        _SN_WGAN = SN_WGAN()
        _SN_WGAN.path = 'SN_WGAN_CIFAR_{}/'.format(i)
        _SN_WGAN.train(train_loader)

        _DCGAN = DCGAN()
        _DCGAN.path = 'DCGAN_CIFAR_{}/'.format(i)
        _DCGAN.train(train_loader)