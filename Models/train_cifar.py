import sys
sys.path.append('../')
from WGAN_CIFAR import WGAN
from WGAN_CIFAR_GP import WGAN as WGAN_GP
from SN_WGAN_CIFAR import WGAN as SN_WGAN
from DCGAN_CIFAR import DCGAN
from Dataset.CIFAR_dataloader import train_loader
import os

if __name__ == '__main__':
    for i in range(3):
        WGAN = WGAN()
        WGAN.path = "WGAN_CIFAR_{}/".format(i)
        try:
            os.mkdir('../Results/WGAN_CIFAR')
        except:
            pass
        WGAN.train(train_loader)

        WGAN_GP = WGAN_GP()
        WGAN_GP.path = 'WGAN_CIFAR_GP_{}/'.format(i)
        try:
            os.mkdir('../Results/WGAN_CIFAR_GP/')
        except:
            pass
        WGAN_GP.train(train_loader)

        SN_WGAN = SN_WGAN()
        SN_WGAN.path = 'SN_WGAN_CIFAR_{}/'.format(i)
        try:
            os.mkdir('../Results/SN_WGAN_CIFAR/')
        except:
            pass
        SN_WGAN.train(train_loader)

        DCGAN = DCGAN()
        DCGAN.path = 'DCGAN_CIFAR_{}/'.format(i)
        try:
            os.mkdir('../Results/DCGAN_CIFAR/')
        except:
            pass
        DCGAN.train(train_loader)