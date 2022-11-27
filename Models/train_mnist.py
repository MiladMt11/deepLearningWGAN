import sys
sys.path.append('../')
from WGAN_MNIST import WGAN
from WGAN_MNIST_GP import WGAN as WGAN_GP
from SN_WGAN_MNIST import WGAN as SN_WGAN
from Dataset.MNIST_Data_loader import train_loader
import os

if __name__ == '__main__':
    WGAN = WGAN()
    try:
        os.mkdir('../Results/WGAN_MNIST/')
    except:
        pass
    WGAN.train(train_loader)
    WGAN_GP = WGAN_GP()
    try:
        os.mkdir('../Results/WGAN_MNIST_GP/')
    except:
        pass
    WGAN_GP.train(train_loader)
    SN_WGAN = SN_WGAN()
    try:
        os.mkdir('../Results/SN_WGAN_MNIST/')
    except:
        pass
    SN_WGAN.train(train_loader)