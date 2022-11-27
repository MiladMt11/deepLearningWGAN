import sys
sys.path.append('../')
from WGAN_MNIST import WGAN
from WGAN_MNIST_GP import WGAN as WGAN_GP
from SN_WGAN_MNIST import WGAN as SN_WGAN
from Dataset.MNIST_Data_loader import train_loader
import os

if __name__ == '__main__':
    for i in range(3):
        WGAN = WGAN()
        path = "WGAN_MNIST_{}/".format(i)
        try:
            os.mkdir('../Results/'+path)
        except:
            pass
        WGAN.path = path
        WGAN.train(train_loader)

        WGAN_GP = WGAN_GP()
        path = 'WGAN_MNIST_GP_{}/'.format(i)
        try:
            os.mkdir('../Results/'+path)
        except:
            pass
        WGAN_GP.path = path
        WGAN_GP.train(train_loader)

        SN_WGAN = SN_WGAN()
        path = 'SN_WGAN_MNIST_{}/'.format(i)
        try:
            os.mkdir('../Results/'+path)
        except:
            pass
        SN_WGAN.path = path
        SN_WGAN.train(train_loader)