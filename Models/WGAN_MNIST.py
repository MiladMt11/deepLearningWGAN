import sys
sys.path.append('../')
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Dataset.MNIST_Data_loader import train_loader
from torchvision import utils

writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            *Conv(num_input, 1024),
            *Conv(1024, 512),
            *Conv(512, 256),
            nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_output, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.Net(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 64),
            *Conv(64, 256),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
        )
        self.Flatten = nn.Flatten()

    def forward(self, input):
        output = self.Net(input)
        return output

class WGAN():
    def __init__(self):
        self.G = Generator(100, 1).to(device)
        self.D = Discriminator(1).to(device)
        self.epoch = 0
        self.maxepochs = int(1e3)
        self.optim_G = torch.optim.RMSprop(self.G.parameters(), lr=5e-5)
        self.optim_D = torch.optim.RMSprop(self.G.parameters(), lr=5e-5)
        self.G_losses = []
        self.Real_losses = []
        self.Fake_losses = []
        self.weight_cliping_limit = 0.01
        self.D_iter = 5

    def train(self, train_loader):
        try:
            os.mkdir('../checkpoint/WGAN_MNIST/')
        except:
            pass
        try:
            self.load()
        except:
            pass
        while self.epoch < self.maxepochs:
            for x, _ in train_loader:
                x = x.to(device)
                batch_size = x.size(0)
                for p in self.D.parameters():
                    p.requires_grad = True
                for i in range(self.D_iter):
                    # train the discreiminator
                    self.D.zero_grad()
                    for p in self.D.parameters():
                        p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                    D_real = self.D(x)
                    loss_real = -D_real.mean(0).view(1)
                    loss_real.backward()
                    z = torch.randn((batch_size, 100, 1, 1)).to(device)
                    x_fake = self.G(z)
                    loss_fake = self.D(x_fake.detach())
                    loss_fake = loss_fake.mean(0).view(1)
                    loss_fake.backward()
                    self.optim_D.step()
                    loss_D = loss_fake + loss_real
                    self.Real_losses.append(loss_real.item())
                    self.Fake_losses.append(loss_fake.item())

                z = torch.randn((batch_size, 100, 1, 1)).to(device)
                self.G.zero_grad()
                for p in self.D.parameters():
                    p.requires_grad = False
                x_fake = self.G(z)
                loss_G = self.D(x_fake)
                loss_G = -loss_G.mean(0).view(1)
                # train the generator
                loss_G.backward()
                self.optim_G.step()
                self.G_losses.append(loss_G.item())
            print("epoch:{}, G_loss:{}".format(self.epoch, loss_G.cpu().detach().numpy()))
            print("D_real_loss:{}, D_fake_loss:{}".format(loss_real.cpu().detach().numpy(),
                                                          loss_fake.cpu().detach().numpy()))

            if self.epoch % 20 == 0:
                self.save()
                self.evaluate()
            self.epoch += 1

    def save(self):
        torch.save({"epoch": self.epoch,
                    "G_state_dict": self.G.state_dict(),
                    "optimizer_G": self.optim_G.state_dict(),
                    "losses_G": self.G_losses}, "../checkpoint/WGAN_MNIST/G.pth")
        torch.save({"D_state_dict": self.D.state_dict(),
                    "optimizer_D": self.optim_D.state_dict(),
                    "losses_fake": self.Fake_losses,
                    "losses_real": self.Real_losses}, "../checkpoint/WGAN_MNIST/D.pth")
        torch.save({"epoch": self.epoch,
                    "G_state_dict": self.G.state_dict(),
                    "optimizer_G": self.optim_G.state_dict(),
                    "losses_G": self.G_losses}, "../checkpoint/WGAN_MNIST/G_{}.pth".format(self.epoch))
        torch.save({"D_state_dict": self.D.state_dict(),
                    "optimizer_D": self.optim_D.state_dict(),
                    "losses_fake": self.Fake_losses,
                    "losses_real": self.Real_losses}, "../checkpoint/WGAN_MNIST/D_{}.pth".format(self.epoch))
        print("model saved!")

    def load(self):
        checkpoint_G = torch.load("../checkpoint/WGAN_MNIST/G.pth")
        checkpoint_D = torch.load("../checkpoint/WGAN_MNIST/D.pth")
        self.epoch = checkpoint_G["epoch"]
        self.G.load_state_dict(checkpoint_G["G_state_dict"])
        self.optim_G.load_state_dict(checkpoint_G["optimizer_G"])
        self.G_losses = checkpoint_G["losses_G"]
        self.D.load_state_dict(checkpoint_D["D_state_dict"])
        self.optim_D.load_state_dict(checkpoint_D["optimizer_D"])
        self.Fake_losses = checkpoint_D["losses_fake"]
        self.Real_losses = checkpoint_D["losses_real"]
        print("model loaded!")

    def evaluate(self):
        self.load()
        z = torch.randn((1, 100, 1, 1)).to(device)
        with torch.no_grad():
            fake_img = self.G(z)
            fake_img = fake_img.data.cpu()
            grid = utils.make_grid(fake_img)
            utils.save_image(grid, '../Results/WGAN_MNIST/img_generatori_iter_{}.png'.format(self.epoch))

if __name__ == '__main__':
    WGAN = WGAN()
    try:
        os.mkdir('../Results/WGAN_MNIST/')
    except:
        pass
    WGAN.train(train_loader)