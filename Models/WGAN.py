import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Dataset.CIFAR_dataloader import train_loader
writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer

        self.Net = nn.Sequential(
            *Conv(num_input, 256),
            *Conv(256, 128),
            *Conv(128, 64),
            *Conv(64, 32),
            nn.ConvTranspose2d(32, num_output, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
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
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.LeakyReLU(0.2))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 32),
            *Conv(32, 64),
            *Conv(64, 128),
        )
        self.conv = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0)
        self.Flatten = nn.Flatten()

    def forward(self, input):
        output = self.Net(input)
        output = self.conv(output)
        return output

class WGAN():
    def __init__(self):
        self.G = Generator(100, 3).to(device)
        self.D = Discriminator(3).to(device)
        self.epochs = int(1e3)

    def train(self, train_loader):
        optim_G = torch.optim.RMSprop(self.G.parameters(), lr=5e-5)
        optim_D = torch.optim.RMSprop(self.D.parameters(), lr=5e-5)
        self.load()
        for epoch in range(self.epochs):
            for x, _ in train_loader:
                x = x.to(device)
                batch_size = x.size(0)
                self.D.zero_grad()
                self.G.zero_grad()
                D_real = self.D(x)
                loss_real = -D_real.mean(0).view(1)
                z = torch.randn((batch_size, 100, 1, 1)).to(device)
                x_fake = self.G(z)
                loss_fake = self.D(x_fake.detach())
                loss_fake = loss_fake.mean(0).view(1)
                loss_D = loss_fake + loss_real
                # train the discreiminator
                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()
                x_fake = self.G(z)
                loss_G = self.D(x_fake)
                loss_G = -loss_G.mean()
                # train the generator
                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()
            if epoch % 20 == 0:
                self.save()

    def save(self):
        torch.save(self.G.state_dict(), "../checkpoint/WGAN/G.pth")
        torch.save(self.D.state_dict(), "../checkpoint/WGAN/D.pth")
        print("model saved!")

    def load(self):
        self.G.load_state_dict(torch.load("../checkpoint/WGAN/G.pth"))
        self.D.load_state_dict(torch.load("../checkpoint/WGAN/D.pth"))
        print("load")

if __name__ == '__main__':
    WGAN = WGAN()
    WGAN.train(train_loader)
