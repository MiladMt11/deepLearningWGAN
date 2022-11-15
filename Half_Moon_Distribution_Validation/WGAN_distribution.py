import numpy as np
import seaborn as sns
import torch
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torch.optim as optim
from sklearn import datasets
import matplotlib.pylab as plt
import matplotlib.animation
from scipy.stats import multivariate_normal
from time import time

writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def moon_data(n):

    x,y = datasets.make_moons(n_samples=n,noise=0.1)
    t = torch.tensor(x).to(device)
    return t.type(torch.FloatTensor)

class Generator(nn.Module):
    def __init__(self, num_input):
        super(Generator, self).__init__()
        def Linear(input_nums, output_nums):
            layer = []
            layer.append(nn.Linear(input_nums, output_nums))
            layer.append(nn.LeakyReLU())
            return layer

        self.Net = nn.Sequential(
            *Linear(num_input, 128),
            *Linear(128, 64),
            *Linear(64, 32),
            nn.Linear(32, 2)
        )

    def forward(self, input):
        output = self.Net(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator, self).__init__()
        def Linear(input_nums, output_nums):
            layer = []
            layer.append(nn.Linear(input_nums, output_nums))
            layer.append(nn.LeakyReLU())
            return layer

        self.Net = nn.Sequential(
            *Linear(input_nums, 128),
            *Linear(128, 64),
            *Linear(64, 32),
            nn.Linear(32, 2),
        )

    def forward(self, input):
        output = self.Net(input)
        return output

class WGAN():
    def __init__(self):
        self.G = Generator(2).to(device)
        self.D = Discriminator(2).to(device)
        self.epochs = int(4e3)
        self.batch_size = 64
        self.weight_cliping_limit = 0.01

    def sample_gen(self, num=1000):
        with torch.no_grad():
            return self.G(torch.randn((num, 2), device=device))  # changed to 2

    def train(self):
        optim_G = torch.optim.RMSprop(self.G.parameters(), lr=1e-5)
        optim_D = torch.optim.RMSprop(self.D.parameters(), lr=1e-5)
        try:
            self.load()
        except:
            pass
        for epoch in range(self.epochs):
            for batch in range(self.batch_size):
                x = moon_data(self.batch_size)
                x = x.to(device)
                z = torch.randn((self.batch_size,2), device=device)
                for p in self.D.parameters():
                    p.requires_grad = True
                self.D.zero_grad()
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                D_real = self.D(x)
                loss_real = -D_real.mean()
                x_fake = self.G(z)
                loss_fake = self.D(x_fake.detach())
                loss_fake = loss_fake.mean()
                loss_D = loss_fake + loss_real
                # train the discreiminator
                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()
                for p in self.D.parameters():
                    p.requires_grad = False
                self.G.zero_grad()
                x_fake = self.G(z)
                loss_G = self.D(x_fake)
                loss_G = -loss_G.mean()
                # train the generator
                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()
            if epoch % 200 == 0:
                print("epoch: {} Loss real of discriminator: {}".format(epoch, loss_real))
                print("epoch: {} Loss fake of discriminator: {}".format(epoch, loss_fake))
                print("epoch: {} Loss of Generator: {}".format(epoch, loss_G))
                self.save()

    def save(self):
        torch.save(self.G.state_dict(), "H:/Courses_files/Master/"
                                        "02456_Deep_learning/deepLearningWGAN/checkpoint/"
                                        "WGAN_distribution/G_distribution.pth")
        torch.save(self.D.state_dict(), "H:/Courses_files/Master/"
                                        "02456_Deep_learning/deepLearningWGAN/checkpoint/"
                                        "WGAN_distribution/D_distribution.pth")
        print("model saved!")

    def load(self):
        self.G.load_state_dict(torch.load("H:/Courses_files/Master"
                                              "/02456_Deep_learning/deepLearningWGAN/checkpoint/"
                                              "WGAN_distribution/G_distribution.pth"))
        self.D.load_state_dict(torch.load("H:/Courses_files/Master"
                            "/02456_Deep_learning/deepLearningWGAN/checkpoint/"
                            "WGAN_distribution/D_distribution.pth"))
        print("load G and D!")

    def evaluate(self, num = 1000):
        self.load()
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        z = torch.randn((num, 2), device=device)
        with torch.no_grad():
            fake_s = self.G(z)
        real_ = moon_data(num)
        sns.scatterplot(x=torch.flatten(real_[:, :1]).cpu().numpy(), y=torch.flatten(real_[:, 1:]).cpu().numpy(), ax=ax[1],
                        color="blue")
        sns.scatterplot(x=torch.flatten(fake_s[:, :1]).cpu().numpy(), y=torch.flatten(fake_s[:, 1:]).cpu().numpy(), ax=ax[1],
                        color="red")
        plt.show()

if __name__ == '__main__':
    WGAN = WGAN()
    # WGAN.train()
    WGAN.evaluate(1000)