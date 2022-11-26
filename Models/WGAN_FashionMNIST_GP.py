import sys
sys.path.append('../')
import torch
import torch.nn as nn
import os
from torch import autograd
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Dataset.FASHION_MNIST_Data_loader import train_loader
from torchvision import utils
from torchmetrics.image.fid import FrechetInceptionDistance
# writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fid = FrechetInceptionDistance().to(device)
def get_fid(real_images, fake_images):
    '''
        Takes real image batch and generated 'fake' image batch
        Returns FID score, using the pytorch.metrics package
    '''
    # add 2 extra channels for MNIST (as required by InceptionV3
    if real_images.shape[1] != 3:
        real_images = torch.cat([real_images, real_images, real_images], 1)
    if fake_images.shape[1] != 3:
        fake_images = torch.cat([fake_images, fake_images, fake_images], 1)

    # if images not uint8 format, convert them (required format by fid model)
    if real_images.dtype != torch.uint8 or fake_images.dtype != torch.uint8:
        real_images = real_images.type(torch.cuda.ByteTensor)
        fake_images = fake_images.type(torch.cuda.ByteTensor)

    fid.update(real_images, real=True)  # <--- currently running out of memory here
    fid.update(fake_images, real=False)
    return fid.compute().item()
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
        self.optim_D = torch.optim.RMSprop(self.D.parameters(), lr=5e-5)
        self.G_losses = []
        self.Real_losses = []
        self.Fake_losses = []
        self.D_iter = 5
        self.G_best = Generator(100, 1).to(device)
        self.fid_score = []
        self.best_fid = 1e10
        self.lambda_term = 10

    def train(self, train_loader):
        try:
            os.mkdir('../checkpoint/WGAN_FashionMNIST_GP/')
        except:
            pass
        try:
            self.load()
        except:
            pass
        self.G.train()
        self.D.train()
        while self.epoch < self.maxepochs + 1:
            for x, _ in train_loader:
                x = x.to(device)
                batch_size = x.size(0)
                for p in self.D.parameters():
                    p.requires_grad = True
                for i in range(self.D_iter):
                    # train the discreiminator
                    self.D.zero_grad()
                    D_real = self.D(x)
                    loss_real = -D_real.mean(0).view(1)
                    loss_real.backward()
                    z = torch.randn((batch_size, 100, 1, 1)).to(device)
                    x_fake = self.G(z).detach()
                    loss_fake = self.D(x_fake)
                    loss_fake = loss_fake.mean(0).view(1)
                    loss_fake.backward()
                    # gradient penalty
                    gp = self.calculate_gradient_penalty(x.data, x_fake.data)
                    gp.backward()
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
                fid_score = get_fid(x, x_fake.detach())
                self.fid_score.append(fid_score)
                if fid_score < self.best_fid:
                    self.best_fid = fid_score
                    self.G_best = self.G
                print("FID score: {}".format(fid_score))
            self.epoch += 1

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(real_images.size(0),1,1,1).uniform_(0,1).to(device)
        eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3))

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def save(self):
        torch.save({"epoch": self.epoch,
                    "G_state_dict": self.G.state_dict(),
                    "G_best_state_dict": self.G_best.state_dict(),
                    "optimizer_G": self.optim_G.state_dict(),
                    "losses_G": self.G_losses,
                    "FID scores": self.fid_score,
                    "Best FID score": self.best_fid}, "../checkpoint/WGAN_FashionMNIST_GP/G.pth")
        torch.save({"D_state_dict": self.D.state_dict(),
                    "optimizer_D": self.optim_D.state_dict(),
                    "losses_fake": self.Fake_losses,
                    "losses_real": self.Real_losses}, "../checkpoint/WGAN_FashionMNIST_GP/D.pth")
        if self.epoch == self.maxepochs:
            torch.save({"epoch": self.epoch,
                        "G_state_dict": self.G.state_dict(),
                        "G_best_state_dict": self.G_best.state_dict(),
                        "optimizer_G": self.optim_G.state_dict(),
                        "losses_G": self.G_losses,
                        "FID scores": self.fid_score,
                        "Best FID score": self.best_fid}, "../checkpoint/WGAN_FashionMNIST_GP/G_{}.pth")
            torch.save({"D_state_dict": self.D.state_dict(),
                        "optimizer_D": self.optim_D.state_dict(),
                        "losses_fake": self.Fake_losses,
                        "losses_real": self.Real_losses}, "../checkpoint/WGAN_FashionMNIST_GP/D_{}.pth".format(self.epoch))
        print("model saved!")

    def load(self):
        checkpoint_G = torch.load("../checkpoint/WGAN_FashionMNIST_GP/G.pth")
        checkpoint_D = torch.load("../checkpoint/WGAN_FashionMNIST_GP/D.pth")
        self.epoch = checkpoint_G["epoch"]
        self.G.load_state_dict(checkpoint_G["G_state_dict"])
        self.G_best.load_state_dict(checkpoint_G["G_best_state_dict"])
        self.optim_G.load_state_dict(checkpoint_G["optimizer_G"])
        self.G_losses = checkpoint_G["losses_G"]
        self.fid_score = checkpoint_G["FID scores"]
        self.best_fid = checkpoint_G["Best FID score"]
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
            utils.save_image(grid, '../Results/WGAN_FashionMNIST_GP/img_generatori_iter_{}.png'.format(self.epoch))

if __name__ == '__main__':
    WGAN = WGAN()
    try:
        os.mkdir('../Results/WGAN_FashionMNIST_GP/')
    except:
        pass
    WGAN.train(train_loader)