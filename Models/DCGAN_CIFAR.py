import sys
sys.path.append('H:/Courses_files/Master/02456_Deep_learning/deepLearningWGAN')
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Dataset.CIFAR_dataloader import train_loader
import os
# import wandb
from torchvision import utils

# wandb.init(project="DCGAN-project", entity="projekt17")
# wandb.config = {
#   "learning_rate": 1e-4,
#   "epochs": int(1e3),
#   "batch_size": 64
# }

writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
            *Conv(256, 64),
            nn.ConvTranspose2d(64, num_output, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
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
            layer.append(nn.LeakyReLU(0.2, inplace=True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 64),
            *Conv(64, 256),
            *Conv(256, 512),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.Net(input)
        output = torch.squeeze(output, dim=-1)
        output = torch.squeeze(output, dim=-1)
        return output

class DCGAN():
    def __init__(self):
        self.G = Generator(100, 3).to(device)
        self.D = Discriminator(3).to(device)
        self.epochs = int(1e3)
        self.loss_func = nn.BCELoss()

    def train(self, train_loader):
        G_losses = []
        Real_losses = []
        Fake_losses = []
        try:
            os.mkdir('../checkpoint/DCGAN_CIFAR/')
        except:
            pass
        optim_G = torch.optim.Adam(self.G.parameters(),lr=1e-4, betas=(0.5, 0.999))
        optim_D = torch.optim.Adam(self.D.parameters(),lr=1e-4, betas=(0.5, 0.999))
        try:
            self.load()
        except:
            self.D.apply(weights_init)
            print('parameters initialization')
        train_loader = train_loader.to(device)
        for epoch in range(self.epochs):
            for x, _ in train_loader:
                # x = x.to(device)
                batch_size = x.size(0)
                true_label = torch.ones(batch_size, 1).to(device)
                fake_label = torch.zeros(batch_size, 1).to(device)
                self.D.zero_grad()
                self.G.zero_grad()
                D_real = self.D(x)
                loss_real = self.loss_func(D_real, true_label)
                loss_real.backward()
                z = torch.randn((batch_size, 100, 1, 1)).to(device)
                x_fake = self.G(z)
                D_fake = self.D(x_fake.detach())
                loss_fake = self.loss_func(D_fake, fake_label)
                loss_fake.backward()
                # loss_D = loss_fake + loss_real
                # wandb.log({"loss_D": loss_D})
                # train the discreiminator
                # loss_D.backward()
                optim_D.step()
                Real_losses.append(loss_real.item())
                Fake_losses.append(loss_fake.item())
                x_fake = self.G(z)
                loss_G = self.D(x_fake)
                loss_G = self.loss_func(loss_G, true_label)
                # wandb.log({"loss_G": loss_G})
                # train the generator
                loss_G.backward()
                optim_G.step()
                G_losses.append(loss_G.item())
            print("epoch:{}, G_loss:{}".format(epoch, loss_G.cpu().detach().numpy()))
            print("D_real_loss:{}, D_fake_loss:{}".format(loss_real.cpu().detach().numpy(),
                                                                   loss_fake.cpu().detach().numpy()))
            if epoch % 20 == 0:
                self.save()
                self.evaluate(epoch = epoch)

    def save(self):
        torch.save(self.G.state_dict(), "../checkpoint/DCGAN_CIFAR/G.pth")
        torch.save(self.D.state_dict(), "../checkpoint/DCGAN_CIFAR/D.pth")
        print("model saved!")

    def load(self):
        self.G.load_state_dict(torch.load("../checkpoint/DCGAN_CIFAR/G.pth"))
        self.D.load_state_dict(torch.load("../checkpoint/DCGAN_CIFAR/D.pth"))
        print("model loaded!")

    def evaluate(self, epoch = 0):
        self.load()
        z = torch.randn((1, 100, 1, 1)).to(device)
        with torch.no_grad():
            fake_img = self.G(z)
            fake_img = fake_img.data.cpu()
            grid = utils.make_grid(fake_img)
            utils.save_image(grid, '../Results/DCGAN_CIFAR/img_generatori_iter_{}.png'.format(epoch))

if __name__ == '__main__':
    DCGAN = DCGAN()
    try:
        os.mkdir('../Results/DCGAN_CIFAR/')
    except:
        pass
    DCGAN.train(train_loader)
    # DCGAN.evaluate()
