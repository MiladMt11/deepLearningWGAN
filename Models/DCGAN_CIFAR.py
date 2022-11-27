import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Dataset.CIFAR_dataloader import train_loader
import os
# import wandb
from torchvision import utils
from torchmetrics.image.fid import FrechetInceptionDistance
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

writer = SummaryWriter()

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

class Res_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Block, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.extra = nn.Sequential()
        if in_channel != out_channel:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1)),
                nn.BatchNorm2d(out_channel)
            )
        self.Relu = nn.ReLU()

    def forward(self, x):
        out = self.Conv(x)
        x = self.extra(x)
        out = self.Relu(out + x)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.Conv_x = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.blk1 = Res_Block(out_channel, out_channel)
        self.blk2 = Res_Block(out_channel, out_channel)
        self.blk3 = Res_Block(out_channel, out_channel)
        self.blk4 = Res_Block(out_channel, out_channel)
        self.out = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.Relu = nn.ReLU()

    def forward(self, x):
        out = self.Conv(x)
        x = self.Conv_x(x)
        out = self.blk4(self.blk3(self.blk2(self.blk1(out))))
        # out = self.blk3(self.blk2(self.blk1(out)))
        # out = self.blk2(self.blk1(out))
        out = self.Relu(x + out)
        out = self.out(out)
        return out

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
            ResNet(1024, 1024),
            *Conv(1024, 512),
            ResNet(512, 512),
            *Conv(512, 256),
            ResNet(256, 256),
            *Conv(256, 64),
            ResNet(64, 64),
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
            ResNet(64, 64),
            *Conv(64, 256),
            ResNet(256, 256),
            *Conv(256, 512),
            ResNet(512, 512),
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
        self.epoch = 0
        self.maxepochs = int(1e3)
        self.loss_func = nn.BCELoss()
        self.optim_G = torch.optim.Adam(self.G.parameters(),lr=1e-4, betas=(0.5, 0.999))
        self.optim_D = torch.optim.Adam(self.D.parameters(),lr=1e-4, betas=(0.5, 0.999))
        self.G_losses = []
        self.Real_losses = []
        self.Fake_losses = []
        self.G_best = Generator(100, 3).to(device)
        self.fid_score = []
        self.best_fid = 1e10
        self.path = 'DCGAN_CIFAR/'

    def train(self, train_loader):
        try:
            os.mkdir('../checkpoint/'+self.path)
        except:
            pass
        try:
            self.load()
        except:
            self.D.apply(weights_init)
            print('parameters initialization')
        while self.epoch < self.maxepochs + 1:
            for x, _ in train_loader:
                x = x.to(device)
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
                self.optim_D.step()
                self.Real_losses.append(loss_real.item())
                self.Fake_losses.append(loss_fake.item())
                x_fake = self.G(z)
                loss_G = self.D(x_fake)
                loss_G = self.loss_func(loss_G, true_label)
                # wandb.log({"loss_G": loss_G})
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

    def save(self):
        torch.save({"epoch": self.epoch,
                    "G_state_dict": self.G.state_dict(),
                    "G_best_state_dict": self.G_best.state_dict(),
                    "optimizer_G": self.optim_G.state_dict(),
                    "losses_G": self.G_losses,
                    "FID scores": self.fid_score,
                    "Best FID score": self.best_fid}, "../checkpoint/"+self.path+"G.pth")
        torch.save({"D_state_dict": self.D.state_dict(),
                    "optimizer_D": self.optim_D.state_dict(),
                    "losses_fake": self.Fake_losses,
                    "losses_real": self.Real_losses}, "../checkpoint/"+self.path+"D.pth")
        if self.epoch == self.maxepochs:
            torch.save({"epoch": self.epoch,
                        "G_state_dict": self.G.state_dict(),
                        "G_best_state_dict": self.G_best.state_dict(),
                        "optimizer_G": self.optim_G.state_dict(),
                        "losses_G": self.G_losses,
                        "FID scores": self.fid_score,
                        "Best FID score": self.best_fid}, "../checkpoint/"+self.path+"G_{}.pth")
            torch.save({"D_state_dict": self.D.state_dict(),
                        "optimizer_D": self.optim_D.state_dict(),
                        "losses_fake": self.Fake_losses,
                        "losses_real": self.Real_losses}, "../checkpoint/"+self.path+"D_{}.pth".format(self.epoch))
        print("model saved! path: "+self.path)

    def load(self):
        checkpoint_G = torch.load("../checkpoint/"+self.path+"G.pth")
        checkpoint_D = torch.load("../checkpoint/"+self.path+"D.pth")
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
        print("model loaded! path: "+self.path)

    def evaluate(self):
        self.load()
        z = torch.randn((1, 100, 1, 1)).to(device)
        with torch.no_grad():
            fake_img = self.G(z)
            fake_img = fake_img.data.cpu()
            grid = utils.make_grid(fake_img)
            utils.save_image(grid, '../Results/DCGAN_CIFAR/img_generatori_iter_{}.png'.format(self.epoch))

if __name__ == '__main__':
    DCGAN = DCGAN()
    DCGAN.path = 'DCGAN_CIFAR/'
    try:
        os.mkdir('../Results/')
    except:
        pass
    DCGAN.train(train_loader)
    # DCGAN.evaluate()
