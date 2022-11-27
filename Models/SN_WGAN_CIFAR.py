import sys
sys.path.append('../')
import torch
import torch.nn as nn
import os
# from torch.utils.tensorboard import SummaryWriter
from Dataset.CIFAR_dataloader import train_loader
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
class Res_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Conv = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            self.conv2,
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.extra = nn.Sequential()
        if in_channel != out_channel:
            self.extra = nn.Sequential(
                self.conv3,
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
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Conv = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.Conv_x = nn.Sequential(
            self.conv2,
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.blk1 = Res_Block(out_channel, out_channel)
        self.blk2 = Res_Block(out_channel, out_channel)
        self.blk3 = Res_Block(out_channel, out_channel)
        self.blk4 = Res_Block(out_channel, out_channel)
        self.out = nn.Sequential(
            self.conv3,
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

class Res_Block_D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Block_D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Conv = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(),
            nn.utils.parametrizations.spectral_norm(self.conv2),
            nn.ReLU()
        )
        self.extra = nn.Sequential()
        if in_channel != out_channel:
            self.extra = nn.Sequential(
                nn.utils.parametrizations.spectral_norm(self.conv3),
            )
        self.Relu = nn.ReLU()

    def forward(self, x):
        out = self.Conv(x)
        x = self.extra(x)
        out = self.Relu(out + x)
        return out

class ResNet_D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet_D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1,1), padding=1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Conv = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU()
        )
        self.Conv_x = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv2),
            nn.ReLU()
        )
        self.blk1 = Res_Block_D(out_channel, out_channel)
        self.blk2 = Res_Block_D(out_channel, out_channel)
        self.blk3 = Res_Block_D(out_channel, out_channel)
        self.blk4 = Res_Block_D(out_channel, out_channel)
        self.out = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv3),
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
            conv = nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1))
            bn = nn.BatchNorm2d(output_nums)
            nn.init.xavier_uniform_(conv.weight.data, 1.)
            layer.append(conv)
            layer.append(bn)
            layer.append(nn.ReLU(True))
            return layer
        self.conv = nn.ConvTranspose2d(64, num_output, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        nn.init.xavier_uniform_(self.conv.weight.data, 1.)
        self.Net = nn.Sequential(
            *Conv(num_input, 1024),
            ResNet(1024, 1024),
            *Conv(1024, 512),
            ResNet(512, 512),
            *Conv(512, 256),
            ResNet(256, 256),
            *Conv(256, 64),
            ResNet(64, 64),
            self.conv,
            nn.Tanh()
        )

    def forward(self, input):
        output = self.Net(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_nums, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv3.weight.data, 1.)
        self.Net = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(self.conv1),
            nn.ReLU(True),
            ResNet_D(64,64),
            nn.utils.parametrizations.spectral_norm(self.conv2),
            nn.ReLU(True),
            ResNet_D(256, 256),
            nn.utils.parametrizations.spectral_norm(self.conv3),
            nn.ReLU(True),
            ResNet_D(512, 512)
        )
        self.conv = nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1,1), padding=0)

    def forward(self, input):
        output = self.Net(input)
        output = self.conv(output)
        return output

class WGAN():
    def __init__(self):
        self.G = Generator(100, 3).to(device)
        self.D = Discriminator(3).to(device)
        self.epoch = 0
        self.maxepochs = int(1e3)
        self.optim_G = torch.optim.RMSprop(self.G.parameters(), lr=5e-5)
        self.optim_D = torch.optim.RMSprop(self.D.parameters(), lr=5e-5)
        self.G_losses = []
        self.Real_losses = []
        self.Fake_losses = []
        self.weight_cliping_limit = 0.01
        self.D_iter = 5
        self.G_best = Generator(100, 3).to(device)
        self.fid_score = []
        self.best_fid = 1e10
        self.path = 'SN_WGAN_CIFAR/'

    def train(self, train_loader):
        try:
            os.mkdir('../checkpoint/'+self.path)
        except:
            pass
        try:
            self.load()
        except:
            print('parameters initialization')
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
                    # for p in self.D.parameters():
                    #     p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
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
        print("model saved!")

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
        print("model loaded!")

    def evaluate(self):
        self.load()
        z = torch.randn((1, 100, 1, 1)).to(device)
        with torch.no_grad():
            fake_img = self.G(z)
            fake_img = fake_img.data.cpu()
            grid = utils.make_grid(fake_img)
            utils.save_image(grid, '../Results/SN_WGAN_CIFAR/img_generatori_iter_{}.png'.format(self.epoch))

if __name__ == '__main__':
    WGAN = WGAN()
    WGAN.path = 'SN_WGAN_CIFAR/'
    try:
        os.mkdir('../Results/'+WGAN.path)
    except:
        pass
    WGAN.train(train_loader)