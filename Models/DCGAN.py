import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.LeakyReLU(0.2))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 64),
            *Conv(64, 256),
            *Conv(256, 512),
        )
        self.Flatten = nn.Flatten()
        self.dense = nn.Linear(512 * 4 * 4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        output = self.Net(input)
        output = self.Flatten(output)
        output = self.dense(output)
        output = self.activation(output)
        return output

if __name__ == '__main__':
    z = torch.randn((1, 100, 1, 1)).to(device)
    G = Generator(100, 3).to(device)
    output = G(z)
    fake_image = torch.squeeze(output, dim=0)
    fake_image = fake_image.view(32, 32, 3)
    fake_image = fake_image.cpu().detach().numpy()
    writer.add_image("test", fake_image,2,dataformats='HWC')
