import torch
import torch.nn as nn
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Generator(nn.Module):
    def __init__(self, num_input):
        super(Generator, self).__init__()
        self.Net = nn.Sequential(
            nn.Linear(num_input, 256), # [bz, 1, 100] -> [bz, 1, 256]
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28), # [bz, 1, 1024] -> [bz, 1, 28 * 28]
            nn.Sigmoid() # Image intensities are in [0, 1]
        )

    def forward(self, input):
        output = Flatten()(input)
        output = torch.unsqueeze(output, dim=1)
        output = self.Net(output)
        output = output.view(input.size(0), 1, 28, 28)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Net = nn.Sequential(
            nn.Linear(1, 64), # [bz, 1, 28, 28] -> [bz, 28 * 28, 1] -> [bz, 28* 28, 64]
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 1), # [bz, 28* 28, 1]
            nn.Dropout(0.1),
            Flatten(), # [bz, 28 * 28] 2D tensor
            nn.Linear(784, 1), #[bz, 1]
            nn.Sigmoid()
        )

    def forward(self, input):
        input = torch.unsqueeze(Flatten()(input), dim=-1)
        output = self.Net(input)
        return output
