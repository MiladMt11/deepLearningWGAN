import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_real_data():
    data = []
    for i in range(1024):
        data.append(np.random.randn() + 3.0)
    return np.array(data, dtype=float)

batch_size = 64
data = generate_real_data()
data = data.astype(np.float32)
data = np.reshape(data, (1024, 1))
data = torch.from_numpy(data).to(device)
data = TensorDataset(data, data)
data = DataLoader(data, batch_size=batch_size, shuffle=True)
train_loader = data
latent_dim = 100
generator = nn.Sequential(
    nn.Linear(latent_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.ReLU()
).to(device)

discriminator = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
).to(device)

loss = torch.nn.BCELoss()
discriminator_loss, generator_loss = [], []

generator_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))

num_epochs = 50
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []

    for x, _ in train_loader:
        batch_size = x.size(0)
        # True data is given label 1, while fake data is given label 0
        true_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)

        discriminator.zero_grad()
        generator.zero_grad()

        # Step 1. Send real data through discriminator
        #         and backpropagate its errors.
        x_true = Variable(x).to(device)
        output = discriminator(x_true)

        error_true = loss(output, true_label)
        error_true.backward()

        # Step 2. Generate fake data G(z), where z ~ N(0, 1)
        #         is a latent code.
        z = torch.randn(batch_size, latent_dim)
        z = Variable(z, requires_grad=False).to(device)
        x_fake = generator(z)

        # Step 3. Send fake data through discriminator
        #         propagate error and update D weights.
        # --------------------------------------------
        # Note: detach() is used to avoid compounding generator gradients
        output = discriminator(x_fake.detach())

        error_fake = loss(output, fake_label)
        error_fake.backward()
        discriminator_optim.step()

        # Step 4. Send fake data through discriminator _again_
        #         propagate the error of the generator and
        #         update G weights.
        output = discriminator(x_fake)

        error_generator = loss(output, true_label)
        error_generator.backward()
        generator_optim.step()

        batch_d_loss.append((error_true / (error_true + error_fake)).item())
        batch_g_loss.append(error_generator.item())

    discriminator_loss.append(np.mean(batch_d_loss))
    generator_loss.append(np.mean(batch_g_loss))

    f, axarr = plt.subplots(1, 2, figsize=(18, 7))

    # Loss
    ax = axarr[0]
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.plot(np.arange(epoch + 1), discriminator_loss)
    ax.plot(np.arange(epoch + 1), generator_loss, linestyle="--")
    ax.legend(['Discriminator', 'Generator'])

    # Latent space samples
    ax = axarr[1]
    ax.set_title('Samples from generator')
    ax.axis('off')

    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim)
        z = Variable(z, requires_grad=False).to(device)
        x_fake = generator(z)

