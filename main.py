from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
from Data_loader import train_loader, test_loader
import torch
from GAN_Model import Generator, Discriminator
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dim = 100
discriminator = Discriminator().to(device)
generator = Generator(latent_dim).to(device)

loss = torch.nn.BCELoss()
print("Using device:", device)

generator_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))
tmp_img = "tmp_gan_out.png"
discriminator_loss, generator_loss = [], []

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
        z = torch.randn(batch_size, latent_dim, 1, 1)
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

    # -- Plotting --
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

    rows, columns = 7, 7

    # Generate data
    with torch.no_grad():
        z = torch.randn(batch_size, 1, latent_dim)
        z = Variable(z, requires_grad=False).to(device)
        x_fake = generator(z)

    canvas = np.zeros((28 * rows, columns * 28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x_fake.cpu().data[idx]
    ax.imshow(canvas, cmap='gray')

    plt.savefig(tmp_img)
    plt.close(f)
    display(Image(filename=tmp_img))
    clear_output(wait=True)

    os.remove(tmp_img)