from torch.autograd import Variable
from Dataset.CIFAR_dataloader import train_loader, test_loader
import torch
from Models.DCGAN import Generator, Discriminator
import numpy as np
import matplotlib.image
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dim = 100
discriminator = Discriminator(3).to(device)
generator = Generator(latent_dim, 3).to(device)

loss = torch.nn.BCELoss()
print("Using device:", device)

generator_optim = torch.optim.Adam(generator.parameters(), 2e-4, betas=(0.5, 0.999))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), 2e-4, betas=(0.5, 0.999))
tmp_img = "tmp_gan_out.png"
discriminator_loss, generator_loss = [], []

num_epochs = 1000
for epoch in range(num_epochs):
    batch_d_loss, batch_g_loss = [], []
    generator.train()
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
    writer.add_scalar('Discreminator Loss true/train', error_true, epoch)
    writer.add_scalar('Discreminator Loss fake/train', error_fake, epoch)
    writer.add_scalar('Generator Loss/train', np.mean(batch_g_loss), epoch)

    # writer.add_image("test", matplotlib.image.imread('Results/img_generatori_iter_{}.png'.format(epoch)), 2, dataformats='HWC')
    if epoch % 100 == 0:
        z = torch.randn(1, 100, 1, 1).to(device)
        generator.eval()
        fake_image = generator(z)
        samples = fake_image.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'Results/img_generatori_iter_{}.png'.format(epoch))
        torch.save(generator.state_dict(), "checkpoint/G.pth")
        torch.save(discriminator.state_dict(), "checkpoint/D.pth")
        print('model saved!')