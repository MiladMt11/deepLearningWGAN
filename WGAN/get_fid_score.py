import torch
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