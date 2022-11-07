from GAN_Model import Generator, Discriminator


def train(num_input, num_output):
    G = Generator(num_input, num_output)
    D = Discriminator()