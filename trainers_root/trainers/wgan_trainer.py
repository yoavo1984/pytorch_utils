# Author : Yoav Orlev.

import torch
# TODO return d/g loss on every train iteration.


class WganParams(object):
    """
    This class plays the role of containing the relevant parameters for the WganTrainer class.
    """
    def __init__(self, lr=0.00005, clipping=0.01, batch_size=64, n_critic=5, latent_dim=100):
        if lr:
            self.lr = lr
        if clipping:
            self.clipping = clipping
        if batch_size:
            self.batch_size = batch_size  # This is obsolete as the dataloader will decide the batch size.
        if n_critic:
            self.n_critic = n_critic
        if latent_dim:
            self.latent_dim = latent_dim


class WganTrainer(object):
    """
    Use this class to train a generator and discriminator according to the wgan procedure.
    Implementation follows the algorithm described in :
    https://arxiv.org/abs/1701.07875 [Wasserstein GAN] by Arjovsky et al.
    """

    # If cuda is available we use it by default.
    is_cuda = True if torch.cuda.is_available() else False

    def __init__(self, data_loader, discriminator, generator, parameters=False):
        """

        :param data_loader: pytorch DataLoader object with the dataset you wish to train on.
        :param discriminator: pytorch nn.Module object representing the discriminator.
        :param generator: pytorch nn.Module object representing the generator.
        :param parameters: WganParams object.
        """
        if self.is_cuda:
            discriminator.cuda()
            generator.cuda()

        # Use supplied parameters or default if none was supplied.
        self.params = parameters if parameters else WganParams()
        self.params.batch_size = data_loader.batch_size

        self.discriminator = discriminator
        self.generator = generator
        self.data_loader = data_loader

        # Defining and initializing discriminator and generator optimizers.
        self.d_optimizer = 0
        self.g_optimizer = 0
        self._init_optimizers()

    def _init_optimizers(self):
        self.d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(),
                                               lr=self.params.lr)
        self.g_optimizer = torch.optim.RMSprop(self.generator.parameters(),
                                               lr=self.params.lr)

    def _sample_generator(self, detach=False):
        z = torch.randn(self.params.batch_size, self.params.latent_dim)

        if self.is_cuda:
            z = z.cuda()

        if detach:
            return self.generator(z).detach()
        else:
            return self.generator(z)

    def train(self):
        """
        One iteration of the wgan training procedure. *One iteration goes over the entire dataset*.
        :return: The loss at the current iteration.
        """
        d_loss = []
        g_loss = []
        for index, (real, _) in enumerate(self.data_loader):
            d_loss.append(self._train_discriminator(real))

            # Every n_critic batches train the generator.
            if index % self.params.n_critic == 0:
                g_loss.append((self._train_generator()))

        return d_loss, g_loss

    def _train_discriminator(self, real_batch):
        self.d_optimizer.zero_grad()

        # Generate fake instances
        fake_batch = self._sample_generator(detach=True)

        # Cudaing
        if self.is_cuda:
            real_batch = real_batch.cuda()
            fake_batch = fake_batch.cuda()

        # Getting Discriminator loss
        loss = -1 * (torch.mean(self.discriminator(real_batch)) -
                     torch.mean(self.discriminator(fake_batch)))

        # Back prop
        loss.backward()
        self.d_optimizer.step()

        # Weight clipping
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.params.clipping, self.params.clipping)

        return loss.data

    def _train_generator(self):
        self.g_optimizer.zero_grad()

        fake_batch = self._sample_generator()
        loss = -1 * torch.mean(self.discriminator(fake_batch))
        loss.backward()

        self.g_optimizer.step()

        return loss.data

    def set_cuda(self, is_cuda):
        """
        Force cuda setting to a given value.
        :param is_cuda: boolen value to set to is_cuda.
        """
        self.is_cuda = is_cuda

