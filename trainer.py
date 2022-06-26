import math
import torch
import torch.nn as nn
from typing import Dict, Type

from network import BlocksLoader
from saver import ImageSaver
from losses import DiscriminatorLoss, VGGLoss
from utils import save_params, load_params, check_dirs


class VDTrainer:
    def __init__(
            self,
            arguments: Type,
            data_loaders: Dict,
            visualisation: ImageSaver):
        self.args = arguments
        self.bs = arguments.batch_size
        self.latent = arguments.latent_dim
        self.device = torch.device('cuda:' + arguments.device if torch.cuda.is_available() else 'cpu')
        self.data_loaders = data_loaders
        self.visual = visualisation
        self.block_loader = BlocksLoader(arguments, self.device)
        self.min_gen_loss = math.inf
        check_dirs(arguments)

    def fit(self):
        #  Define neural networks models
        networks = self.block_loader.get_all_networks()

        if self.args.continue_from_pretrained:
            load_params(self.args, networks)

        generator = networks['generator']
        discriminator = networks['discriminator']
        encoder = networks['encoder']
        decoder = networks['decoder']
        vgg = networks['vgg']

        #  Define opimizers with parameters from corresponding models
        optimizer_generator = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) +
                                               list(generator.parameters()), lr=0.0004, betas=(0, 0.9))
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

        #  Define losses
        discrim_loss = DiscriminatorLoss()
        loss_vgg = VGGLoss(vgg)
        L1_loss = nn.L1Loss()

        for epoch in range(self.args.epoches):
            for n, (real_samples, target_samples) in enumerate(self.data_loaders['train']):

                real_samples = real_samples.to(self.device)
                target_samples = target_samples.to(self.device)
                real_samples_labels = torch.ones((self.bs, 1)).to(self.device)
                latent_space_samples = torch.randn((self.bs, self.latent)).to(self.device)
                with torch.no_grad():
                    vec = generator.style(real_samples, target_samples, latent_space_samples)
                    generated_samples = decoder(vec)

                pred_real = discriminator(real_samples)
                pred_fake = discriminator(generated_samples)
                loss_real = discrim_loss(pred_real, self.device)
                loss_fake = discrim_loss(pred_fake, self.device)
                loss_discriminator = loss_real + loss_fake

                # Training the discriminator
                optimizer_discriminator.zero_grad()
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Training the generator
                optimizer_generator.zero_grad()
                latent, z_mean, z_log_var = encoder(real_samples)
                style = generator.style(real_samples, target_samples, latent)
                fake_img = decoder(style)

                with torch.no_grad():
                    output_discriminator_generated = discriminator(fake_img)

                kld_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp(), dim=1), dim=0)
                descriminator_loss = -torch.mean(fake_img)
                loss_generator = L1_loss(output_discriminator_generated, real_samples_labels)
                vgg_loss = loss_vgg(fake_img, target_samples)
                vae_loss = loss_generator + kld_loss + descriminator_loss + vgg_loss

                vae_loss.backward()
                optimizer_generator.step()

                self.visual.save_data(real_samples, fake_img, target_samples)

                if self.min_gen_loss > vae_loss.detach().cpu().item():
                    self.min_gen_loss = vae_loss.detach().cpu().item()
                    save_params(self.args,
                                {'generator': generator,
                                 'discriminator': discriminator,
                                 'encoder': encoder,
                                 'decoder': decoder,
                                 'optimizer_generator': optimizer_generator,
                                 'optimizer_discriminator': optimizer_discriminator})

            print(f'Train in epoch {epoch} generator loss is {vae_loss.detach().cpu().item()}, '
                  f'descriminator_loss is {loss_discriminator.detach().cpu().item()}')

            if self.args.test_while_training:
                for n, (real_samples, target_samples) in enumerate(self.data_loaders['test']):
                    real_samples = real_samples.to(self.device)
                    target_samples = target_samples.to(self.device)

                    latent, z_mean, z_log_var = encoder(real_samples)
                    style = generator.style(real_samples, target_samples, latent)
                    fake_img = decoder(style)

                    with torch.no_grad():
                        output_discriminator_generated = discriminator(fake_img)

                    kld_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp(), dim=1), dim=0)
                    descriminator_loss = -torch.mean(fake_img)
                    loss_generator = L1_loss(output_discriminator_generated, real_samples_labels)
                    vgg_loss = loss_vgg(fake_img, target_samples)
                    vae_loss = loss_generator + kld_loss + descriminator_loss + vgg_loss

                print(f'Test loss in epoch {epoch} is {vae_loss.detach().cpu().item()}')
