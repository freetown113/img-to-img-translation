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
        self.lr = arguments.lerning_rate
        self.betas = arguments.betas
        self.save_images = arguments.save_images
        self.visual = visualisation
        self.block_loader = BlocksLoader(arguments, self.device)
        self.min_gen_loss = math.inf
        check_dirs(arguments)

    def fit(self):
        #  Define neural networks models
        networks = self.block_loader.get_all_networks()

        #  Load weights if requested
        if self.args.continue_from_pretrained:
            load_params(self.args, networks)

        #  Split neural network model into blocks
        generator = networks['generator']
        discriminator = networks['discriminator']
        encoder = networks['encoder']
        decoder = networks['decoder']
        vgg = networks['vgg']

        #  Define opimizers with parameters from corresponding models
        optimizer_generator = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) +
                                               list(generator.parameters()), lr=self.lr * 2, betas=self.betas)
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=self.lr / 2, betas=self.betas)

        #  Define losses
        d_loss = DiscriminatorLoss(self.device)
        loss_vgg = VGGLoss(vgg)
        L1_loss = nn.L1Loss()

        # Describe training loop
        for epoch in range(self.args.epoches):
            for n, (real_samples, target_samples) in enumerate(self.data_loaders['train']):

                #  Copy batch of data to device
                real_samples = real_samples.to(self.device)
                target_samples = target_samples.to(self.device)

                # Training the discriminator
                optimizer_discriminator.zero_grad()
                latent_space_samples = torch.randn((self.bs, self.latent)).to(self.device)
                with torch.no_grad():
                    vec = generator.style(real_samples, target_samples, latent_space_samples)
                    generated_samples = decoder(vec)

                fake_and_real = torch.cat([generated_samples, target_samples], dim=0)
                discriminator_out = discriminator(fake_and_real)
                pred_fake, pred_real = d_loss.divide_pred(discriminator_out)

                loss_real = d_loss.discrim_loss(pred_real, True)
                loss_fake = d_loss.discrim_loss(pred_fake, True)
                loss_discriminator = loss_real + loss_fake

                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Training the generator
                optimizer_generator.zero_grad()
                latent, _, _ = encoder(real_samples)
                style = generator.style(real_samples, target_samples, latent)
                fake_img = decoder(style)

                fake_and_real = torch.cat([fake_img, target_samples], dim=0)

                with torch.no_grad():
                    discriminator_out = discriminator(fake_and_real)

                pred_fake, pred_real = d_loss.divide_pred(discriminator_out)

                loss_fake = d_loss.discrim_loss(pred_fake, False)

                num_D = len(pred_fake)
                GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
                for i in range(num_D):
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):
                        unweighted_loss = L1_loss(pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * 1.0 / num_D

                vgg_loss = loss_vgg(fake_img, target_samples)

                vae_loss = loss_fake + vgg_loss + GAN_Feat_loss

                vae_loss.backward()
                optimizer_generator.step()

                if self.save_images:
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
                if epoch % 10 == 0 and epoch > 0:
                    for n, (real_samples, target_samples) in enumerate(self.data_loaders['test']):
                        #  Copy batch of data to device
                        real_samples = real_samples.to(self.device)
                        target_samples = target_samples.to(self.device)

                        # Testing the discriminator
                        latent_space_samples = torch.randn((self.bs, self.latent)).to(self.device)
                        with torch.no_grad():
                            vec = generator.style(real_samples, target_samples, latent_space_samples)
                            generated_samples = decoder(vec)

                        fake_and_real = torch.cat([generated_samples, target_samples], dim=0)
                        discriminator_out = discriminator(fake_and_real)
                        pred_fake, pred_real = d_loss.divide_pred(discriminator_out)

                        loss_real = d_loss.discrim_loss(pred_real, True)
                        loss_fake = d_loss.discrim_loss(pred_fake, True)
                        loss_discriminator = loss_real + loss_fake

                        # Testing the generator
                        latent, _, _ = encoder(real_samples)
                        style = generator.style(real_samples, target_samples, latent)
                        fake_img = decoder(style)

                        fake_and_real = torch.cat([fake_img, target_samples], dim=0)

                        with torch.no_grad():
                            discriminator_out = discriminator(fake_and_real)

                        pred_fake, pred_real = d_loss.divide_pred(discriminator_out)

                        loss_fake = d_loss.discrim_loss(pred_fake, False)

                        num_D = len(pred_fake)
                        GAN_Feat_loss = torch.FloatTensor(1).fill_(0).to(self.device)
                        for i in range(num_D):
                            num_intermediate_outputs = len(pred_fake[i]) - 1
                            for j in range(num_intermediate_outputs):
                                unweighted_loss = L1_loss(pred_fake[i][j], pred_real[i][j].detach())
                                GAN_Feat_loss += unweighted_loss * 1.0 / num_D

                        vgg_loss = loss_vgg(fake_img, target_samples)

                        vae_loss = loss_fake + vgg_loss + GAN_Feat_loss

                    print(f'Test in epoch {epoch} generator loss is {vae_loss.detach().cpu().item()}, '
                          f'descriminator_loss is {loss_discriminator.detach().cpu().item()}')
