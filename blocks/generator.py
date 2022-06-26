import torch
import torch.nn as nn

from blocks.base import BaseNetwork
import factory
from blocks.vit import SimpleViT


class Generator(BaseNetwork):
    """
    A class that describes Generator NN model. Generator creates image tensor based on
    features of original image, target image and on the latent space.
    """
    def __init__(self, arguments):
        super().__init__()
        self.l_dim = arguments['latent_dim']
        self.vit_orig = SimpleViT(
                        image_size=256,
                        patch_size=32,
                        num_classes=512,
                        dim=1024,
                        depth=6,
                        heads=16,
                        mlp_dim=2048)
        self.vit_style = SimpleViT(
                        image_size=256,
                        patch_size=32,
                        num_classes=512,
                        dim=1024,
                        depth=6,
                        heads=16,
                        mlp_dim=2048)
        self.linear4 = nn.Linear(self.l_dim * 3, self.l_dim)

    def style(self, original, style, latent):
        original = self.vit_orig(original)
        style = self.vit_style(style)
        z = torch.cat([original, style, latent], dim=1)
        return nn.functional.leaky_relu(self.linear4(z), 2e-1)


def initialize() -> None:
    factory.register(Generator.__name__, Generator)
