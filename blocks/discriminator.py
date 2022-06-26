import torch.nn as nn

from blocks.base import BaseNetwork
import factory
from blocks.vit import SimpleViT


class Discriminator(BaseNetwork):
    """
    A class that describes Discriminator NN model. Discriminator try to evaluate input image
    and assign a probabiliy of it to be real or created by generator.
    """
    def __init__(self):
        super().__init__()
        self.vit = SimpleViT(
                        image_size=256,
                        patch_size=32,
                        num_classes=1024,
                        dim=1024,
                        depth=6,
                        heads=16,
                        mlp_dim=2048)
        self.outp = nn.Linear(1024, 1)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        x = self.vit(x)
        x = self.outp(x)
        return self.prob(x)


def initialize() -> None:
    factory.register(Discriminator.__name__, Discriminator)
