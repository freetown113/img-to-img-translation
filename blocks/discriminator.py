import numpy as np
import torch.nn as nn
import torch.nn.utils.spectral_norm as spec_norm

from blocks.base import BaseNetwork
import factory
from blocks.vit import SimpleViT


def get_norm_layer():

    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        layer = spec_norm(layer)

        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, arguments):
        super().__init__()

        for i in range(2):
            subnetD = NLayerDiscriminator(arguments)
            self.add_module('discrim_layer_%d' % i, subnetD)

    def downsample(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        get_intermediate_features = not False
        for D in self.children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


class NLayerDiscriminator(BaseNetwork):
    def __init__(self, arguments):
        super().__init__()
        self.nf = arguments['nf']
        norm_layer = get_norm_layer()
        sequence = [[nn.Conv2d(3, self.nf, kernel_size=4, stride=2, padding=2),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, 3):
            nf_prev = self.nf
            self.nf = min(self.nf * 2, 512)
            sequence += [[norm_layer(nn.Conv2d(nf_prev, self.nf, kernel_size=4,
                                               stride=2, padding=2)),
                          nn.LeakyReLU(0.2, False)]]

        sequence += [[nn.Conv2d(self.nf, 1, kernel_size=4, stride=1, padding=2)]]

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        return results[1:]


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
    factory.register(MultiscaleDiscriminator.__name__, MultiscaleDiscriminator)
