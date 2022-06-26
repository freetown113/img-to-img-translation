import torch.nn as nn
import torchvision

import factory


class VGG(nn.Module):
    """VGG neural network architecture with pretrained parameters used to compute vgg loss function"""
    def __init__(self, requires_grad=False):
        super().__init__()
        self.outs = [1, 6, 11, 20, 29]
        self.vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Pass input tensor and collect output of layers described in outs attribute"""
        outputs = list()
        for idx, feat in enumerate(self.vgg_pretrained_features):
            x = feat(x)
            if idx in self.outs:
                outputs.append(x)
        return outputs


def initialize() -> None:
    factory.register(VGG.__name__, VGG)
