import torch
import torch.nn as nn
from typing import Callable


class VGGLoss(nn.Module):
    """Loss function calculated based on VGG model"""
    def __init__(
            self,
            vgg: Callable):
        super().__init__()
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class DiscriminatorLoss(nn.Module):
    """Loss function used to calculate a part of generator loss"""
    def forward(self, image, device):

        def get_zero_tensor(input: torch.Tensor):
            zero_tensor = torch.FloatTensor(1).fill_(0).to(device)
            zero_tensor.requires_grad_(False)
            return zero_tensor.expand_as(input)

        minval = torch.min(image - 1, get_zero_tensor(image))
        loss = -torch.mean(minval)
        return loss
