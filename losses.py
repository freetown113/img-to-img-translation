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
    """Class used to calculate Discriminator loss used also for the part of Generator loss calculation"""
    def __init__(self, device):
        self.device = device

    def discrim_loss(self, input, discriminator):
        """Helper function to divide loss calculation logic"""
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.d_loss(pred_i, discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.d_loss(input, discriminator)

    def d_loss(self, image, discriminator):
        """Compute output for given input based on statement rather it used for Generator or Discriminator loss"""
        def get_zero_tensor(input):
            zero_tensor = torch.FloatTensor(1).fill_(0).to(self.device)
            zero_tensor.requires_grad_(False)
            return zero_tensor.expand_as(input)
        if discriminator:
            minval = torch.min(image - 1, get_zero_tensor(image))
            loss = -torch.mean(minval)
        else:
            loss = -torch.mean(image)
        return loss

    def divide_pred(self, pred):
        fake = []
        real = []
        for p in pred:
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])

        return fake, real
