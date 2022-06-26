from ast import arg
import math
import torch
import torch.nn as nn

from blocks.base import BaseNetwork
import factory


def sampling(args):
    z_mean, z_log_var = args
    std = torch.exp(0.5 * z_log_var)
    eps = torch.randn_like(std)
    return eps * std + z_mean


class Encoder(BaseNetwork):
    """A class that describes Encoder NN model. Encoder create latent space for Generator."""
    def __init__(self, arguments):
        super().__init__()
        self.arguments = arguments
        self.ch = arguments['enc_channels']
        self.bs = arguments['batch_size']

        if arguments['test_while_training']:
            self.bs = self.bs // 2

        self.l_dim = arguments['latent_dim']
        self.enc_r = arguments['encode_ratio']
        self.edc = arguments['enc_dec_channels']
        self.im_size = arguments['img_size']

        shape = (self.bs, self.edc,
                 int(math.ceil(self.im_size / self.enc_r)),
                 int(math.ceil(float(self.im_size) / self.enc_r)))

        self.conv1 = nn.Conv2d(3, self.ch, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.ch, self.ch * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.ch * 2, self.ch * 2, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(self.ch * 2, self.ch * 4, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(shape[1] * shape[2] * shape[3], self.ch * 16)
        self.linear2 = nn.Linear(self.ch * 16, self.l_dim)
        self.linear3 = nn.Linear(self.ch * 16, self.l_dim)
        self.sample = sampling

    def forward(self, image):
        x = nn.functional.relu(self.conv1(image))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = x.reshape((self.bs, -1))
        z = nn.functional.relu(self.linear1(x))

        z_mean = self.linear2(z)
        z_log_var = self.linear3(z)
        return self.sample((z_mean, z_log_var)), z_mean, z_log_var


def initialize() -> None:
    factory.register(Encoder.__name__, Encoder)
