import math
import torch.nn as nn

from blocks.base import BaseNetwork
import factory


class Decoder(BaseNetwork):
    """A class that describes Decoder NN model."""
    def __init__(self, arguments):
        super().__init__()
        self.ch = arguments['enc_channels']
        self.bs = arguments['batch_size']

        if arguments['test_while_training']:
            self.bs = self.bs // 2

        self.l_dim = arguments['latent_dim']
        self.enc_r = arguments['encode_ratio']
        self.edc = arguments['enc_dec_channels']
        self.im_size = arguments['img_size']
        self.shape = (
                    self.bs, self.edc,
                    int(math.ceil(self.im_size / self.enc_r)),
                    int(math.ceil(float(self.im_size) / self.enc_r)))
        out_shape = self.shape[1] * self.shape[2] * self.shape[3]
        self.linear = nn.Linear(self.l_dim, out_shape)
        self.deconv1 = nn.ConvTranspose2d(self.shape[1], self.ch * 4, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.ch * 4, self.ch * 2, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.ch * 2, self.ch * 2, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(self.ch * 2, self.ch, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(self.ch, 3, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.linear(input)
        x = x.reshape((-1, self.shape[1], self.shape[2], self.shape[3]))
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        x = nn.functional.relu(self.deconv4(x))
        return nn.functional.tanh(self.deconv5(x))


def initialize() -> None:
    factory.register(Decoder.__name__, Decoder)
