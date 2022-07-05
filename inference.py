import argparse
import torch

from config.config import Config
from dataset import get_data_loader
from network import BlocksLoader
from saver import ImageSaver
from utils import load_params, check_dirs


def launch(config_location):
    config = Config(config_location.config)
    arguments = config.generate_config()

    torch.manual_seed(arguments.seed)

    data_loader = get_data_loader(arguments, 'test')

    visualisation = ImageSaver(arguments)
    check_dirs(arguments)

    device = torch.device('cuda:' + arguments.device if torch.cuda.is_available() else 'cpu')

    block_loader = BlocksLoader(arguments, device)

    #  Define neural networks models
    networks = {net_type: block_loader.build_network(net_type) for net_type in ['generator', 'decoder']}

    #  Load weights
    load_params(arguments, networks)

    #  Split neural network model into blocks
    generator = networks['generator']
    decoder = networks['decoder']

    for n, (real_samples, target_samples) in enumerate(data_loader):

        latent_space_samples = torch.randn((arguments.batch_size, arguments.latent_dim)).to(device)
        real_samples = real_samples.to(device)
        target_samples = target_samples.to(device)

        with torch.no_grad():
            vec = generator.style(real_samples, target_samples, latent_space_samples)
            generated_samples = decoder(vec)

        visualisation.save_data(real_samples, generated_samples, target_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='/img-to-img-translation/config/parameters.yaml',
                        help='Path to the json file with parameters')

    args = parser.parse_args()
    launch(args)
