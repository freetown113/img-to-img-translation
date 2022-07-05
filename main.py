import argparse
import torch

from config.config import Config
from dataset import get_data_loader
from saver import ImageSaver
from trainer import VDTrainer


def main(config_location):
    config = Config(config_location.config)
    arguments = config.generate_config()

    torch.manual_seed(arguments.seed)

    visualisation = ImageSaver(arguments)

    data_loaders = {item: get_data_loader(arguments, item) for item in ['train', 'test']}

    trainer = VDTrainer(
        arguments,
        data_loaders,
        visualisation
    )

    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='/img-to-img-translation/config/parameters.yaml',
                        help='Path to the json file with parameters')

    args = parser.parse_args()
    main(args)
