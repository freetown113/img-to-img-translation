import torch
from typing import Callable, Dict, Type

import factory
import loader


class BlocksLoader:
    def __init__(
            self,
            parameters: Type,
            device: torch.device):
        """
        Class that load an build networks provided from config file
        """
        self.parameters = parameters
        self.device = device
        self.networks = None
        self.load()

    def load(self) -> None:
        loader.load_plugins(self.parameters.plugins)
        networks = {name.split('.')[1]: factory.create(block).to(self.device) for block, name in
                    zip(self.parameters.parameters, self.parameters.plugins)}
        self.networks = networks

    def build_network(
            self,
            type: str) -> Callable:
        """Build a single model provided by the type argument"""
        return self.networks[type]

    def get_all_networks(self) -> Dict[str, Callable]:
        """Build all models proveded in config file"""
        return self.networks
