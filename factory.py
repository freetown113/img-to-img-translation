from typing import Callable, Any
from blocks.base import BaseNetwork

network_creator: dict(str=Callable[..., BaseNetwork]) = {}


def register(neural_network_name: str, creation_function: Callable[..., BaseNetwork]):
    """Register a new neural network type"""
    network_creator[neural_network_name] = creation_function


def remove(neural_network_name: str):
    """Remove a NN from list of registered"""
    network_creator.pop(neural_network_name, 'There is no such NN')


def create(arguments: dict(str=Any)) -> BaseNetwork:
    """Create NN of a specific type given a dictionary of arguments"""
    args_copy = arguments.copy()
    network_type, parameters = args_copy.values()
    try:
        creator = network_creator[network_type]
        return creator(*parameters)
    except KeyError:
        raise ValueError(f'Unknown network type was provided: {network_type}') from None
