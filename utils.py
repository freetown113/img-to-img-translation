import os
import torch
from typing import Type, Dict, Callable


def save_params(
        arguments: Type,
        dictionary: Dict[str, Callable]) -> None:
    """Function used to save parameters of the objects heritated from nn.Module"""
    for name, value in dictionary.items():
        torch.save(value.state_dict(), os.path.join(arguments.project_location,
                                                    arguments.save_weights_path, name + '.pth'))


def load_params(
        arguments: Type,
        models: Dict[str, Callable]) -> None:
    """Function used to load parameters to the objects heritated from nn.Module"""
    for name, model in models.items():
        try:
            model.load_state_dict(torch.load(os.path.join(arguments.project_location, 
                                                         arguments.save_weights_path, name + '.pth')))
        except:
            print(f'Weights for model {name} was not found! (If it\'s vgg don\'t pay attention to this warning)')


def check_dirs(arguments: Type) -> None:
    """Function used to check if folders from config file with 'path' in name exist and create it if not"""
    for el in arguments.__dict__:
        if 'path' in el.split('_'):
            if not os.path.isdir(os.path.join(arguments.project_location, getattr(arguments, el))):
                os.makedirs(os.path.join(arguments.project_location, getattr(arguments, el)))


def check_dir(path: str) -> None:
    """Function used to check if folder path frovided in string argument exist"""
    if not os.path.isdir(path):
        os.makedirs(path)
