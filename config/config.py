from typing import Type
import yaml


class Config:
    """
    A class that serves to generate python object containing parameters from json.

    Attributes
    ----------
    self.path : str
        path to json file with parameters.

    Methods
    -------
    load_args:
        load json file and form a dict object from it
    generate_config:
        create new class with attributes and values corresponding to config parameters
    """
    def __init__(
            self,
            path: str):
        self.path = path
        self.object = None

    def load_yaml(self):
        """Function loads comfiguration parameters from provided yaml config file"""
        with open(self.path, "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
                general, plug, param = parameters.items()
                self.object = dict(general[1])
                [self.object.update({el[0]:el[1]}) for el in (plug, param)]
            except yaml.YAMLError as exc:
                print(f'Following exception occured while opening config file: {exc}')

    def generate_config(self) -> Type:
        """Function creates class Configurator that contains all parameters from config file"""
        self.load_yaml()
        obj = type('Configurator', (), {**self.object})

        if obj.test_while_training:
            obj.batch_size = obj.batch_size // 2

        return obj
