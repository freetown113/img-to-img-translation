import numpy as np
import os
from PIL import Image
import torch
from typing import List, Union


class ImageSaver:
    """
    A class that serves to save images.

    Attributes
    ----------
    self.path : str
        path to directory where saved images will be located.

    Methods
    -------
    save_data:
        Get data from out the outside in several possible types, chech the types and path single
        batch tensors(images to be saved) to save_image method.
    save_image:
        Get three tensors as separate images of the same size, concatinate them horizontally and
        write three in one as a single image in the directory provided by self.path.
    """
    def __init__(self, arguments):
        self.path = os.path.join(arguments.project_location, arguments.result_img_path)

    def save_data(
            self,
            real: Union[torch.Tensor, List],
            predict: Union[torch.Tensor, List],
            target: Union[torch.Tensor, List],
            idx=None) -> None:

        if type(real) != type(predict) != type(target):
            raise TypeError(f'Some or all inputs are of different type: first_arg is {type(real)}, '
                            f'second_arg is {type(predict)}, third_arg is {type(target)}')

        if isinstance(real, List):
            if len(real) != len(predict) != len(target):
                raise TypeError(f'Some or all inputs are of different length: first_arg is {len(real)},'
                                f'second_arg is {len(predict)}, third_arg is {len(target)}')
            else:
                for element in zip(real, predict, target):
                    self.check_data(element)

        if isinstance(real, torch.Tensor):
            if real.shape != predict.shape != target.shape:
                raise TypeError(f'Some or all inputs are of different shape: first_arg is {real.shape},'
                                f'second_arg is {predict.shape}, third_arg is {target.shape}')
            else:
                if len(real.shape) == 4 and real.shape[0] > 1:
                    for idx in range(real.shape[0]):
                        self.save_data(real[idx, ...], predict[idx, ...], target[idx, ...], idx)
                else:
                    meanstd = torch.ones((256, 256, 3)).to(real.device) * 0.5
                    real = (real.permute((1, 2, 0)).detach() * meanstd + meanstd).cpu() * 255.
                    predict = (predict.permute((1, 2, 0)).detach() * meanstd + meanstd).cpu() * 255.
                    target = (target.permute((1, 2, 0)).detach() * meanstd + meanstd).cpu() * 255.

                    self.save_image(real, predict, target, 0 if idx is None else idx)

    def save_image(
            self,
            real: torch.Tensor,
            predict: torch.Tensor,
            output: torch.Tensor,
            image_index: int) -> None:
        """Save single batch of three images concatenated in one"""
        united = torch.cat([real, predict, output], dim=1)
        united = united.numpy()
        Image.fromarray(united.astype(np.uint8)).save(os.path.join(self.path, str(image_index) + '.png'))
