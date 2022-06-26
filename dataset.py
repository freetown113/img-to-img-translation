import itertools
import numpy as np
import os
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import Iterator
import zipfile

from utils import check_dir


class VDDataset:
    """
    A class that serves to describe and build dataset.

    Attributes
    ----------
    self.train : bool
        a flag that define is dataset is designated for the training or for the testing process
    self.transform :
        an instance of torchvision.transforms class that provides a collection of
        transformations to be applied to images while training
    self.target_transform
        an instance of torchvision.transforms class that provides a collection of
        transformations to be applied to images while testing
    self.data_archive : str
        location of the archive with images
    self.dataset_name : str
        name of the dataset
    self.location : str
        the project's directory
    self.dataset_dir : str
        path to the dataset, respectively to the project
    self.data_ids : Set
        set with unique names of unpared images
    self.data : Dict
        Dictionary with path to images divided into ground truth images and targets
    self.data_size : int
        the numder of unpaired images in the dataset
    download : bool
        a flag that shows if we want explicitly download zip archive with images
        or we already have data in our local directory

    Methods
    -------
    get_data:
        walk through image directory and collect information in self.data dictionary
    download_data:
        download zipped data by given URL
    unzip_data
        unzip data to a specific location
    __len__
        return whole given dataset's size
    __getitem__
        return a pair ground truth image and target image
    """
    def __init__(
            self,
            path: str,
            train: bool = True,
            transform: transforms.Compose = None,
            target_transform: transforms.Compose = None,
            download: bool = True):
        """set all parameters to prepare and build dataset"""
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_archive = 'VD_dataset.zip'
        self.dataset_name = 'VD_dataset'
        self.location = os.path.dirname(path)
        self.dataset_dir = path
        self.data_ids = set()
        self.data = {'images': list(), 'targets': list()}
        self.data_size = None
        check_dir(os.path.join(self.dataset_dir, self.dataset_name))
        if download:
            self.download_data()
        self.get_data()

    def get_data(self) -> None:
        """Collect path to images to the dict according to GT/ Target image separation"""
        for root, dirs, files in os.walk(self.dataset_dir):
            if os.path.basename(root) == self.dataset_name:
                for idx, file in enumerate(files):
                    name, ext = os.path.splitext(file)
                    id = name.split('_')[0]
                    self.data_ids.add(id)
                    self.data_size = idx + 1

        if self.data_size != len(self.data_ids) * 2:
            raise AssertionError(f'Your dataset is potentially unpaired!')

        if self.train:
            [self.data_ids.pop() for i in range(int(len(self.data_ids)*0.1))]
        else:
            data_ids = set()
            [data_ids.add(self.data_ids.pop()) for i in range(int(len(self.data_ids)*0.1))]
            self.data_ids = data_ids

        [self.data['images'].append(os.path.join(self.dataset_dir, self.dataset_name, img + '_input.png'))
         for img in self.data_ids]
        [self.data['targets'].append(os.path.join(self.dataset_dir, self.dataset_name, img + '_target.png'))
         for img in self.data_ids]

    def download_data(self) -> None:
        """Download ZIP archive contatining images by URL"""
        # if os.path.exists(os.path.join(self.dataset_dir, self.data_archive)):
        #     return
        url = 'https://visidoncloud.com/s/gbzTSrg7NxBm6qc/download/VD_dataset.zip'

        if os.path.exists(os.path.join(self.dataset_dir, self.data_archive)):
            print(f'Using downloaded file: {os.path.join(self.dataset_dir, self.data_archive)}')
            self.unzip_data()
        else:
            session = requests.Session()
            response = session.get(url, stream=True)

            response_content_generator = response.iter_content(32768)
            first_chunk = None
            while not first_chunk:
                first_chunk = next(response_content_generator)

            self._save_response_content(
                    itertools.chain((first_chunk,),
                                    response_content_generator),
                    os.path.join(self.dataset_dir, self.data_archive))
            response.close()
            self.unzip_data()

    def unzip_data(self) -> None:
        """Unzip images to the self.dataset_dir directory"""
        archive = os.path.join(self.dataset_dir, self.data_archive)
        images = os.path.join(self.dataset_dir)
        
        if not os.path.exists(os.path.join(self.dataset_dir)):
            os.makedirs(os.path.join(self.dataset_dir))
        print(f'Extracting {archive} to {images}')

        with zipfile.ZipFile(archive, "r", compression=zipfile.ZIP_STORED) as zip:
            zip.extractall(images)

    def _save_response_content(
            self,
            response_gen: Iterator[bytes],
            destination: str) -> None:
        """Write data to the ZIP archive by small chunks"""
        with open(destination, "wb+") as f:
            pbar = tqdm(total=None)
            progress = 0

            for chunk in response_gen:
                if chunk:
                    f.write(chunk)
                    progress += len(chunk)
                    pbar.update(progress - pbar.n)
            pbar.close()

    def __len__(self) -> int:
        return len(self.data_ids)

    def __getitem__(
                    self,
                    index: int
                    ) -> np.ndarray:
        image, target = self.data['images'][index], self.data['targets'][index]
        image = np.asarray(Image.open(image))
        target = np.asarray(Image.open(target))

        if self.transform is not None:
            img = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_data_loader(
        arguments,
        mode: str = 'train',
        download: bool = True,
        **kwargs) -> torch.utils.data.DataLoader:
    """
    A function that obtain dataset instance and creade PyTorch type DataLoader

    Arguments
    ----------
    path_to_data : str
        path to directory where located images that the dataset will be based on.
    mode : str
        a string that define is dataset is designated for the training or for the testing process
    download : bool
        a flag that shows if we want explicitly download zip archive with images
        or we already have data in our local directory

    Return value
    ----------
    an object of PyTorch DataLoader class
    """
    train = True if mode == 'train' else False

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((256, 256))
    ])

    dataset = VDDataset(
        arguments.dataset_location,
        train,
        transform=transform,
        target_transform=transform,
        download=arguments.download_data)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        drop_last=True)

    return data_loader
