from patchmentation.collections import Dataset
from patchmentation.utils import loader
from . import datautils
from .data import Data
import os

DOWNLOAD_URL_TRAIN_VAL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
DOWNLOAD_URL_TEST = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'

FOLDER_VOC2007 = os.path.join('VOCdevkit', 'VOC2007')
FOLDER_IMAGES = os.path.join(FOLDER_VOC2007, 'JPEGImages')
FOLDER_ANNOTATIONS = os.path.join(FOLDER_VOC2007, 'Annotations')
FOLDER_IMAGESETS = os.path.join(FOLDER_VOC2007, 'ImageSets/Main')
FILE_IMAGESETS_TRAIN = os.path.join(FOLDER_IMAGESETS, 'train.txt')
FILE_IMAGESETS_VAL = os.path.join(FOLDER_IMAGESETS, 'val.txt')
FILE_IMAGESETS_TEST = os.path.join(FOLDER_IMAGESETS, 'test.txt')

class PascalVOC2007(Data):

    ex = Exception('This method/attribute is not available. Use PascalVOC2007TrainVal or PascalVOC2007Test instead.')

    @property
    def name(self) -> str:
        return 'pascal-voc-2007'

    @property
    def root(self) -> str:
        raise self.ex

    @property
    def folder(self) -> str:
        raise self.ex

    @property
    def archive_name(self) -> str:
        raise self.ex

    @property
    def file_archive(self) -> str:
        raise self.ex

    def load(self, category: str) -> Dataset:
        if category == 'train' or category == 'val':
            return PascalVOC2007TrainVal().load(category)
        if category == 'test':
            return PascalVOC2007Test().load()
        raise ValueError(f'Unexpected category {category}')

    def download(self, overwrite: bool = False) -> None:
        PascalVOC2007TrainVal().download(overwrite)
        PascalVOC2007Test().download(overwrite)

    def extract(self, overwrite: bool = False) -> None:
        PascalVOC2007TrainVal().extract(overwrite)
        PascalVOC2007Test().extract(overwrite)

    def exists_archive(self) -> bool:
        return PascalVOC2007TrainVal().exists_archive() and PascalVOC2007Test().exists_archive()

    def exists(self) -> bool:
        return PascalVOC2007TrainVal().exists() and PascalVOC2007Test().exists()

    def remove_archive(self) -> None:
        PascalVOC2007TrainVal().remove_archive()
        PascalVOC2007Test().remove_archive()
        
    def remove(self) -> None:
        PascalVOC2007TrainVal().remove()
        PascalVOC2007Test().remove()

    def initialize(self) -> None:
        PascalVOC2007TrainVal().initialize()
        PascalVOC2007Test().initialize()

class PascalVOC2007TrainVal(Data):

    @property
    def name(self) -> str:
        return 'pascal-voc-2007-train-val'

    @property
    def folder_images(self) -> str:
        return os.path.join(self.folder, FOLDER_IMAGES)

    @property
    def folder_annotations(self) -> str:
        return os.path.join(self.folder, FOLDER_ANNOTATIONS)

    @property
    def file_imagesets_train(self) -> str:
        return os.path.join(self.folder, FILE_IMAGESETS_TRAIN)

    @property
    def file_imagesets_val(self) -> str:
        return os.path.join(self.folder, FILE_IMAGESETS_VAL)
    
    def load(self, category: str) -> Dataset:
        self.initialize()
        if category == 'train':
            return self.load_train()
        if category == 'val':
            return self.load_val()
        raise ValueError(f'Unexpected category {category}')

    def load_train(self) -> Dataset:
        return loader.load_pascal_voc_dataset(self.folder_images, self.folder_annotations, self.file_imagesets_train)
    
    def load_val(self) -> Dataset:
        return loader.load_pascal_voc_dataset(self.folder_images, self.folder_annotations, self.file_imagesets_val)
        
    def download(self, overwrite: bool = False) -> None:
        datautils.download(DOWNLOAD_URL_TRAIN_VAL, self.file_archive, overwrite)

    def extract(self, overwrite: bool = False) -> None:
        datautils.extract_tar(self.file_archive, self.folder, overwrite)

class PascalVOC2007Test(Data):

    @property
    def name(self) -> str:
        return 'pascal-voc-2007-test'

    @property
    def folder_images(self) -> str:
        return os.path.join(self.folder, FOLDER_IMAGES)

    @property
    def folder_annotations(self) -> str:
        return os.path.join(self.folder, FOLDER_ANNOTATIONS)

    @property
    def file_imagesets(self) -> str:
        return os.path.join(self.folder, FILE_IMAGESETS_TEST)
    
    def load(self) -> Dataset:
        self.initialize()
        return loader.load_pascal_voc_dataset(self.folder_images, self.folder_annotations, self.file_imagesets)

    def download(self, overwrite: bool = False) -> None:
        datautils.download(DOWNLOAD_URL_TEST, self.file_archive, overwrite)

    def extract(self, overwrite: bool = False) -> None:
        datautils.extract_tar(self.file_archive, self.folder, overwrite)
