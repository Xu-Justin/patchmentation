from patchmentation.collections import Dataset
from patchmentation.utils import loader
from . import datautils
from .data import Data
import os

DOWNLOAD_URL_TRAIN_VAL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
DOWNLOAD_URL_TEST = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'

FOLDER_CACHE = os.path.join(datautils.FOLDER_CACHE, 'PascalVOC2007')
FILE_TAR_TRAIN_VAL = os.path.join(datautils.FOLDER_CACHE, 'VOCtrainval_06-Nov-2007.tar')
FILE_TAR_TEST = os.path.join(datautils.FOLDER_CACHE, 'VOCtest_06-Nov-2007.tar')

FOLDER = os.path.join(FOLDER_CACHE, 'VOCdevkit', 'VOC2007')
FOLDER_IMAGES = os.path.join(FOLDER, 'JPEGImages')
FOLDER_ANNOTATIONS = os.path.join(FOLDER, 'Annotations')
FOLDER_IMAGESETS = os.path.join(FOLDER, 'ImageSets', 'Main')
FILE_IMAGESETS_TRAIN = os.path.join(FOLDER_IMAGESETS, 'train.txt')
FILE_IMAGESETS_VAL = os.path.join(FOLDER_IMAGESETS, 'val.txt')
FILE_IMAGESETS_TEST = os.path.join(FOLDER_IMAGESETS, 'test.txt')

class PascalVOC2007(Data):
    
    def load(self, category: str) -> Dataset:
        if category == 'train':
            return self.load_train()
        if category == 'val':
            return self.load_val()
        if category == 'test':
            return self.load_test()
        raise ValueError(f'Unexpected category {category}')

    def load_train(self) -> Dataset:
        if not self.exists_train():
            if not self.exists_tar_train_val():
                self.download_train_val(overwrite=True)
            self.extract_train_val(overwrite=True)
        return loader.load_pascal_voc_dataset(FOLDER_IMAGES, FOLDER_ANNOTATIONS, FILE_IMAGESETS_TRAIN)
        
    def load_val(self) -> Dataset:
        if not self.exists_val():
            if not self.exists_tar_train_val():
                self.download_train_val(overwrite=True)            
            self.extract_train_val(overwrite=True)
        return loader.load_pascal_voc_dataset(FOLDER_IMAGES, FOLDER_ANNOTATIONS, FILE_IMAGESETS_VAL)

    def load_test(self) -> Dataset:
        if not self.exists_test():
            if not self.exists_tar_test():
                self.download_test(overwrite=True)
            self.extract_test(overwrite=True)
        return loader.load_pascal_voc_dataset(FOLDER_IMAGES, FOLDER_ANNOTATIONS, FILE_IMAGESETS_TEST)

    def download(self, overwrite: bool = False) -> None:
        self.download_train_val(overwrite)
        self.download_test(overwrite)

    def download_train_val(self, overwrite: bool = False) -> None:
        datautils.download(DOWNLOAD_URL_TRAIN_VAL, FILE_TAR_TRAIN_VAL, overwrite)

    def download_test(self, overwrite: bool = False) -> None:
        datautils.download(DOWNLOAD_URL_TEST, FILE_TAR_TEST, overwrite)

    def extract(self, overwrite: bool = False) -> None:
        self.extract_train_val(overwrite)
        self.extract_test(overwrite)

    def extract_train_val(self, overwrite: bool = False) -> None:
        datautils.extract_tar(FILE_TAR_TRAIN_VAL, FOLDER_CACHE, overwrite)

    def extract_test(self, overwrite: bool = False) -> None:
        datautils.extract_tar(FILE_TAR_TEST, FOLDER_CACHE, overwrite)

    def exists_archive(self) -> bool:
        return self.exists_train_val() and self.exists_tar_test()

    def exists_tar_train_val(self) -> bool:
        return os.path.exists(FILE_TAR_TRAIN_VAL)

    def exists_tar_test(self) -> bool:
        return os.path.exists(FILE_TAR_TEST)

    def exists(self) -> bool:
        return self.exists_train_val() and self.exists_test()

    def exists_train_val(self) -> bool:
        return self.exists_train() and self.exists_val()

    def exists_train(self) -> bool:
        return os.path.exists(FILE_IMAGESETS_TRAIN)

    def exists_val(self) -> bool:
        return os.path.exists(FILE_IMAGESETS_VAL)

    def exists_test(self) -> bool:
        return os.path.exists(FILE_IMAGESETS_TEST)

    def remove_archive(self) -> None:
        self.remove_tar_train_val()
        self.remove_tar_test()
    
    def remove_tar_train_val(self) -> None:
        datautils.rm(FILE_TAR_TRAIN_VAL)

    def remove_tar_test(self) -> None:
        datautils.rm(FILE_TAR_TEST)

    def remove(self) -> None:
        datautils.rm(FOLDER)
