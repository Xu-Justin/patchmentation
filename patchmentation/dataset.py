import os
import wget
import tarfile
from appdirs import user_cache_dir

from typing import Dict, List, Union
from patchmentation.utils import loader
from patchmentation.utils import deprecated
from patchmentation.collections import Dataset

PASCAL_VOC_2007 = 'PASCAL_VOC_2007'

_CACHEDIR = user_cache_dir('patchmentation-dataset') 
_PASCAL_VOC_2007_FOLDER = os.path.join(_CACHEDIR, 'VOCdevkit', 'VOC2007')

_PASCAL_VOC_2007_URL_TRAIN_VAL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
_PASCAL_VOC_2007_URL_TEST = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
_PASCAL_VOC_2007_TAR_TRAIN_VAL = os.path.join(_CACHEDIR, 'VOCtrainval_06-Nov-2007.tar')
_PASCAL_VOC_2007_TAR_TEST = os.path.join(_CACHEDIR, 'VOCtest_06-Nov-2007.tar')
_PASCAL_VOC_2007_FOLDER_IMAGES = os.path.join(_PASCAL_VOC_2007_FOLDER, 'JPEGImages')
_PASCAL_VOC_2007_FOLDER_ANNOTATIONS = os.path.join(_PASCAL_VOC_2007_FOLDER, 'Annotations')
_PASCAL_VOC_2007_IMAGESETS = os.path.join(_PASCAL_VOC_2007_FOLDER, 'ImageSets', 'Main')
_PASCAL_VOC_2007_IMAGESETS_TRAIN = os.path.join(_PASCAL_VOC_2007_IMAGESETS, 'train.txt')
_PASCAL_VOC_2007_IMAGESETS_VAL   = os.path.join(_PASCAL_VOC_2007_IMAGESETS, 'val.txt')
_PASCAL_VOC_2007_IMAGESETS_TEST  = os.path.join(_PASCAL_VOC_2007_IMAGESETS, 'test.txt')

@deprecated('use patchmentation.data.datautils.download() instead')
def _download(source, target):
    print(f'download from {source} to {target}')
    wget.download(source, target)

@deprecated('use patchmentation.data.datautils.extract_tar() instead')
def _extract_tar(source, target):
    print(f'extract from {source} to {target}')
    with tarfile.open(source) as f:
        f.extractall(target)

@deprecated('use patchmentation.data.datautils.rm() instead')
def _remove_file(file):
    print(f'removing {file}')
    os.remove(file)

@deprecated('use patchmentation.data.Data.load() instead')
def load(dataset, categories: Union[str, List[str]] = None) -> Dict[str, Dataset]:
    if dataset == PASCAL_VOC_2007:
        return load_pascal_voc_2007(categories)
    raise ValueError(f'Unexpected dataset value : {dataset}')

@deprecated('use patchmentation.data.PascalVOC2007().load() instead')
def load_pascal_voc_2007(categories: Union[str, List[str]] = None) -> Dict[str, Dataset]:
    os.makedirs(_CACHEDIR, exist_ok=True)
    if not os.path.exists(_PASCAL_VOC_2007_FOLDER):
        _download_pascal_voc_2007()
    if categories is None:
        categories = ['train', 'val', 'test']
    if isinstance(categories, str):
        categories = [categories]
    dataset = {}
    if 'train' in categories:
        dataset['train'] = loader.load_pascal_voc_dataset(_PASCAL_VOC_2007_FOLDER_IMAGES, _PASCAL_VOC_2007_FOLDER_ANNOTATIONS, _PASCAL_VOC_2007_IMAGESETS_TRAIN)
    if 'val' in categories:
        dataset['val'] = loader.load_pascal_voc_dataset(_PASCAL_VOC_2007_FOLDER_IMAGES, _PASCAL_VOC_2007_FOLDER_ANNOTATIONS, _PASCAL_VOC_2007_IMAGESETS_VAL)
    if 'test' in categories:
        dataset['test'] = loader.load_pascal_voc_dataset(_PASCAL_VOC_2007_FOLDER_IMAGES, _PASCAL_VOC_2007_FOLDER_ANNOTATIONS, _PASCAL_VOC_2007_IMAGESETS_TEST)
    return dataset

@deprecated('use patchmentation.data.PascalVOC2007().download() instead')
def _download_pascal_voc_2007():
    if not os.path.exists(_PASCAL_VOC_2007_TAR_TRAIN_VAL):
            _download_pascal_voc_2007_train_val()
    if not os.path.exists(_PASCAL_VOC_2007_TAR_TEST):
        _download_pascal_voc_2007_test()
    _extract_pascal_voc_2007_train_val()
    _extract_pascal_voc_2007_test()
    _remove_tar_pascal_voc_2007_train_val()
    _remove_tar_pascal_voc_2007_test()

@deprecated('use patchmentation.data.PascalVOC2007TrainVal().download() instead')
def _download_pascal_voc_2007_train_val():
    os.makedirs(_CACHEDIR, exist_ok=True)
    if os.path.exists(_PASCAL_VOC_2007_TAR_TRAIN_VAL):
        raise FileExistsError(_PASCAL_VOC_2007_TAR_TRAIN_VAL)
    _download(_PASCAL_VOC_2007_URL_TRAIN_VAL, _PASCAL_VOC_2007_TAR_TRAIN_VAL)

@deprecated('use patchmentation.data.PascalVOC2007Test().download() instead')
def _download_pascal_voc_2007_test():
    os.makedirs(_CACHEDIR, exist_ok=True)
    if os.path.exists(_PASCAL_VOC_2007_TAR_TEST):
        raise FileExistsError(_PASCAL_VOC_2007_TAR_TEST)
    _download(_PASCAL_VOC_2007_URL_TEST, _PASCAL_VOC_2007_TAR_TEST)

@deprecated('use patchmentation.data.PascalVOC2007TrainVal().extract() instead')
def _extract_pascal_voc_2007_train_val():
    os.makedirs(_CACHEDIR, exist_ok=True)
    _extract_tar(_PASCAL_VOC_2007_TAR_TRAIN_VAL, _CACHEDIR)

@deprecated('use patchmentation.data.PascalVOC2007Test().extract() instead')
def _extract_pascal_voc_2007_test():
    os.makedirs(_CACHEDIR, exist_ok=True)
    _extract_tar(_PASCAL_VOC_2007_TAR_TEST, _CACHEDIR)

@deprecated('use patchmentation.data.PascalVOC2007TrainVal().remove_archive() instead')
def _remove_tar_pascal_voc_2007_train_val():
    if os.path.exists(_PASCAL_VOC_2007_TAR_TRAIN_VAL):
        _remove_file(_PASCAL_VOC_2007_TAR_TRAIN_VAL)

@deprecated('use patchmentation.data.PascalVOC2007Test().remove_archive() instead')
def _remove_tar_pascal_voc_2007_test():
    if os.path.exists(_PASCAL_VOC_2007_TAR_TEST):
        _remove_file(_PASCAL_VOC_2007_TAR_TEST)
