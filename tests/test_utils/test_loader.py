import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests import helper
from patchmentation.utils import loader
from patchmentation.collections import Dataset

import numpy as np
import pytest

YOLO_FOLDER_IMAGES = 'dataset/sample_format_yolo/obj_train_data/'
YOLO_FOLDER_ANNOTATIONS = 'dataset/sample_format_yolo/obj_train_data/'
YOLO_FILE_NAMES = 'dataset/sample_format_yolo/obj.names'

COCO_FOLDER_IMAGES = 'dataset/sample_format_coco/images/'
COCO_FILE_ANNOTATIONS = 'dataset/sample_format_coco/annotations/instances_default.json'

PASCAL_VOC_FOLDER_IMAGES = 'dataset/sample_format_pascal_voc/JPEGImages/'
PASCAL_VOC_FOLDER_ANNOTATIONS = 'dataset/sample_format_pascal_voc/Annotations/'
PASCAL_VOC_FILE_IMAGESETS = 'dataset/sample_format_pascal_voc/ImageSets/Main/default.txt'

def test_save_load_image_array_1():
    image_array = helper.generate_image_array()
    image = loader.save_image_array_temporary(image_array)
    reloaded_image_array = loader.load_image_array(image)
    assert reloaded_image_array.shape[:2] == image_array.shape[:2]
    assert reloaded_image_array.shape[2] == 4
    assert reloaded_image_array.dtype == np.uint8
    assert (reloaded_image_array[:,:,:3] == image_array).all()
    assert (reloaded_image_array[:,:,3] == 255).all()
    
def test_save_load_image_array_2():
    image_array = helper.generate_image_array()
    image = loader.save_image_array_temporary(image_array)
    reloaded_image_array = loader.load_image_array(image.path)
    assert reloaded_image_array.shape[:2] == image_array.shape[:2]
    assert reloaded_image_array.shape[2] == 4
    assert reloaded_image_array.dtype == np.uint8
    assert (reloaded_image_array[:,:,:3] == image_array).all()
    assert (reloaded_image_array[:,:,3] == 255).all()

def test_save_load_image_array_3():
    image_array = helper.generate_image_array()
    image = loader.save_image_array_temporary(image_array)
    reloaded_image = loader.load_image(image.path)
    reloaded_image_array = loader.load_image_array(reloaded_image)
    assert reloaded_image_array.shape[:2] == image_array.shape[:2]
    assert reloaded_image_array.shape[2] == 4
    assert reloaded_image_array.dtype == np.uint8
    assert (reloaded_image_array[:,:,:3] == image_array).all()
    assert (reloaded_image_array[:,:,3] == 255).all()

def test_save_load_mask_image_array_1():
    image_array = helper.generate_mask_image_array()
    mask = loader.save_mask_image_array_temporary(image_array)
    reloaded_image_array = loader.load_image_array(mask)
    assert len(reloaded_image_array.shape) == 2
    assert reloaded_image_array.shape == image_array.shape
    assert reloaded_image_array.dtype == np.uint8
    assert (reloaded_image_array == image_array).all()

def test_save_load_mask_image_array_2():
    image_array = helper.generate_image_array(channel=4)
    image = loader.save_image_array_temporary(image_array)
    reloaded_image_array = loader.load_image_array(image)
    assert reloaded_image_array.shape[:2] == image_array.shape[:2]
    assert reloaded_image_array.shape[2] == 4
    assert reloaded_image_array.dtype == np.uint8
    assert (reloaded_image_array == image_array).all()

def test_save_image_array_error():
    image_array = helper.generate_image_array(channel=1)
    with pytest.raises(TypeError):
        loader.save_image_array(image_array, "")

def test_load_image_array_error():
    with pytest.raises(TypeError):
        loader.load_image_array(5)

def test_loader_yolo():
    dataset = loader.load_yolo_dataset(YOLO_FOLDER_IMAGES, YOLO_FOLDER_ANNOTATIONS, YOLO_FILE_NAMES)
    validate_dataset(dataset)

def test_loader_coco():
    dataset = loader.load_coco_dataset(COCO_FOLDER_IMAGES, COCO_FILE_ANNOTATIONS)
    validate_dataset(dataset)

def test_loader_pascal_voc():
    dataset = loader.load_pascal_voc_dataset(PASCAL_VOC_FOLDER_IMAGES, PASCAL_VOC_FOLDER_ANNOTATIONS, PASCAL_VOC_FILE_IMAGESETS)
    validate_dataset(dataset)

_NCLASSES = 3
_NIMAGES = 30
_CLASSES = ['person', 'car', 'horse']

def validate_dataset(dataset: Dataset) -> None:
    assert isinstance(dataset, Dataset)
    assert len(dataset.classes) == _NCLASSES
    assert helper.compare_unordered_list_equal(dataset.classes, _CLASSES)
    assert len(dataset.image_patches) == _NIMAGES
