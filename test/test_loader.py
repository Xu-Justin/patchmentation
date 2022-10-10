import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test import helper
from patchmentation.utils import loader
from patchmentation.utils import validator
from patchmentation.collections import BBox

import pytest
import numpy as np

YOLO_FOLDER_IMAGES = 'dataset/sample_format_yolo/obj_train_data/'
YOLO_FOLDER_ANNOTATIONS = 'dataset/sample_format_yolo/obj_train_data/'
YOLO_FILE_NAMES = 'dataset/sample_format_yolo/obj.names'

def test_save_load_image_array_1():
    image_array = helper.generate_image_array()
    image = loader.save_image_array_temporary(image_array)
    reloaded_image_array = loader.load_image_array(image)
    validator.validate_image_array(reloaded_image_array)
    assert (reloaded_image_array == image_array).all()

def test_save_load_image_array_2():
    image_array = helper.generate_image_array()
    image = loader.save_image_array_temporary(image_array)
    reloaded_image_array = loader.load_image_array(image.path)
    validator.validate_image_array(reloaded_image_array)
    assert (reloaded_image_array == image_array).all()

def test_save_load_image_array_3():
    image_array = helper.generate_image_array()
    image = loader.save_image_array_temporary(image_array)
    reloaded_image = loader.load_image(image.path)
    reloaded_image_array = loader.load_image_array(reloaded_image)
    validator.validate_image_array(reloaded_image_array)
    assert (reloaded_image_array == image_array).all()

def test_load_patch_array():
    image_array = np.array([
        [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
        [[4, 3, 2], [5, 4, 3], [6, 5, 4]],
        [[7, 8, 9], [8, 9, 10], [9, 10, 11]],
        [[10, 9, 8], [11, 10, 9], [12, 11, 10]]
    ])
    image = loader.save_image_array_temporary(image_array)
    bbox = BBox(1, 0, 3, 4)
    actual_patch_array = loader.load_patch_array(image, bbox)
    expected_patch_array = np.array([
        [[2, 3, 4], [3, 4, 5]],
        [[5, 4, 3], [6, 5, 4]],
        [[8, 9, 10], [9, 10, 11]],
        [[11, 10, 9], [12, 11, 10]]
    ])
    assert (actual_patch_array == expected_patch_array).all()

def test_loader_yolo():
    dataset = loader.load_yolo_dataset(YOLO_FOLDER_IMAGES, YOLO_FOLDER_ANNOTATIONS, YOLO_FILE_NAMES)
    validator.validate_Dataset(dataset)

@pytest.mark.skip('Not implemented')
def test_loader_coco():
    dataset = loader.load_coco_dataset()
    validator.validate_Dataset(dataset)

@pytest.mark.skip('Not implemented')
def test_loader_pascal_voc():
    dataset = loader.load_pascal_voc_dataset()
    validator.validate_Dataset(dataset)
