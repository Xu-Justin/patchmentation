import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests import helper
from patchmentation.utils import loader
from patchmentation.utils import validator
from patchmentation.collections import BBox

import pytest
import numpy as np

YOLO_FOLDER_IMAGES = 'dataset/sample_format_yolo/obj_train_data/'
YOLO_FOLDER_ANNOTATIONS = 'dataset/sample_format_yolo/obj_train_data/'
YOLO_FILE_NAMES = 'dataset/sample_format_yolo/obj.names'

COCO_FOLDER_IMAGES = 'dataset/sample_format_coco/images/'
COCO_FILE_ANNOTATIONS = 'dataset/sample_format_coco/annotations/instances_default.json'

PASCAL_VOC_FOLDER_IMAGES = 'dataset/sample_format_pascal_voc/JPEGImages/'
PASCAL_VOC_FOLDER_ANNOTATIONS = 'dataset/sample_format_pascal_voc/Annotations/'
PASCAL_VOC_FILE_IMAGESETS = 'dataset/sample_format_pascal_voc/ImageSets/Main/default.txt'
PASCAL_VOC_FILE_CLASSES = 'dataset/sample_format_pascal_voc/classes.txt'

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

def test_loader_yolo():
    dataset = loader.load_yolo_dataset(YOLO_FOLDER_IMAGES, YOLO_FOLDER_ANNOTATIONS, YOLO_FILE_NAMES)
    validator.validate_Dataset(dataset)

@pytest.mark.skip('Not implemented')
def test_loader_coco():
    dataset = loader.load_coco_dataset(COCO_FOLDER_IMAGES, COCO_FILE_ANNOTATIONS)
    validator.validate_Dataset(dataset)

@pytest.mark.skip('Not implemented')
def test_loader_pascal_voc():
    dataset = loader.load_pascal_voc_dataset(PASCAL_VOC_FOLDER_IMAGES, PASCAL_VOC_FOLDER_ANNOTATIONS, PASCAL_VOC_FILE_IMAGESETS, PASCAL_VOC_FILE_CLASSES)
    validator.validate_Dataset(dataset)
