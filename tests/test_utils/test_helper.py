import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests import helper
from patchmentation.collections import BBox, Mask, EmptyMask, Image, Patch, ImagePatch, Dataset

import numpy as np

def test_generate_bbox_1():
    width = 20
    height = 1000
    bbox = helper.generate_BBox(width, height)
    assert isinstance(bbox, BBox)
    assert bbox.width <= width
    assert bbox.height <= height

def test_generate_bbox_2():
    width = 1000
    height = 20
    bbox = helper.generate_BBox(width, height)
    assert isinstance(bbox, BBox)
    assert bbox.width <= width
    assert bbox.height <= height

def test_generate_mask_1():
    width = 100
    height = 20
    mask = helper.generate_Mask(width, height)
    assert isinstance(mask, Mask)
    assert mask.shape == (height, width)

def test_generate_mask_2():
    width = 20
    height = 100
    mask = helper.generate_Mask(width, height)
    assert isinstance(mask, Mask)
    assert mask.shape == (height, width)

def test_generate_mask_empty_1():
    width = 100
    height = 20
    mask = helper.generate_Mask_Empty(width, height)
    assert isinstance(mask, Mask)
    assert mask == EmptyMask(width, height)

def test_generate_mask_empty_2():
    width = 20
    height = 100
    mask = helper.generate_Mask_Empty(width, height)
    assert isinstance(mask, Mask)
    assert mask == EmptyMask(width, height)

def test_generate_image_1():
    width = 100
    height = 20
    image = helper.generate_Image(width, height)
    assert isinstance(image, Image)
    assert image.shape == (height, width, 4)
    assert image.mask == EmptyMask(width, height)

def test_generate_image_2():
    width = 20
    height = 100
    image = helper.generate_Image(width, height)
    assert isinstance(image, Image)
    assert image.shape == (height, width, 4)
    assert image.mask == EmptyMask(width, height)

def test_generate_image_3():
    width = 20
    height = 100
    mask = helper.generate_Mask(width, height)
    image = helper.generate_Image(width, height, mask)
    assert isinstance(image, Image)
    assert image.shape == (height, width, 4)
    assert image.mask == mask

def test_generate_image_4():
    width = 20
    height = 100
    mask = True
    image = helper.generate_Image(width, height, mask)
    assert isinstance(image, Image)
    assert image.shape == (height, width, 4)
    assert image.mask.shape == (height, width)

def test_generate_patch():
    image = helper.generate_Image()
    patch = helper.generate_Patch(image)
    assert isinstance(patch, Patch)
    assert patch.image == image

def test_generate_imagePatch():
    imagePatch = helper.generate_ImagePatch()
    assert isinstance(imagePatch, ImagePatch)

def test_generate_dataset():
    dataset = helper.generate_Dataset()
    assert isinstance(dataset, Dataset)

def test_compare_float_equal_1():
    float_1 = 0.1 + 0.2
    float_2 = 0.3
    assert helper.compare_float_equal(float_1, float_2)

def test_compare_float_equal_2():
    float_1 = 0.1 + 0.2
    float_2 = 0.30001
    assert not helper.compare_float_equal(float_1, float_2)

def test_compare_float_1():
    float_1 = 0.1 + 0.2
    float_2 = 0.3
    assert helper.compare_float(float_1, float_2) == 0

def test_compare_float_2():
    float_1 = 0.1 + 0.2
    float_2 = 0.30001
    assert helper.compare_float(float_1, float_2) == -1

def test_compare_float_3():
    float_1 = 0.1 + 0.2
    float_2 = 0.29999
    assert helper.compare_float(float_1, float_2) == 1

def test_compare_unordered_list_equal_1():
    obj_1 = object()
    obj_2 = object()
    obj_3 = object()
    list_1 = [obj_1, obj_2, obj_3, obj_1]
    list_2 = [obj_3, obj_1, obj_1, obj_2]
    assert helper.compare_unordered_list_equal(list_1, list_2)
    assert list_1 == [obj_1, obj_2, obj_3, obj_1]
    assert list_2 == [obj_3, obj_1, obj_1, obj_2]

def test_compare_unordered_list_equal_2():
    obj_1 = object()
    obj_2 = object()
    obj_3 = object()
    list_1 = [obj_1, obj_2, obj_3, obj_1]
    list_2 = [obj_3, obj_1, obj_2, obj_2]
    assert not helper.compare_unordered_list_equal(list_1, list_2)
    assert list_1 == [obj_1, obj_2, obj_3, obj_1]
    assert list_2 == [obj_3, obj_1, obj_2, obj_2]

def test_compare_unordered_list_equal_3():
    obj_1 = object()
    obj_2 = object()
    obj_3 = object()
    list_1 = [obj_1, obj_2, obj_3]
    list_2 = [obj_3, obj_1, obj_3, obj_2]
    assert not helper.compare_unordered_list_equal(list_1, list_2)
    assert list_1 == [obj_1, obj_2, obj_3]
    assert list_2 == [obj_3, obj_1, obj_3, obj_2]

def test_check_grayscale_1():
    image_array = np.array([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        [[4, 4, 4], [5, 5, 5], [6, 6, 6]]
    ])
    assert helper.check_grayscale(image_array)

def test_check_grayscale_2():
    image_array = np.array([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        [[4, 4, 4], [5, 4, 5], [6, 6, 6]]
    ])
    assert not helper.check_grayscale(image_array)

