import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import Patch, BBox
from tests import helper

import numpy as np
import pytest

def test_patch():
    image = helper.generate_Image()
    bbox = helper.generate_BBox(image.width, image.height)
    class_name = helper.generate_class_name()
    mask = helper.generate_Mask(image.width, image.height)
    patch = Patch(image, bbox, class_name, mask)
    assert patch.image is image
    assert patch.bbox is bbox
    assert patch.class_name == class_name
    assert patch.mask == mask
    assert (image, bbox, class_name) == tuple(patch)
    str(patch)

def test_patch_image_array():
    width = 10
    height = 20
    image = helper.generate_Image(width, height, True)
    bbox = BBox(1, 2, 5, 7)
    patch = Patch(image, bbox, None)
    assert patch.shape == (bbox.height, bbox.width, 4)
    assert patch.width == bbox.width
    assert patch.height == bbox.height
    assert patch.channel == 4
    image_array = patch.image_array
    assert image_array.shape == (bbox.height, bbox.width, 4)
    assert image_array.dtype == np.uint8

def test_patch_image_array_with_mask():
    width = 10
    height = 20
    image = helper.generate_Image(width, height, True)
    bbox = BBox(1, 2, 5, 7)
    mask = helper.generate_Mask(width, height)
    patch = Patch(image, bbox, None, mask)
    assert patch.shape == (bbox.height, bbox.width, 4)
    assert patch.width == bbox.width
    assert patch.height == bbox.height
    assert patch.channel == 4
    image_array = patch.image_array
    assert image_array.shape == (bbox.height, bbox.width, 4)
    assert image_array.dtype == np.uint8

def test_patch_image_array_with_zero_mask():
    width = 10
    height = 20
    image = helper.generate_Image(width, height, True)
    bbox = BBox(1, 2, 5, 7)
    mask = helper.generate_zero_Mask(width, height)
    patch = Patch(image, bbox, None, mask)
    assert patch.shape == (bbox.height, bbox.width, 4)
    assert patch.width == bbox.width
    assert patch.height == bbox.height
    assert patch.channel == 4
    image_array = patch.image_array
    assert image_array.shape == (bbox.height, bbox.width, 4)
    assert image_array.dtype == np.uint8
    assert (image_array[:, :, 3] == 0).all()

def test_patch_shape_cache_clear():
    width = 10
    height = 20
    image = helper.generate_Image(width, height, True)
    bbox1 = BBox(3, 5, 8, 6)
    patch = Patch(image, bbox1, None)
    assert patch.shape == (bbox1.height, bbox1.width, 4)
    assert patch.width == bbox1.width
    assert patch.height == bbox1.height
    assert patch.channel == 4
    bbox2 = BBox(4, 1, 6, 15)
    patch.bbox = bbox2
    assert patch.shape == (bbox2.height, bbox2.width, 4)
    assert patch.width == bbox2.width
    assert patch.height == bbox2.height
    assert patch.channel == 4

def test_patch_error_bbox():
    width = 10
    height = 20
    image = helper.generate_Image(width, height, True)
    Patch(image, BBox(0, 0, width, height), None)
    with pytest.raises(ValueError):
        Patch(image, BBox(0, 0, width+1, height), None)
    with pytest.raises(ValueError):
        Patch(image, BBox(0, 0, width, height+1), None)

def test_patch_error_image():
    width = 10
    height = 20    
    patch = Patch(None, BBox(0, 0, width, height), None)
    image = helper.generate_Image(width, height, True)
    patch.image = image
    with pytest.raises(ValueError):
        image = helper.generate_Image(width-1, height, True)
        patch.image = image
    with pytest.raises(ValueError):
        image = helper.generate_Image(width, height-1, True)
        patch.image = image

def test_patch_mask_error():
    width = 10
    height = 20   
    image = helper.generate_Image(width, height, True) 
    patch = Patch(image, BBox(0, 0, width, height), None)
    with pytest.raises(ValueError):
        mask = helper.generate_Mask(width-1, height)
        patch.mask = mask
    with pytest.raises(ValueError):
        mask = helper.generate_Mask(width, height+1)
        patch.mask = mask
