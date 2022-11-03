import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import Patch, BBox
from patchmentation.utils import validator
from tests import helper

import numpy as np

def test_patch():
    image = helper.generate_Image()
    bbox = helper.generate_BBox()
    class_name = helper.generate_class_name()
    patch = Patch(image, bbox, class_name)
    assert patch.image is image
    assert patch.bbox is bbox
    assert patch.class_name == class_name
    assert (image, bbox, class_name) == tuple(patch)
    patch.summary()

def test_patch_image_array():
    width = 10
    height = 20
    image = helper.generate_Image(width, height)
    bbox = BBox(1, 2, 5, 7)
    patch = Patch(image, bbox, None)
    assert patch.shape() == (bbox.height(), bbox.width(), 3)
    assert patch.width() == bbox.width()
    assert patch.height() == bbox.height()
    assert patch.channel() == 3
    image_array = patch.image_array()
    validator.validate_image_array(image_array, check_shape=True, expected_width=bbox.width(), expected_height=bbox.height(), expected_channel=3)

def test_patch_mask_image_array():
    width = 20
    height = 10
    mask = helper.generate_Mask(width, height)
    image = helper.generate_Image(width, height, mask)
    bbox = BBox(3, 2, 5, 8)
    patch = Patch(image, bbox, None)
    assert patch.shape() == (bbox.height(), bbox.width(), 4)
    assert patch.width() == bbox.width()
    assert patch.height() == bbox.height()
    assert patch.channel() == 4
    image_array = patch.image_array()
    validator.validate_image_array(image_array, check_shape=True, expected_width=bbox.width(), expected_height=bbox.height(), expected_channel=4)
