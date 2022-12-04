import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import Patch, BBox
from tests import helper

import numpy as np

def test_patch():
    image = helper.generate_Image()
    bbox = helper.generate_BBox(image.width, image.height)
    class_name = helper.generate_class_name()
    patch = Patch(image, bbox, class_name)
    assert patch.image is image
    assert patch.bbox is bbox
    assert patch.class_name == class_name
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
    patch = Patch(image, bbox2, None)
    assert patch.shape == (bbox2.height, bbox2.width, 4)
    assert patch.width == bbox2.width
    assert patch.height == bbox2.height
    assert patch.channel == 4
    