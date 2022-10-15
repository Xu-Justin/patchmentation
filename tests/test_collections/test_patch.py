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
    image_array = patch.image_array()
    validator.validate_image_array(image_array, check_shape=True, expected_width=4, expected_height=5, expected_channel=3)
