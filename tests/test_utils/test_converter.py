import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import BBox, Image, Patch
from patchmentation.utils import converter
from tests import helper

import numpy as np
import pytest

def test_array2image():
    image_array = helper.generate_image_array()
    image = converter.array2image(image_array)
    assert isinstance(image, Image)
    assert (image.image_array[:,:,:3] == image_array[:,:,:3]).all()

def test_array2image_zero_width():
    image_array = helper.generate_image_array(0, 5, 4)
    with pytest.raises(ValueError):
        converter.array2image(image_array)

def test_array2image_zero_height():
    image_array = helper.generate_image_array(5, 0, 4)
    with pytest.raises(ValueError):
        converter.array2image(image_array)

def test_image2patch():
    image = helper.generate_Image()
    class_name = helper.generate_class_name()
    patch = converter.image2patch(image, class_name)
    assert isinstance(patch, Patch)
    assert patch.class_name == class_name
    assert patch.image is image
    assert (patch.image_array == image.image_array).all()

def test_array2patch():
    image_array = helper.generate_image_array()
    class_name = helper.generate_class_name()
    patch = converter.array2patch(image_array, class_name)
    assert isinstance(patch, Patch)
    assert patch.class_name == class_name
    assert (patch.image_array[:,:,:3] == image_array[:,:,:3]).all()

def test_patch2image():
    image_array = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
    ])
    bbox = BBox(1, 0, 3, 1)
    _image = converter.array2image(image_array)
    patch = Patch(_image, bbox)
    image = converter.patch2image(patch)
    assert isinstance(image, Image)
    actual_image_array = image.image_array
    expected_image_array = np.array([
        [[4, 5, 6], [7, 8, 9]]
    ])
    assert (actual_image_array[:,:,:3] == expected_image_array[:,:,:3]).all()