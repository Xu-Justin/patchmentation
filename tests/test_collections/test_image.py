import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from patchmentation.collections import Image
from patchmentation.utils import validator
from patchmentation.utils import loader
from tests import helper

def test_image():
    path = helper.generate_filename('.jpg')
    image = Image(path)
    assert image.path == path
    image.summary()
    
def test_image_image_array():
    width = 10
    height = 20
    image = helper.generate_Image(width, height)
    assert image.shape() == (height, width, 3)
    assert image.width() == width
    assert image.height() == height
    assert image.channel() == 3
    image_array = image.image_array()
    validator.validate_image_array(image_array, check_shape=True, expected_width=width, expected_height=height, expected_channel=3)

def test_image_mask():
    width = 20
    height = 15
    mask = helper.generate_Mask(width, height)
    image = helper.generate_Image(width, height, mask)
    assert image.shape() == (height, width, 4)
    assert image.width() == width
    assert image.height() == height
    assert image.channel() == 4
    assert image.mask is mask
    image_array = image.image_array()
    validator.validate_image_array(image_array, check_shape=True, expected_width=width, expected_height=height, expected_channel=4)

def test_image_get_mask_1():
    width = 3
    height = 5
    image = helper.generate_Image(width, height)
    mask_image_array = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ])
    image.mask = loader.save_mask_image_array_temporary(mask_image_array)
    assert (image.get_mask().image_array() == mask_image_array).all()

def test_image_get_mask_2():
    width = 3
    height = 5
    image = helper.generate_Image(width, height)
    mask_image_array = np.array([
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
    ])
    assert (image.get_mask().image_array() == mask_image_array).all()
