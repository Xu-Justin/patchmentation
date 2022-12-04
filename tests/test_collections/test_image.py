import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from patchmentation.collections import Image
from tests import helper

def test_image():
    temp_image = helper.generate_Image()
    mask = helper.generate_Mask(temp_image.width, temp_image.height)
    image = Image(temp_image.path, mask)
    assert image.path == temp_image.path
    assert image.mask == mask
    assert (temp_image.path, mask) == tuple(image)
    str(image)
    
def test_image_image_array():
    width = 10
    height = 20
    image = helper.generate_Image(width, height, True)
    assert image.shape == (height, width, 4)
    assert image.width == width
    assert image.height == height
    assert image.channel == 4
    image_array = image.image_array
    assert image_array.shape == (height, width, 4)
    assert image_array.dtype == np.uint8

def test_image_image_array_no_mask():
    width = 10
    height = 20
    image = helper.generate_Image(width, height, None)
    assert image.shape == (height, width, 4)
    assert image.width == width
    assert image.height == height
    assert image.channel == 4
    image_array = image.image_array
    assert image_array.shape == (height, width, 4)
    assert image_array.dtype == np.uint8
    assert (image_array[:, :, 3] == 255).all()

def test_image_shape_cache_clear():
    width1, height1 = 10, 20
    image1 = helper.generate_Image(width1, height1)
    width2, height2 = 20, 10
    image2 = helper.generate_Image(width2, height2)
    image = Image(image1.path, None)
    assert image.width == width1
    assert image.height == height1
    image.mask = None
    image.path = image2.path
    assert image.width == width2
    assert image.height == height2
    