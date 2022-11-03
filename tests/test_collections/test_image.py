import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import Image
from patchmentation.utils import validator
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
