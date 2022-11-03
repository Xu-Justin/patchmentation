import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import Mask
from patchmentation.utils import validator
from tests import helper

def test_mask():
    path = helper.generate_filename('.jpg')
    mask = Mask(path)
    assert mask.path == path
    mask.summary()
    
def test_mask_image_array():
    width = 10
    height = 20
    mask = helper.generate_Mask(width, height)
    assert mask.shape() == (height, width)
    assert mask.width() == width
    assert mask.height() == height
    image_array = mask.image_array()
    validator.validate_mask_image_array(image_array, check_shape=True, expected_width=width, expected_height=height)
