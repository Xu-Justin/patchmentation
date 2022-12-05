import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import Mask, EmptyMask
from patchmentation.collections.mask import equal
from patchmentation.utils import loader
from tests import helper

import numpy as np
import pytest

def test_mask():
    path = "path"
    mask = Mask(path)
    assert mask.path == path
    str(mask)
    
def test_mask_image_array():
    width = 10
    height = 20
    mask = helper.generate_Mask(width, height)
    assert mask.shape == (height, width)
    assert mask.width == width
    assert mask.height == height
    image_array = mask.image_array
    assert image_array.shape == (height, width)
    assert image_array.dtype == np.uint8

def test_mask_shape_cache_clear():
    width1, height1 = 10, 20
    mask1 = helper.generate_Mask(width1, height1)
    width2, height2 = 15, 30
    mask2 = helper.generate_Mask(width2, height2)
    mask = Mask(mask1.path)
    assert mask.shape == (height1, width1)
    assert mask.width == width1
    assert mask.height == height1
    mask.path = mask2.path
    assert mask.shape == (height2, width2)
    assert mask.width == width2
    assert mask.height == height2

def test_mask_eq_true_1():
    path = "path"
    mask1 = Mask(path)
    mask2 = Mask(path)
    assert mask1 == mask2

def test_mask_eq_true_2():
    width = 10
    height = 20
    mask1 = helper.generate_Mask_Empty(width, height)
    mask2 = EmptyMask(width, height)
    assert mask1 == mask2

def test_mask_eq_false_1():
    mask1 = helper.generate_Mask(10, 20)
    mask2 = helper.generate_Mask(20, 10)
    assert mask1.path != mask2.path
    assert mask1 != mask2

def test_mask_eq_false_2():
    mask1 = helper.generate_Mask(10, 20)
    mask2 = helper.generate_Mask(10, 20)
    assert mask1.path != mask2.path
    assert mask1.shape == mask2.shape
    assert mask1 != mask2

def test_mask_eq_false_3():
    width = 10
    height = 20
    mask1 = helper.generate_Mask(width, height)
    mask2 = EmptyMask(width, height)
    assert mask1 != mask2

def test_mask_eq_false_4():
    mask = helper.generate_Mask()
    assert mask != ""
    assert mask != None

def test_empty_mask():
    width = 10
    height = 20
    empty_mask = EmptyMask(width, height)
    assert empty_mask.shape == (height, width)
    assert empty_mask.width == width
    assert empty_mask.height == height
    str(empty_mask)

def test_empty_mask_image_array():
    width = 2
    height = 3
    empty_mask = EmptyMask(width, height)
    assert empty_mask.shape == (height, width)
    assert empty_mask.width == width
    assert empty_mask.height == height
    image_array = empty_mask.image_array
    assert image_array.shape == (height, width)
    assert image_array.dtype == np.uint8
    assert (image_array == 255).all()

def test_empty_mask_width():
    empty_mask = EmptyMask(None, None)
    empty_mask.width = 10
    empty_mask.width = 0
    with pytest.raises(ValueError):
        empty_mask.width = -1

def test_empty_mask_height():
    empty_mask = EmptyMask(None, None)
    empty_mask.height = 5
    empty_mask.height = 0
    with pytest.raises(ValueError):
        empty_mask.height = -1

def test_empty_mask_eq_true_1():
    width = 10
    height = 20
    empty_mask1 = EmptyMask(width, height)
    empty_mask2 = EmptyMask(width, height)
    assert empty_mask1 == empty_mask2
    
def test_empty_mask_eq_true_2():
    width = 10
    height = 20
    mask1 = EmptyMask(width, height)
    mask2 = helper.generate_Mask_Empty(width, height)
    assert mask1 == mask2

def test_empty_mask_eq_false_1():
    width1, height1 = 10, 20
    width2, height2 = 20, 10
    mask1 = EmptyMask(width1, height1)
    mask2 = EmptyMask(width2, height2)
    assert mask1 != mask2

def test_empty_mask_eq_false_2():
    width = 10
    height = 20
    mask1 = EmptyMask(width, height)
    mask2 = helper.generate_Mask(width, height)
    assert mask1 != mask2

def test_empty_mask_eq_false_3():
    width = 10
    height = 20
    mask = EmptyMask(width, height)
    assert mask != ""
    assert mask != None

def test_eq_1():
    mask1 = EmptyMask(10, 20)
    mask2 = EmptyMask(10, 20)
    assert equal(mask1, mask2)

def test_eq_2():
    mask1 = Mask("path")
    mask2 = Mask("path")
    assert equal(mask1, mask2)

def test_eq_3():
    mask1 = helper.generate_Mask(10, 20)
    mask_image_array = mask1.image_array
    mask2 = loader.save_mask_image_array_temporary(mask_image_array)
    assert mask1.path != mask2.path
    assert equal(mask1, mask2)

def test_eq_4():
    mask1 = helper.generate_Mask_Empty(10, 20)
    mask2 = EmptyMask(10, 20)
    assert equal(mask1, mask2)
    assert equal(mask2, mask1)

def test_eq_5():
    mask1 = helper.generate_Mask(10, 20)
    mask_image_array = mask1.image_array
    mask_image_array[1, 2] = 1
    mask_image_array[5, 3] = 2
    mask2 = loader.save_mask_image_array_temporary(mask_image_array)
    assert mask1.path != mask2.path
    assert not equal(mask1, mask2)

def test_eq_6():
    mask1 = helper.generate_Mask(10, 20)
    mask2 = EmptyMask(10, 20)
    assert mask1 != mask2

def test_eq_7():
    mask1 = None
    mask2 = EmptyMask(10, 20)
    with pytest.raises(TypeError):
        equal(mask1, mask2)

def test_eq_8():
    mask1 = helper.generate_Mask(10, 20)
    mask2 = None
    with pytest.raises(TypeError):
        equal(mask1, mask2)
