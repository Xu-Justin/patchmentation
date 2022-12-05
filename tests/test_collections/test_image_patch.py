import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import ImagePatch
from tests import helper

import numpy as np
import pytest

def test_image_patch():
    image = helper.generate_Image()
    patches = helper.generate_patches(image)
    image_patch = ImagePatch(image, patches)
    assert image_patch.image is image
    assert image_patch.patches is patches
    assert (image, patches) == tuple(image_patch)
    assert image_patch.shape == image.shape
    assert image_patch.width == image.width
    assert image_patch.height == image.height
    assert image_patch.channel == image.channel
    assert image_patch.n_patches == len(patches)
    str(image_patch)

def test_image_patch_image_array():
    image_patch = helper.generate_ImagePatch()
    image_array = image_patch.image_array()
    assert image_array.shape == (image_patch.height, image_patch.width, 4)
    assert image_array.dtype == np.uint8

def test_image_patch_image_array_with_classes():
    classes = helper.generate_classes(5)
    image_patch = helper.generate_ImagePatch(classes)
    image_array = image_patch.image_array(classes[:3])
    assert image_array.shape == (image_patch.height, image_patch.width, 4)
    assert image_array.dtype == np.uint8

def test_image_patch_patches_none():
    image = helper.generate_Image()
    image_patch = ImagePatch(image, None)
    assert image_patch.image is image
    assert image_patch.patches == []

def test_image_patch_patches_error():
    image1 = helper.generate_Image(10, 20)
    patches1 = helper.generate_patches(image1, number_of_patch=2)
    image2 = helper.generate_Image(20, 10)
    patches2 = helper.generate_patches(image2, number_of_patch=2)
    assert image1.path != image2.path
    with pytest.raises(ValueError):
        image_patch = ImagePatch(image1, patches1 + patches2)
    