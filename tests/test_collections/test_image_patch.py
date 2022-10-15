import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import ImagePatch
from patchmentation.utils import validator
from tests import helper

def test_image_patch():
    image = helper.generate_Image()
    patches = helper.generate_patches(image)
    image_patch = ImagePatch(image, patches)
    assert image_patch.image is image
    assert image_patch.patches is patches
    assert (image, patches) == tuple(image_patch)
    image_patch.summary()

def test_image_patch_image_array():
    image_patch = helper.generate_ImagePatch()
    image_array = image_patch.image_array()
    validator.validate_image_array(image_array, check_shape=True)
