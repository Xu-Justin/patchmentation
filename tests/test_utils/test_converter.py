import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.utils import converter
from tests import helper

def test_array2image():
    image_array = helper.generate_image_array()
    image = converter.array2image(image_array)
    assert (image.image_array() == image_array).all()

def test_image2patch():
    image = helper.generate_Image()
    class_name = helper.generate_class_name()
    patch = converter.image2patch(image, class_name)
    assert patch.class_name == class_name
    assert patch.image is image
    assert (patch.image_array() == image.image_array()).all()

def test_array2patch():
    image_array = helper.generate_image_array()
    class_name = helper.generate_class_name()
    patch = converter.array2patch(image_array, class_name)
    assert patch.class_name == class_name
    assert (patch.image_array() == image_array).all()
