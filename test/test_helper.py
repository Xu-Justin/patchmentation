import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test import helper
from patchmentation.utils import validator

def test_generate_bbox_1():
    width = 20
    height = 1000
    bbox = helper.generate_bbox(width, height)
    validator.validate_BBox(bbox, width, height)

def test_generate_bbox_2():
    width = 1000
    height = 20
    bbox = helper.generate_bbox(width, height)
    validator.validate_BBox(bbox, width, height)

def test_generate_image_1():
    width = 100
    height = 20
    image = helper.generate_image(width, height)
    validator.validate_Image(image, expected_width=width, expected_height=height)

def test_generate_image_2():
    width = 20
    height = 100
    image = helper.generate_image(width, height)
    validator.validate_Image(image, expected_width=width, expected_height=height)

def test_generate_patch():
    image = helper.generate_image()
    patch = helper.generate_patch(image)
    validator.validate_Patch(patch)

def test_generate_imagePatch():
    imagePatch = helper.generate_imagePatch()
    validator.validate_ImagePatch(imagePatch)

def test_generate_dataset():
    dataset = helper.generate_dataset()
    validator.validate_Dataset(dataset)
