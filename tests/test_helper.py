import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests import helper
from patchmentation.utils import validator

def test_generate_bbox_1():
    width = 20
    height = 1000
    bbox = helper.generate_BBox(width, height)
    validator.validate_BBox(bbox, width=width, height=height)

def test_generate_bbox_2():
    width = 1000
    height = 20
    bbox = helper.generate_BBox(width, height)
    validator.validate_BBox(bbox, width=width, height=height)

def test_generate_image_1():
    width = 100
    height = 20
    image = helper.generate_Image(width, height)
    validator.validate_Image(image, expected_width=width, expected_height=height)

def test_generate_image_2():
    width = 20
    height = 100
    image = helper.generate_Image(width, height)
    validator.validate_Image(image, expected_width=width, expected_height=height)

def test_generate_patch():
    image = helper.generate_Image()
    patch = helper.generate_Patch(image)
    validator.validate_Patch(patch)

def test_generate_imagePatch():
    imagePatch = helper.generate_ImagePatch()
    validator.validate_ImagePatch(imagePatch)

def test_generate_dataset():
    dataset = helper.generate_Dataset()
    validator.validate_Dataset(dataset)

def test_compare_float_equal_1():
    float_1 = 0.1 + 0.2
    float_2 = 0.3
    assert helper.compare_float_equal(float_1, float_2)

def test_compare_float_equal_2():
    float_1 = 0.1 + 0.2
    float_2 = 0.30001
    assert not helper.compare_float_equal(float_1, float_2)

def test_compare_float_1():
    float_1 = 0.1 + 0.2
    float_2 = 0.3
    assert helper.compare_float(float_1, float_2) == 0

def test_compare_float_2():
    float_1 = 0.1 + 0.2
    float_2 = 0.30001
    assert helper.compare_float(float_1, float_2) == -1

def test_compare_float_3():
    float_1 = 0.1 + 0.2
    float_2 = 0.29999
    assert helper.compare_float(float_1, float_2) == 1

def test_compare_unordered_list_equal_1():
    obj_1 = object()
    obj_2 = object()
    obj_3 = object()
    list_1 = [obj_1, obj_2, obj_3, obj_1]
    list_2 = [obj_3, obj_1, obj_1, obj_2]
    assert helper.compare_unordered_list_equal(list_1, list_2)
    assert list_1 == [obj_1, obj_2, obj_3, obj_1]
    assert list_2 == [obj_3, obj_1, obj_1, obj_2]

def test_compare_unordered_list_equal_2():
    obj_1 = object()
    obj_2 = object()
    obj_3 = object()
    list_1 = [obj_1, obj_2, obj_3, obj_1]
    list_2 = [obj_3, obj_1, obj_2, obj_2]
    assert not helper.compare_unordered_list_equal(list_1, list_2)
    assert list_1 == [obj_1, obj_2, obj_3, obj_1]
    assert list_2 == [obj_3, obj_1, obj_2, obj_2]

def test_compare_unordered_list_equal_3():
    obj_1 = object()
    obj_2 = object()
    obj_3 = object()
    list_1 = [obj_1, obj_2, obj_3]
    list_2 = [obj_3, obj_1, obj_3, obj_2]
    assert not helper.compare_unordered_list_equal(list_1, list_2)
    assert list_1 == [obj_1, obj_2, obj_3]
    assert list_2 == [obj_3, obj_1, obj_3, obj_2]

if __name__ == '__main__':
    test_compare_unordered_list_equal_1()