import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests import helper
from patchmentation.utils import filter
from patchmentation.utils import Comparator
from patchmentation.utils import converter

def test_filter_width_1():
    width = 20
    comparator = Comparator.GreaterThan
    filter_width = filter.FilterWidth(width, comparator)

    patches = [
        converter.array2patch(helper.generate_image_array(10, 20)),
        converter.array2patch(helper.generate_image_array(20, 30)),
        converter.array2patch(helper.generate_image_array(30, 40)),
        converter.array2patch(helper.generate_image_array(30, 10)),
        converter.array2patch(helper.generate_image_array(25, 25)),
    ]
    actual_filtered_patches = filter_width(patches)
    expected_filtered_patches = [
        patches[2],
        patches[3],
        patches[4],
    ]
    assert helper.compare_unordered_list_equal(actual_filtered_patches, expected_filtered_patches)

def test_filter_width_2():
    width = 20
    comparator = Comparator.LessEqual
    filter_width = filter.FilterWidth(width, comparator)

    patches = [
        converter.array2patch(helper.generate_image_array(10, 20)),
        converter.array2patch(helper.generate_image_array(20, 30)),
        converter.array2patch(helper.generate_image_array(30, 40)),
        converter.array2patch(helper.generate_image_array(30, 10)),
        converter.array2patch(helper.generate_image_array(25, 25)),
    ]
    actual_filtered_patches = filter_width(patches)
    expected_filtered_patches = [
        patches[0],
        patches[1],
    ]
    assert helper.compare_unordered_list_equal(actual_filtered_patches, expected_filtered_patches)

def test_filter_height_1():
    height = 20
    comparator = Comparator.GreaterEqual
    filter_height= filter.FilterHeight(height, comparator)

    patches = [
        converter.array2patch(helper.generate_image_array(10, 20)),
        converter.array2patch(helper.generate_image_array(20, 30)),
        converter.array2patch(helper.generate_image_array(30, 40)),
        converter.array2patch(helper.generate_image_array(30, 10)),
        converter.array2patch(helper.generate_image_array(25, 25)),
    ]
    actual_filtered_patches = filter_height(patches)
    expected_filtered_patches = [
        patches[0],
        patches[1],
        patches[2],
        patches[4],
    ]
    assert helper.compare_unordered_list_equal(actual_filtered_patches, expected_filtered_patches)

def test_filter_height_2():
    height = 20
    comparator = Comparator.Equal
    filter_height= filter.FilterHeight(height, comparator)

    patches = [
        converter.array2patch(helper.generate_image_array(10, 20)),
        converter.array2patch(helper.generate_image_array(20, 30)),
        converter.array2patch(helper.generate_image_array(30, 40)),
        converter.array2patch(helper.generate_image_array(30, 10)),
        converter.array2patch(helper.generate_image_array(25, 25)),
    ]
    actual_filtered_patches = filter_height(patches)
    expected_filtered_patches = [
        patches[0]
    ]
    assert helper.compare_unordered_list_equal(actual_filtered_patches, expected_filtered_patches)
