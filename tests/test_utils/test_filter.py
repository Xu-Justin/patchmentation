import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests import helper
from patchmentation.utils import filter
from patchmentation.utils import Comparator

def test_filter_width_1():
    width = 20
    comparator = Comparator.GreaterThan
    filter_width = filter.FilterWidth(width, comparator)

    list_image_array = [
        helper.generate_image_array(10, 20),
        helper.generate_image_array(20, 30),
        helper.generate_image_array(30, 40),
        helper.generate_image_array(30, 10),
        helper.generate_image_array(25, 25),
    ]
    actual_filtered_image_array = filter_width(list_image_array)
    expected_filtered_image_array = [
        list_image_array[2],
        list_image_array[3],
        list_image_array[4],
    ]
    assert helper.compare_unordered_list_equal(actual_filtered_image_array, expected_filtered_image_array)

def test_filter_width_2():
    width = 20
    comparator = Comparator.LessEqual
    filter_width = filter.FilterWidth(width, comparator)

    list_image_array = [
        helper.generate_image_array(10, 20),
        helper.generate_image_array(20, 30),
        helper.generate_image_array(30, 40),
        helper.generate_image_array(30, 10),
        helper.generate_image_array(25, 25),
    ]
    actual_filtered_image_array = filter_width(list_image_array)
    expected_filtered_image_array = [
        list_image_array[0],
        list_image_array[1],
    ]
    assert helper.compare_unordered_list_equal(actual_filtered_image_array, expected_filtered_image_array)

def test_filter_height_1():
    height = 20
    comparator = Comparator.GreaterEqual
    filter_height= filter.FilterHeight(height, comparator)

    list_image_array = [
        helper.generate_image_array(10, 20),
        helper.generate_image_array(20, 30),
        helper.generate_image_array(30, 40),
        helper.generate_image_array(30, 10),
        helper.generate_image_array(25, 25),
    ]
    actual_filtered_image_array = filter_height(list_image_array)
    expected_filtered_image_array = [
        list_image_array[0],
        list_image_array[1],
        list_image_array[2],
        list_image_array[4],
    ]
    assert helper.compare_unordered_list_equal(actual_filtered_image_array, expected_filtered_image_array)

def test_filter_height_2():
    height = 20
    comparator = Comparator.Equal
    filter_height= filter.FilterHeight(height, comparator)

    list_image_array = [
        helper.generate_image_array(10, 20),
        helper.generate_image_array(20, 30),
        helper.generate_image_array(30, 40),
        helper.generate_image_array(30, 10),
        helper.generate_image_array(25, 25),
    ]
    actual_filtered_image_array = filter_height(list_image_array)
    expected_filtered_image_array = [
        list_image_array[0]
    ]
    assert helper.compare_unordered_list_equal(actual_filtered_image_array, expected_filtered_image_array)
