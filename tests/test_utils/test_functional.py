import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.utils import functional as F
from patchmentation.utils import loader
from patchmentation.utils import validator
from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset
from tests import helper

import numpy as np
import pytest

def test_intersection_1():
    bbox_1 = BBox(1, 0, 6, 3)
    bbox_2 = BBox(3, 1, 7, 6)
    bbox_actual = F.intersection(bbox_1, bbox_2)
    bbox_expected = BBox(3, 1, 6, 3)
    assert bbox_actual == bbox_expected

def test_intersection_2():
    bbox_1 = BBox(1, 0, 9, 4)
    bbox_2 = BBox(3, 1, 7, 6)
    bbox_actual = F.intersection(bbox_1, bbox_2)
    bbox_expected = BBox(3, 1, 7, 4)
    assert bbox_actual == bbox_expected

def test_intersection_3():
    bbox_1 = BBox(5, 4, 6, 5)
    bbox_2 = BBox(3, 1, 7, 6)
    bbox_actual = F.intersection(bbox_1, bbox_2)
    bbox_expected = BBox(5, 4, 6, 5)
    assert bbox_actual == bbox_expected

def test_intersection_4():
    bbox_1 = BBox(3, 1, 7, 6)
    bbox_2 = BBox(3, 1, 7, 6)
    bbox_actual = F.intersection(bbox_1, bbox_2)
    bbox_expected = BBox(3, 1, 7, 6)
    assert bbox_actual == bbox_expected

def test_intersection_5():
    bbox_1 = BBox(5, 1, 6, 7)
    bbox_2 = BBox(3, 4, 7, 6)
    bbox_actual = F.intersection(bbox_1, bbox_2)
    bbox_expected = BBox(5, 4, 6, 6)
    assert bbox_actual == bbox_expected

def test_intersection_6():
    bbox_1 = BBox(5, 6, 6, 7)
    bbox_2 = BBox(3, 4, 7, 6)
    bbox_actual = F.intersection(bbox_1, bbox_2)
    bbox_expected = BBox(5, 6, 6, 6)
    assert bbox_actual == bbox_expected

def test_intersection_7():
    bbox_1 = BBox(8, 6, 9, 7)
    bbox_2 = BBox(3, 4, 7, 6)
    bbox_actual = F.intersection(bbox_1, bbox_2)
    bbox_expected = BBox(0, 0, 0, 0)
    assert bbox_actual == bbox_expected

def test_intersection_8():
    bbox_1 = BBox(8, 3, 9, 4)
    bbox_2 = BBox(3, 4, 7, 6)
    bbox_actual = F.intersection(bbox_1, bbox_2)
    bbox_expected = BBox(0, 0, 0, 0)
    assert bbox_actual == bbox_expected

def test_intersection_over_union_1():
    bbox_1 = BBox(1, 5, 5, 8)
    bbox_2 = BBox(2, 2, 6, 7)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 6 / 26
    assert helper.compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_2():
    bbox_1 = BBox(1, 6, 6, 8)
    bbox_2 = BBox(3, 5, 4, 9)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 2 / 12
    assert helper.compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_3():
    bbox_1 = BBox(1, 4, 6, 8)
    bbox_2 = BBox(3, 5, 4, 7)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 2 / 20
    assert helper.compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_4():
    bbox_1 = BBox(5, 6, 6, 7)
    bbox_2 = BBox(3, 4, 7, 6)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 0
    assert helper.compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_5():
    bbox_1 = BBox(8, 6, 9, 7)
    bbox_2 = BBox(3, 4, 7, 6)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 0
    assert helper.compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_6():
    bbox_1 = BBox(8, 3, 9, 4)
    bbox_2 = BBox(3, 4, 7, 6)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 0
    assert helper.compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_7():
    bbox_1 = BBox(1, 1, 1, 1)
    bbox_2 = BBox(2, 2, 2, 2)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)  
    iou_expected = 0
    assert helper.compare_float_equal(iou_actual, iou_expected)

def test_IOU():
    assert F.IOU == F.intersection_over_union

def test_scale_length_1():
    length = 10
    scale = 0.5
    actual_length = F.scale_length(length, scale)
    expected_length = 5
    assert actual_length == expected_length

def test_scale_length_2():
    length = 10
    scale = 1.0
    actual_length = F.scale_length(length, scale)
    expected_length = 10
    assert actual_length == expected_length

def test_scale_length_3():
    length = 10
    scale = 1.5
    actual_length = F.scale_length(length, scale)
    expected_length = 15
    assert actual_length == expected_length

def test_scale_length_4():
    length = 10
    scale = 0.0
    actual_length = F.scale_length(length, scale)
    expected_length = 0
    assert actual_length == expected_length

def test_scale_dimension_1():
    width = 10
    height = 20
    scale = 1.0
    actual_dimension = F.scale_dimension(width, height, scale)
    expected_dimension = (10, 20)
    assert actual_dimension == expected_dimension

def test_scale_dimension_2():
    width = 10
    height = 20
    scale = 0.5
    actual_dimension = F.scale_dimension(width, height, scale)
    expected_dimension = (5, 10)
    assert actual_dimension == expected_dimension

def test_scale_dimension_3():
    width = 10
    height = 20
    scale = 1.5
    actual_dimension = F.scale_dimension(width, height, scale)
    expected_dimension = (15, 30)
    assert actual_dimension == expected_dimension

def test_scale_dimension_4():
    width = 10
    height = 20
    scale = 0.0
    actual_dimension = F.scale_dimension(width, height, scale)
    expected_dimension = (0, 0)
    assert actual_dimension == expected_dimension

def test_scale_bbox_1():
    bbox = BBox(1, 2, 3, 4)
    scale = 1.0
    actual_bbox = F.scale_bbox(bbox, scale)
    expected_bbox = BBox(1, 2, 3, 4)
    assert actual_bbox == expected_bbox

def test_scale_bbox_2():
    bbox = BBox(1, 2, 11, 32)
    scale = 0.5
    actual_bbox = F.scale_bbox(bbox, scale)
    expected_bbox = BBox(1, 2, 6, 17)
    assert actual_bbox == expected_bbox

def test_scale_bbox_3():
    bbox = BBox(1, 2, 11, 32)
    scale = 1.5
    actual_bbox = F.scale_bbox(bbox, scale)
    expected_bbox = BBox(1, 2, 16, 47)
    assert actual_bbox == expected_bbox

def test_scale_bbox_4():
    bbox = BBox(1, 2, 11, 32)
    scale = 0.0
    actual_bbox = F.scale_bbox(bbox, scale)
    expected_bbox = BBox(1, 2, 1, 2)
    assert actual_bbox == expected_bbox

def test_resize_image_array_1():
    width = 10
    height = 20
    image_array = helper.generate_image_array(width, height)
    expected_width = 5
    expected_height = 7
    actual_image_array = F.resize_image_array(image_array, expected_width, expected_height)
    validator.validate_image_array(actual_image_array, expected_width=expected_width, expected_height=expected_height)

def test_resize_image_array_2():
    width = 10
    height = 20
    image_array = helper.generate_image_array(width, height)
    expected_width = 12
    expected_height = 9
    actual_image_array = F.resize_image_array(image_array, expected_width, expected_height)
    validator.validate_image_array(actual_image_array, expected_width=expected_width, expected_height=expected_height)

@pytest.mark.filterwarnings('error')
def test_display_image_array_Grayscale():
    image_array = helper.generate_mask_image_array(10, 20)
    assert len(image_array.shape) == 2
    F.display_image_array(image_array, block=False)

@pytest.mark.filterwarnings('error')
def test_display_image_array_BGR():
    image_array = helper.generate_image_array(10, 20, channel=3)
    assert image_array.shape[2] == 3
    F.display_image_array(image_array, block=False)

@pytest.mark.filterwarnings('error')
def test_display_image_array_BGRA():
    image_array = helper.generate_image_array(10, 20, channel=4)
    assert image_array.shape[2] == 4
    F.display_image_array(image_array, block=False)

def test_convert_BGR2RGB():
    image_array = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]] 
    ], 'uint8')
    actual_image_array = F.convert_BGR2RGB(image_array)
    expected_imagea_array = np.array([
        [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
        [[12, 11, 10], [15, 14, 13], [18, 17, 16]] 
    ], 'uint8')
    assert (actual_image_array == expected_imagea_array).all()

def test_convert_BGRA2RGBA():
    image_array = np.array([
        [[1, 2, 3, 50], [4, 5, 6, 52], [7, 8, 9, 54]],
        [[10, 11, 12, 51], [13, 14, 15, 53], [16, 17, 18, 55]] 
    ], 'uint8')
    actual_image_array = F.convert_BGRA2RGBA(image_array)
    expected_imagea_array = np.array([
        [[3, 2, 1, 50], [6, 5, 4, 52], [9, 8, 7, 54]],
        [[12, 11, 10, 51], [15, 14, 13, 53], [18, 17, 16, 55]] 
    ], 'uint8')
    assert (actual_image_array == expected_imagea_array).all()

def test_convert_BGR2Grayscale():
    image_array = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]] 
    ], 'uint8')
    actual_image_array = F.convert_BGR2RGB(image_array)
    helper.check_grayscale(actual_image_array)

def test_convert_BGRA2Grayscale():
    image_array = np.array([
        [[1, 2, 3, 50], [4, 5, 6, 52], [7, 8, 9, 54]],
        [[10, 11, 12, 51], [13, 14, 15, 53], [16, 17, 18, 55]] 
    ], 'uint8')
    actual_image_array = F.convert_BGRA2RGBA(image_array)
    grayscale = actual_image_array[:,:,:3]
    alpha = actual_image_array[:,:,3]
    expected_alpha = np.array([
        [50, 52, 54],
        [51, 53, 55] 
    ], 'uint8')
    helper.check_grayscale(grayscale)
    assert (alpha == expected_alpha).all()

def test_crop_image_array_1():
    image_array = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        [[28, 29, 30], [31, 32, 33], [34, 35, 36]]
    ])
    bbox = BBox(0, 1, 2, 2)
    actual_image_array = F.crop_image_array(image_array, bbox)
    expected_image_array = np.array([
        [[10, 11, 12], [13, 14, 15]]
    ])
    assert (actual_image_array == expected_image_array).all()

def test_crop_image_array_2():
    image_array = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        [[28, 29, 30], [31, 32, 33], [34, 35, 36]]
    ])
    bbox = BBox(1, 0, 2, 3)
    actual_image_array = F.crop_image_array(image_array, bbox)
    expected_image_array = np.array([
        [[4, 5, 6]],
        [[13, 14, 15]],
        [[22, 23, 24]]
    ])
    assert (actual_image_array == expected_image_array).all()

def test_gaussian_kernel_2d_1():
    kernel_size = 5
    sigma = 1.0
    expected_kernel = np.array([
        [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
        [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
        [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
        [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
        [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]
    ])
    actual_kernel = F.gaussian_kernel_2d(kernel_size, sigma)
    assert np.allclose(expected_kernel, actual_kernel)

def test_gaussian_kernel_2d_2():
    kernel_size = 3
    sigma = 0.5
    expected_kernel = np.array([
        [0.01134374, 0.08381951, 0.01134374],
        [0.08381951, 0.61934703, 0.08381951],
        [0.01134374, 0.08381951, 0.01134374]
    ])
    actual_kernel = F.gaussian_kernel_2d(kernel_size, sigma)
    assert np.allclose(expected_kernel, actual_kernel)

def test_overlay_image_1():
    image_a_image_array = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        [[28, 29, 30], [31, 32, 33], [34, 35, 36]]
    ])
    image_b_image_array = np.array([
        [[101, 102, 103], [104, 105, 106]],
        [[107, 108, 109], [110, 111, 112]],
        [[113, 114, 115], [116, 117, 118]]
    ])
    bbox = BBox(1, 0, 3, 3)
    image_a = loader.save_image_array_temporary(image_a_image_array)
    image_b = loader.save_image_array_temporary(image_b_image_array)
    expected_image_array = np.array([
        [[1, 2, 3, 255], [101, 102, 103, 255], [104, 105, 106, 255]],
        [[10, 11, 12, 255], [107, 108, 109, 255], [110, 111, 112, 255]],
        [[19, 20, 21, 255], [113, 114, 115, 255], [116, 117, 118, 255]],
        [[28, 29, 30, 255], [31, 32, 33, 255], [34, 35, 36, 255]]
    ])
    actual_image_array = F.overlay_image(image_a, image_b, bbox).image_array()
    assert (expected_image_array == actual_image_array).all()

def test_overlay_image_2():
    image_a_image_array = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        [[28, 29, 30], [31, 32, 33], [34, 35, 36]]
    ])
    image_b_image_array = np.array([
        [[101, 102, 103], [104, 105, 106]],
        [[107, 108, 109], [110, 111, 112]],
        [[113, 114, 115], [116, 117, 118]]
    ])
    bbox = BBox(0, 1, 2, 4)
    image_a = loader.save_image_array_temporary(image_a_image_array)
    image_b = loader.save_image_array_temporary(image_b_image_array)
    expected_image_array = np.array([
        [[1, 2, 3, 255], [4, 5, 6, 255], [7, 8, 9, 255]],
        [[101, 102, 103, 255], [104, 105, 106, 255], [16, 17, 18, 255]],
        [[107, 108, 109, 255], [110, 111, 112, 255], [25, 26, 27, 255]],
        [[113, 114, 115, 255], [116, 117, 118, 255], [34, 35, 36, 255]]
    ])
    actual_image_array = F.overlay_image(image_a, image_b, bbox).image_array()
    assert (expected_image_array == actual_image_array).all()

def test_overlay_image_3():
    image_a_image_array = np.array([
        [[1, 2, 3, 255], [4, 5, 6, 255], [7, 8, 9, 255]],
        [[10, 11, 12, 255], [13, 14, 15, 255], [16, 17, 18, 255]],
        [[19, 20, 21, 255], [22, 23, 24, 255], [25, 26, 27, 255]],
        [[28, 29, 30, 255], [31, 32, 33, 255], [34, 35, 36, 255]]
    ])
    image_b_image_array = np.array([
        [[101, 102, 103, 0], [104, 105, 106, 50]],
        [[107, 108, 109, 100], [110, 111, 112, 150]],
        [[113, 114, 115, 200], [116, 117, 118, 255]]
    ])
    bbox = BBox(1, 1, 3, 4)
    image_a = loader.save_image_array_temporary(image_a_image_array)
    image_b = loader.save_image_array_temporary(image_b_image_array)
    expected_image_array = np.array([
        [[1, 2, 3, 255], [4, 5, 6, 255], [7, 8, 9, 255]],
        [[10, 11, 12, 255], [13, 14, 15, 255], [33, 34, 35, 255]],
        [[19, 20, 21, 255], [55, 56, 57, 255], [75, 76, 77, 255]],
        [[28, 29, 30, 255], [95, 96, 97, 255], [116, 117, 118, 255]]
    ])
    actual_image_array = F.overlay_image(image_a, image_b, bbox).image_array()
    assert (expected_image_array == actual_image_array).all()

def test_overlay_image_4():
    image_a_image_array = np.array([
        [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]],
        [[10, 11, 12, 50], [13, 14, 15, 50], [16, 17, 18, 50]],
        [[19, 20, 21, 100], [22, 23, 24, 100], [25, 26, 27, 100]],
        [[28, 29, 30, 150], [31, 32, 33, 150], [34, 35, 36, 150]]
    ])
    image_b_image_array = np.array([
        [[101, 102, 103, 0], [104, 105, 106, 50]],
        [[107, 108, 109, 100], [110, 111, 112, 150]],
        [[113, 114, 115, 200], [116, 117, 118, 255]]
    ])
    bbox = BBox(0, 0, 2, 3)
    image_a = loader.save_image_array_temporary(image_a_image_array)
    image_b = loader.save_image_array_temporary(image_b_image_array)
    expected_image_array = np.array([
        [[0, 0, 0, 0], [20, 20, 20, 50], [7, 8, 9, 0]],
        [[43, 43, 44, 130], [65, 66, 67, 170], [16, 17, 18, 50]],
        [[90, 91, 91, 221], [116, 117, 118, 255], [25, 26, 27, 100]],
        [[28, 29, 30, 150], [31, 32, 33, 150], [34, 35, 36, 150]]
    ])
    actual_image_array = F.overlay_image(image_a, image_b, bbox).image_array()
    assert (expected_image_array == actual_image_array).all()

def test_visibility_thresholding_1():
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 0, 5, 9)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.5
    list_non_removal_patch_bbox = None
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_3]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_2():
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 0, 5, 9)
    list_patch_bbox = [patch_bbox_1, patch_bbox_3, patch_bbox_2]
    visibility_threshold = 0.5
    list_non_removal_patch_bbox = None
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_2]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_3():
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(0, 0, 5, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.5
    list_non_removal_patch_bbox = None
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_4():
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(0, 0, 5, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.4
    list_non_removal_patch_bbox = None
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_5():
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(0, 0, 5, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    patch_bbox_4 = helper.generate_Patch(image), BBox(0, 0, 5, 4)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.4
    list_non_removal_patch_bbox = [patch_bbox_4]
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_2]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_6():
    INF = float('inf')
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(0, 0, 5, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    patch_bbox_4 = helper.generate_Patch(image), BBox(0, 0, 5, 4)
    patch_bbox_5 = helper.generate_Patch(image), BBox(-INF, 5, INF, 6)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.4
    list_non_removal_patch_bbox = [patch_bbox_4, patch_bbox_5]
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_2]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_7():
    INF = float('inf')
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(0, 0, 5, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    patch_bbox_4 = helper.generate_Patch(image), BBox(0, 0, 5, 4)
    patch_bbox_5 = helper.generate_Patch(image), BBox(-INF, 5, INF, INF)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.4
    list_non_removal_patch_bbox = [patch_bbox_4, patch_bbox_5]
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = []
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_8():
    INF = float('inf')
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    patch_bbox_2 = helper.generate_Patch(image), BBox(5, 5, 10, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 5, 5, 15)
    patch_bbox_4 = helper.generate_Patch(image), BBox(-5, -10, 5, 10)
    patch_bbox_5 = helper.generate_Patch(image), BBox(-INF, 3, INF, 5)
    patch_bbox_6 = helper.generate_Patch(image), BBox(INF, 0, INF, INF)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.4
    list_non_removal_patch_bbox = [patch_bbox_4, patch_bbox_5, patch_bbox_6]
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_2, patch_bbox_3]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_9():
    INF = float('inf')
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    patch_bbox_2 = helper.generate_Patch(image), BBox(5, 5, 10, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 5, 5, 15)
    patch_bbox_4 = helper.generate_Patch(image), BBox(-5, -10, 5, 10)
    patch_bbox_5 = helper.generate_Patch(image), BBox(-INF, 3, INF, 5)
    patch_bbox_6 = helper.generate_Patch(image), BBox(INF, 0, INF, INF)
    list_patch_bbox = []
    visibility_threshold = 0.4
    list_non_removal_patch_bbox = [patch_bbox_4, patch_bbox_5, patch_bbox_6]
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = []
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_10():
    INF = float('inf')
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 5, 5)
    patch_bbox_2 = helper.generate_Patch(image), BBox(5, 5, 10, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(0, 5, 5, 15)
    patch_bbox_4 = helper.generate_Patch(image), BBox(-5, -10, 5, 10)
    patch_bbox_5 = helper.generate_Patch(image), BBox(-INF, 3, INF, 5)
    patch_bbox_6 = helper.generate_Patch(image), BBox(INF, 0, INF, INF)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.4
    list_non_removal_patch_bbox = []
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_11():
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(10, 0, 20, 5)
    patch_bbox_3 = helper.generate_Patch(image), BBox(10, 5, 20, 10)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 1.0
    list_non_removal_patch_bbox = None
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_visibility_thresholding_12():
    image = helper.generate_Image()
    patch_bbox_1 = helper.generate_Patch(image), BBox(0, 0, 10, 10)
    patch_bbox_2 = helper.generate_Patch(image), BBox(0, 0, 5, 10)
    patch_bbox_3 = helper.generate_Patch(image), BBox(5, 0, 100, 10)
    list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    visibility_threshold = 0.0
    list_non_removal_patch_bbox = None
    actual_list_patch_bbox = F.visibility_thresholding(list_patch_bbox, visibility_threshold, list_non_removal_patch_bbox)
    expected_list_patch_bbox = [patch_bbox_1, patch_bbox_2, patch_bbox_3]
    assert actual_list_patch_bbox == expected_list_patch_bbox

def test_get_negative_patch():
    image_patch = helper.generate_ImagePatch()
    iou_threshold = 0.5
    negative_patch = F.get_negative_patch(image_patch, iou_threshold)
    for patch in image_patch.patches:
        assert F.intersection_over_union(patch.bbox, negative_patch.bbox) < iou_threshold
    assert negative_patch.class_name == F.NEGATIVE_PATCH_CLASS_NAME
