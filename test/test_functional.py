import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.utils import functional as F
from patchmentation.utils import validator
from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset
from test import helper

import numpy as np
import pytest

def test_calculcate_width_1():
    xmin, xmax = 1, 5
    ymin, ymax = None, None
    bbox = BBox(xmin, ymin, xmax, ymax)
    width = F.calculate_width(bbox)
    assert width == 4

def test_calculcate_width_2():
    xmin, xmax = 1, 1
    ymin, ymax = None, None
    bbox = BBox(xmin, ymin, xmax, ymax)
    width = F.calculate_width(bbox)
    assert width == 0

def test_calculate_height_1():
    xmin, xmax = None, None
    ymin, ymax = 1, 5
    bbox = BBox(xmin, ymin, xmax, ymax)
    height = F.calculate_height(bbox)
    assert height == 4

def test_calculate_height_2():
    xmin, xmax = None, None
    ymin, ymax = 1, 1
    bbox = BBox(xmin, ymin, xmax, ymax)
    height = F.calculate_height(bbox)
    assert height == 0

def test_calculate_area_1():
    xmin, xmax = 1, 5
    ymin, ymax = 6, 8
    bbox = BBox(xmin, ymin, xmax, ymax)
    area = F.calculate_area(bbox)
    assert area == 8

def test_calculate_area_2():
    xmin, xmax = 1, 1
    ymin, ymax = 6, 8
    bbox = BBox(xmin, ymin, xmax, ymax)
    area = F.calculate_area(bbox)
    assert area == 0

def test_calculate_area_3():
    xmin, xmax = 1, 5
    ymin, ymax = 6, 6
    bbox = BBox(xmin, ymin, xmax, ymax)
    area = F.calculate_area(bbox)
    assert area == 0

def test_calculate_area_4():
    xmin, xmax = 5, 5
    ymin, ymax = 6, 6
    bbox = BBox(xmin, ymin, xmax, ymax)
    area = F.calculate_area(bbox)
    assert area == 0

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

def test_visibility_suppression_1():
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 5), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 9), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.5
    non_removal_patches = None
    attr_bbox = 'bbox'
    attr_non_removal_patches_bbox = None
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_1, patch_3]
    assert actual_patches == expected_patches

def test_visibility_suppression_2():
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 5), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 9), None)
    patches = [patch_1, patch_3, patch_2]
    visibility_threshold = 0.5
    non_removal_patches = None
    attr_bbox = 'bbox'
    attr_non_removal_patches_bbox = None
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_1, patch_2]
    assert actual_patches == expected_patches

def test_visibility_suppression_3():
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 10), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 5), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.5
    non_removal_patches = None
    attr_bbox = 'bbox'
    attr_non_removal_patches_bbox = None
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_3]
    assert actual_patches == expected_patches

def test_visibility_suppression_4():
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 10), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 5), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.4
    non_removal_patches = None
    attr_bbox = 'bbox'
    attr_non_removal_patches_bbox = None
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_1, patch_2, patch_3]
    assert actual_patches == expected_patches

def test_visibility_suppression_5():
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 10), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 5), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.5
    non_removal_patches = None
    attr_bbox = 'attr_bbox'
    setattr(patch_1, attr_bbox, BBox(0, 0, 10, 10))
    setattr(patch_2, attr_bbox, BBox(0, 0, 5, 9))
    setattr(patch_3, attr_bbox, BBox(0, 0, 5, 5))
    attr_non_removal_patches_bbox = None
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_1, patch_3]
    assert actual_patches == expected_patches

def test_visibility_suppression_6():
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 10), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 5), None)
    patch_4 = Patch(None, BBox(0, 0, 5, 4), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.4
    non_removal_patches = [patch_4]
    attr_bbox = 'bbox'
    attr_non_removal_patches_bbox = 'bbox'
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_1, patch_2]
    assert actual_patches == expected_patches

def test_visibility_suppression_7():
    INF = float('inf')
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 10), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 5), None)
    patch_4 = Patch(None, BBox(0, 0, 5, 4), None)
    patch_5 = Patch(None, BBox(-INF, 5, INF, 6), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.4
    non_removal_patches = [patch_4, patch_5]
    attr_bbox = 'bbox'
    attr_non_removal_patches_bbox = 'bbox'
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_1]
    assert actual_patches == expected_patches

def test_visibility_suppression_8():
    INF = float('inf')
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 10), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 5), None)
    patch_4 = Patch(None, BBox(0, 0, 5, 4), None)
    patch_5 = Patch(None, BBox(-INF, 5, INF, INF), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.4
    non_removal_patches = [patch_4, patch_5]
    attr_bbox = 'bbox'
    attr_non_removal_patches_bbox = 'bbox'
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = []
    assert actual_patches == expected_patches

def test_visibility_suppression_9():
    INF = float('inf')
    patch_1 = Patch(None, BBox(0, 0, 10, 10), None)
    patch_2 = Patch(None, BBox(0, 0, 5, 10), None)
    patch_3 = Patch(None, BBox(0, 0, 5, 5), None)
    patch_4 = Patch(None, BBox(0, 0, 5, 4), None)
    patch_5 = Patch(None, BBox(-INF, 5, INF, 6), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.4
    non_removal_patches = [patch_4, patch_5]
    attr_bbox = 'attr_bbox'
    setattr(patch_1, attr_bbox, BBox(0, 0, 10, 10))
    setattr(patch_2, attr_bbox, BBox(0, 0, 5, 9))
    setattr(patch_3, attr_bbox, BBox(0, 0, 5, 5))
    attr_non_removal_patches_bbox = 'attr_non_removal_patches_bbox'
    setattr(patch_4, attr_non_removal_patches_bbox, BBox(-INF, -INF, INF, 0))
    setattr(patch_5, attr_non_removal_patches_bbox, BBox(-INF, 5, INF, INF))
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_3]
    assert actual_patches == expected_patches

def test_visibility_suppression_10():
    INF = float('inf')
    patch_1 = Patch(None, BBox(0, 0, 5, 5), None)
    patch_2 = Patch(None, BBox(5, 5, 10, 10), None)
    patch_3 = Patch(None, BBox(0, 5, 5, 15), None)
    patch_4 = Patch(None, BBox(-5, -10, 5, 10), None)
    patch_5 = Patch(None, BBox(-INF, 3, INF, 5), None)
    patch_6 = Patch(None, BBox(INF, 0, INF, INF), None)
    patches = [patch_1, patch_2, patch_3]
    visibility_threshold = 0.4
    non_removal_patches = [patch_4, patch_5, patch_6]
    attr_bbox = 'bbox'
    attr_non_removal_patches_bbox = 'bbox'
    actual_patches = F.visibility_suppression(patches, visibility_threshold, non_removal_patches, attr_bbox, attr_non_removal_patches_bbox)
    expected_patches = [patch_2, patch_3]
    assert actual_patches == expected_patches

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

def test_place_image_array_1():
    patch_array = np.array([
        [[1, 11], [2, 22]],
        [[33, 3], [44, 4]],
        [[5, 55], [6, 66]]
    ])
    image_array = np.array([
        [[-1, -11], [-2, -22], [-3, -33]],
        [[-44, -4], [-55, -5], [-66, -6]],
        [[-77, -7], [-88, -8], [-99, -9]],
        [[-10, -100], [-11, -110], [-12, -120]]
    ])
    bbox = BBox(1, 0, 3, 3)
    actual_bbox = F.place_image_array(patch_array, image_array, bbox)
    expected_image_array = np.array([
        [[-1, -11], [1, 11], [2, 22]],
        [[-44, -4], [33, 3], [44, 4]],
        [[-77, -7], [5, 55], [6, 66]],
        [[-10, -100], [-11, -110], [-12, -120]]
    ])
    expected_bbox = BBox(1, 0, 3, 3)
    assert (image_array == expected_image_array).all()
    assert actual_bbox == expected_bbox

def test_place_image_array_2():
    patch_array = np.array([
        [[1, 11], [2, 22]],
        [[33, 3], [44, 4]],
        [[5, 55], [6, 66]]
    ])
    image_array = np.array([
        [[-1, -11], [-2, -22], [-3, -33]],
        [[-44, -4], [-55, -5], [-66, -6]],
        [[-77, -7], [-88, -8], [-99, -9]],
        [[-10, -100], [-11, -110], [-12, -120]]
    ])
    bbox = BBox(2, 2, 4, 5)
    actual_bbox = F.place_image_array(patch_array, image_array, bbox)
    expected_image_array = np.array([
        [[-1, -11], [-2, -22], [-3, -33]],
        [[-44, -4], [-55, -5], [-66, -6]],
        [[-77, -7], [-88, -8], [1, 11]],
        [[-10, -100], [-11, -110], [33, 3]]
    ])
    expected_bbox = BBox(2, 2, 3, 4)
    assert (image_array == expected_image_array).all()
    assert actual_bbox == expected_bbox

def test_place_image_array_3():
    patch_array = np.array([
        [[1, 11], [2, 22]],
        [[33, 3], [44, 4]],
        [[5, 55], [6, 66]]
    ])
    image_array = np.array([
        [[-1, -11], [-2, -22], [-3, -33]],
        [[-44, -4], [-55, -5], [-66, -6]],
        [[-77, -7], [-88, -8], [-99, -9]],
        [[-10, -100], [-11, -110], [-12, -120]]
    ])
    bbox = BBox(-1, -1, 1, 2)
    actual_bbox = F.place_image_array(patch_array, image_array, bbox)
    expected_image_array = np.array([
        [[44, 4], [-2, -22], [-3, -33]],
        [[6, 66], [-55, -5], [-66, -6]],
        [[-77, -7], [-88, -8], [-99, -9]],
        [[-10, -100], [-11, -110], [-12, -120]]
    ])
    expected_bbox = BBox(0, 0, 1, 2)
    assert (image_array == expected_image_array).all()
    assert actual_bbox == expected_bbox

@pytest.mark.filterwarnings('error')
def test_display_image_array():
    image_array = helper.generate_image_array(10, 20)
    F.display_image_array(image_array, block=False)
