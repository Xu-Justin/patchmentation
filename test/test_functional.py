import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.utils import functional as F
from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset
from test.helper import compare_float_equal, compare_unordered_list_equal
from test.helper import generate_classes, generate_patches

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
    assert compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_2():
    bbox_1 = BBox(1, 6, 6, 8)
    bbox_2 = BBox(3, 5, 4, 9)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 2 / 12
    assert compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_3():
    bbox_1 = BBox(1, 4, 6, 8)
    bbox_2 = BBox(3, 5, 4, 7)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 2 / 20
    assert compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_4():
    bbox_1 = BBox(5, 6, 6, 7)
    bbox_2 = BBox(3, 4, 7, 6)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 0
    assert compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_5():
    bbox_1 = BBox(8, 6, 9, 7)
    bbox_2 = BBox(3, 4, 7, 6)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 0
    assert compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_6():
    bbox_1 = BBox(8, 3, 9, 4)
    bbox_2 = BBox(3, 4, 7, 6)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)
    iou_expected = 0
    assert compare_float_equal(iou_actual, iou_expected)

def test_intersection_over_union_7():
    bbox_1 = BBox(1, 1, 1, 1)
    bbox_2 = BBox(2, 2, 2, 2)
    iou_actual = F.intersection_over_union(bbox_1, bbox_2)  
    iou_expected = 0
    assert compare_float_equal(iou_actual, iou_expected)

def test_IOU():
    assert F.IOU == F.intersection_over_union
