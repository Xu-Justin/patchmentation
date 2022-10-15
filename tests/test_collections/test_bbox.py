import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import BBox
from tests import helper

def test_bbox():
    xmin, xmax = helper.generate_x()
    ymin, ymax = helper.generate_y()
    bbox = BBox(xmin, ymin, xmax, ymax)
    assert bbox.xmin == xmin
    assert bbox.ymin == ymin
    assert bbox.xmax == xmax
    assert bbox.ymax == ymax
    assert (xmin, ymin, xmax, ymax) == tuple(bbox)
    bbox.summary()
    
def test_bbox_width_1():
    xmin = 4
    xmax = 10
    bbox = BBox(xmin, None, xmax, None)
    actual_width = bbox.width()
    expected_width = 6
    assert actual_width == expected_width

def test_bbox_width_2():
    xmin = 4
    xmax = 4
    bbox = BBox(xmin, None, xmax, None)
    actual_width = bbox.width()
    expected_width = 0
    assert actual_width == expected_width


def test_bbox_height_2():
    ymin = 4
    ymax = 4
    bbox = BBox(None, ymin, None, ymax)
    actual_height = bbox.height()
    expected_height = 0
    assert actual_height == expected_height

def test_bbox_area_1():
    xmin, xmax = 4, 10
    ymin, ymax = 6, 8
    bbox = BBox(xmin, ymin, xmax, ymax)
    actual_area = bbox.area()
    expected_area = 12
    assert actual_area == expected_area

def test_bbox_area_2():
    xmin, xmax = 4, 4
    ymin, ymax = 6, 8
    bbox = BBox(xmin, ymin, xmax, ymax)
    actual_area = bbox.area()
    expected_area = 0
    assert actual_area == expected_area

def test_bbox_area_3():
    xmin, xmax = 4, 10
    ymin, ymax = 8, 8
    bbox = BBox(xmin, ymin, xmax, ymax)
    actual_area = bbox.area()
    expected_area = 0
    assert actual_area == expected_area

def test_bbox_area_4():
    xmin, xmax = 10, 10
    ymin, ymax = 6, 6
    bbox = BBox(xmin, ymin, xmax, ymax)
    actual_area = bbox.area()
    expected_area = 0
    assert actual_area == expected_area
