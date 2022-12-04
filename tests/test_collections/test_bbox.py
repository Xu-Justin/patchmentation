import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import BBox
import pytest

def test_bbox():
    xmin, xmax = 1, 2
    ymin, ymax = 3, 4
    bbox = BBox(xmin, ymin, xmax, ymax)
    assert bbox.xmin == xmin
    assert bbox.ymin == ymin
    assert bbox.xmax == xmax
    assert bbox.ymax == ymax
    assert (xmin, ymin, xmax, ymax) == tuple(bbox)
    str(bbox)
    
def test_bbox_width_1():
    xmin = 4
    xmax = 10
    bbox = BBox(xmin, None, xmax, None)
    actual_width = bbox.width
    expected_width = 6
    assert actual_width == expected_width

def test_bbox_width_2():
    xmin = 4
    xmax = 4
    bbox = BBox(xmin, None, xmax, None)
    actual_width = bbox.width
    expected_width = 0
    assert actual_width == expected_width

def test_bbox_height_1():
    ymin = 4
    ymax = 9
    bbox = BBox(None, ymin, None, ymax)
    actual_height = bbox.height
    expected_height = 5
    assert actual_height == expected_height

def test_bbox_height_2():
    ymin = 4
    ymax = 4
    bbox = BBox(None, ymin, None, ymax)
    actual_height = bbox.height
    expected_height = 0
    assert actual_height == expected_height

def test_bbox_area_1():
    xmin, xmax = 4, 10
    ymin, ymax = 6, 8
    bbox = BBox(xmin, ymin, xmax, ymax)
    actual_area = bbox.area
    expected_area = 12
    assert actual_area == expected_area

def test_bbox_area_2():
    xmin, xmax = 4, 4
    ymin, ymax = 6, 8
    bbox = BBox(xmin, ymin, xmax, ymax)
    actual_area = bbox.area
    expected_area = 0
    assert actual_area == expected_area

def test_bbox_area_3():
    xmin, xmax = 4, 10
    ymin, ymax = 8, 8
    bbox = BBox(xmin, ymin, xmax, ymax)
    actual_area = bbox.area
    expected_area = 0
    assert actual_area == expected_area

def test_bbox_area_4():
    xmin, xmax = 10, 10
    ymin, ymax = 6, 6
    bbox = BBox(xmin, ymin, xmax, ymax)
    actual_area = bbox.area
    expected_area = 0
    assert actual_area == expected_area

def test_bbox_failed_xmin():
    bbox = BBox(1, 5, 10, 15)
    bbox.xmin = 10
    with pytest.raises(ValueError):
        bbox.xmin = 11

def test_bbox_failed_xmax():
    bbox = BBox(1, 5, 10, 15)
    bbox.xmax = 1
    with pytest.raises(ValueError):
        bbox.xmax = 0

def test_bbox_failed_ymin():
    bbox = BBox(1, 5, 10, 15)
    bbox.ymin = 15
    with pytest.raises(ValueError):
        bbox.ymin = 16

def test_bbox_failed_ymax():
    bbox = BBox(1, 5, 10, 15)
    bbox.ymax = 5
    with pytest.raises(ValueError):
        bbox.ymax = 4

def test_bbox_failed_xmin_lt0():
    bbox = BBox(None, None, None, None)
    with pytest.raises(ValueError):
        bbox.xmin = -1

def test_bbox_failed_ymin_lt0():
    bbox = BBox(None, None, None, None)
    with pytest.raises(ValueError):
        bbox.ymin = -1

def test_bbox_failed_xmax_lt0():
    bbox = BBox(None, None, None, None)
    with pytest.raises(ValueError):
        bbox.xmax = -1

def test_bbox_failed_ymax_lt0():
    bbox = BBox(None, None, None, None)
    with pytest.raises(ValueError):
        bbox.ymax = -1
