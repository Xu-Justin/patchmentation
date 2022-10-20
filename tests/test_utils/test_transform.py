import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests import helper
from patchmentation.utils import transform

import pytest

def test_resize_1():
    width = None
    height = None
    aspect_ratio = None
    with pytest.raises(TypeError):
        transform.Resize(width, height, aspect_ratio)

def test_resize_2():
    width = 60
    height = None
    aspect_ratio = None
    resize = transform.Resize(width, height, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 60
    expected_image_height = 40
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height

def test_resize_3():
    width = None
    height = 60
    aspect_ratio = None
    resize = transform.Resize(width, height, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 20
    expected_image_height = 60
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height

def test_resize_4():
    width = 80
    height = 60
    aspect_ratio = None
    resize = transform.Resize(width, height, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 80
    expected_image_height = 60
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height

def test_resize_5():
    width = 80
    height = None
    aspect_ratio = 'auto'
    resize = transform.Resize(width, height, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 80
    expected_image_height = 160
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height
    assert helper.compare_float_equal(image_width / image_height, actual_image_width / actual_image_height)

def test_resize_6():
    width = None
    height = 20
    aspect_ratio = 'auto'
    resize = transform.Resize(width, height, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 10
    expected_image_height = 20
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height
    assert helper.compare_float_equal(image_width / image_height, actual_image_width / actual_image_height)

def test_resize_7():
    width = 40
    height = None
    aspect_ratio = (2, 1)
    resize = transform.Resize(width, height, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 40
    expected_image_height = 20
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height
    assert helper.compare_float_equal(aspect_ratio[0] / aspect_ratio[1], actual_image_width / actual_image_height)

def test_resize_8():
    width = None
    height = 60
    aspect_ratio = (4, 8)
    resize = transform.Resize(width, height, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 30
    expected_image_height = 60
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height
    assert helper.compare_float_equal(aspect_ratio[0] / aspect_ratio[1], actual_image_width / actual_image_height)

def test_resize_9():
    width = 100
    height = 60
    aspect_ratio = (4, 8)
    resize = transform.Resize(width, height, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 100
    expected_image_height = 60
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height

def test_random_resize_1():
    width_range = (20, 40)
    height_range = (40, 60)
    aspect_ratio = None
    random_resize = transform.RandomResize(width_range, height_range, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    assert actual_image_width >= width_range[0] and actual_image_width <= width_range[1]
    assert actual_image_height >= height_range[0] and actual_image_height <= height_range[1]

def test_random_resize_2():
    width_range = (30, 40)
    height_range = (50, 60)
    aspect_ratio = None
    random_resize = transform.RandomResize(width_range, height_range, aspect_ratio)
    
    image_width = 60
    image_height = 20
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    assert actual_image_width >= width_range[0] and actual_image_width <= width_range[1]
    assert actual_image_height >= height_range[0] and actual_image_height <= height_range[1]

def test_random_resize_3():
    width_range = (20, 40)
    height_range = None
    aspect_ratio = None
    random_resize = transform.RandomResize(width_range, height_range, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    assert actual_image_width >= width_range[0] and actual_image_width <= width_range[1]
    assert actual_image_height == image_height

def test_random_resize_4():
    width_range = None
    height_range = (50, 60)
    aspect_ratio = None
    random_resize = transform.RandomResize(width_range, height_range, aspect_ratio)
    
    image_width = 60
    image_height = 20
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    assert actual_image_width == image_width
    assert actual_image_height >= height_range[0] and actual_image_height <= height_range[1]

def test_random_resize_5():
    width_range = (20, 40)
    height_range = None
    aspect_ratio = 'auto'
    random_resize = transform.RandomResize(width_range, height_range, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    assert actual_image_width >= width_range[0] and actual_image_width <= width_range[1]
    assert helper.compare_float_equal(image_width / image_height, actual_image_width / actual_image_height)

def test_random_resize_6():
    width_range = None
    height_range = (50, 60)
    aspect_ratio = 'auto'
    random_resize = transform.RandomResize(width_range, height_range, aspect_ratio)
    
    image_width = 60
    image_height = 20
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    assert actual_image_height >= height_range[0] and actual_image_height <= height_range[1]
    assert helper.compare_float_equal(image_width / image_height, actual_image_width / actual_image_height)

def test_random_resize_7():
    width_range = (20, 40)
    height_range = None
    aspect_ratio = (1, 3)
    random_resize = transform.RandomResize(width_range, height_range, aspect_ratio)
    
    image_width = 20
    image_height = 40
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    assert actual_image_width >= width_range[0] and actual_image_width <= width_range[1]
    assert helper.compare_float_equal(aspect_ratio[0] / aspect_ratio[1], actual_image_width / actual_image_height, epsilon=0.01)

def test_random_resize_8():
    width_range = None
    height_range = (50, 60)
    aspect_ratio = (3, 2)
    random_resize = transform.RandomResize(width_range, height_range, aspect_ratio)
    
    image_width = 60
    image_height = 20
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_resize(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    assert actual_image_height >= height_range[0] and actual_image_height <= height_range[1]
    assert helper.compare_float_equal(aspect_ratio[0] / aspect_ratio[1], actual_image_width / actual_image_height, epsilon=0.01)

def test_random_resize_9():
    width_range = None
    height_range = None
    aspect_ratio = None
    with pytest.raises(TypeError):
        transform.RandomResize(width_range, height_range, aspect_ratio)

def test_scale_1():
    scale_width = None
    scale_height = None
    aspect_ratio = None
    with pytest.raises(TypeError):
        transform.Scale(scale_width, scale_height, aspect_ratio)

def test_scale_2():
    scale_width = 1.5
    scale_height = None
    aspect_ratio = None
    scale = transform.Scale(scale_width, scale_height, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 60
    expected_image_height = 60
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height

def test_scale_3():
    scale_width = None
    scale_height = 0.4
    aspect_ratio = None
    scale = transform.Scale(scale_width, scale_height, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 40
    expected_image_height = 24
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height

def test_scale_4():
    scale_width = 0.8
    scale_height = 1.2
    aspect_ratio = None
    scale = transform.Scale(scale_width, scale_height, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 32
    expected_image_height = 72
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height

def test_scale_5():
    scale_width = 2.0
    scale_height = 0.5
    aspect_ratio = 'auto'
    scale = transform.Scale(scale_width, scale_height, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 80
    expected_image_height = 30
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height

def test_scale_6():
    scale_width = 0.8
    scale_height = None
    aspect_ratio = 'auto'
    scale = transform.Scale(scale_width, scale_height, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 32
    expected_image_height = 48
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height
    assert helper.compare_float_equal(image_width / image_height, actual_image_width / actual_image_height)

def test_scale_7():
    scale_width = None
    scale_height = 0.5
    aspect_ratio = 'auto'
    scale = transform.Scale(scale_width, scale_height, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 20
    expected_image_height = 30
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height
    assert helper.compare_float_equal(image_width / image_height, actual_image_width / actual_image_height)

def test_scale_8():
    scale_width = None
    scale_height = 0.5
    aspect_ratio = (1, 2)
    scale = transform.Scale(scale_width, scale_height, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 15
    expected_image_height = 30
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height
    assert helper.compare_float_equal(aspect_ratio[0] / aspect_ratio[1], actual_image_width / actual_image_height)

def test_scale_9():
    scale_width = 1.5
    scale_height = None
    aspect_ratio = (3, 1)
    scale = transform.Scale(scale_width, scale_height, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    expected_image_width = 60
    expected_image_height = 20
    assert actual_image_width == expected_image_width
    assert actual_image_height == expected_image_height
    assert helper.compare_float_equal(aspect_ratio[0] / aspect_ratio[1], actual_image_width / actual_image_height)

def test_random_scale_1():
    scale_width_range = None
    scale_height_range = None
    aspect_ratio = None
    with pytest.raises(TypeError):
        transform.RandomScale(scale_width_range, scale_height_range, aspect_ratio)

def test_random_scale_2():
    scale_width_range = (0.5, 1.5)
    scale_height_range = None
    aspect_ratio = None
    random_scale = transform.RandomScale(scale_width_range, scale_height_range, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    actual_image_width_scale = actual_image_width / image_width
    actual_image_height_scale = actual_image_height / image_height
    expected_image_height = 60
    assert actual_image_height == expected_image_height
    assert actual_image_width_scale >= scale_width_range[0] and actual_image_width_scale <= scale_width_range[1]
    assert helper.compare_float_equal(actual_image_height_scale, 1.0)

def test_random_scale_3():
    scale_width_range = None
    scale_height_range = (1.5, 2.0)
    aspect_ratio = None
    random_scale = transform.RandomScale(scale_width_range, scale_height_range, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    actual_image_width_scale = actual_image_width / image_width
    actual_image_height_scale = actual_image_height / image_height
    expected_image_width = 40
    assert actual_image_width == expected_image_width
    assert actual_image_height_scale >= scale_height_range[0] and actual_image_height_scale <= scale_height_range[1]
    assert helper.compare_float_equal(actual_image_width_scale, 1.0)

def test_random_scale_4():
    scale_width_range = (0.5, 0.8)
    scale_height_range = (1.5, 2.0)
    aspect_ratio = None
    random_scale = transform.RandomScale(scale_width_range, scale_height_range, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    actual_image_width_scale = actual_image_width / image_width
    actual_image_height_scale = actual_image_height / image_height
    assert actual_image_width_scale >= scale_width_range[0] and actual_image_width_scale <= scale_width_range[1]
    assert actual_image_height_scale >= scale_height_range[0] and actual_image_height_scale <= scale_height_range[1]

def test_random_scale_5():
    scale_width_range = (0.5, 1.5)
    scale_height_range = None
    aspect_ratio = 'auto'
    random_scale = transform.RandomScale(scale_width_range, scale_height_range, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    actual_image_width_scale = actual_image_width / image_width
    actual_image_height_scale = actual_image_height / image_height
    assert actual_image_width_scale >= scale_width_range[0] and actual_image_width_scale <= scale_width_range[1]
    assert helper.compare_float_equal(image_width / image_height, actual_image_width / actual_image_height, epsilon=0.1)

def test_random_scale_6():
    scale_width_range = None
    scale_height_range = (1.5, 2.0)
    aspect_ratio = 'auto'
    random_scale = transform.RandomScale(scale_width_range, scale_height_range, aspect_ratio)

    image_width = 40
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    actual_image_width_scale = actual_image_width / image_width
    actual_image_height_scale = actual_image_height / image_height
    assert actual_image_height_scale >= scale_height_range[0] and actual_image_height_scale <= scale_height_range[1]
    assert helper.compare_float_equal(image_width / image_height, actual_image_width / actual_image_height, epsilon=0.1)

def test_random_scale_7():
    scale_width_range = (0.5, 1.5)
    scale_height_range = None
    aspect_ratio = (2, 1)
    random_scale = transform.RandomScale(scale_width_range, scale_height_range, aspect_ratio)

    image_width = 80
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    actual_image_width_scale = actual_image_width / image_width
    actual_image_height_scale = actual_image_height / image_height
    assert actual_image_width_scale >= scale_width_range[0] and actual_image_width_scale <= scale_width_range[1]
    assert helper.compare_float_equal(aspect_ratio[0] / aspect_ratio[1], actual_image_width / actual_image_height, epsilon=0.1)

def test_random_scale_8():
    scale_width_range = None
    scale_height_range = (1.5, 2.0)
    aspect_ratio = (1, 3)
    random_scale = transform.RandomScale(scale_width_range, scale_height_range, aspect_ratio)

    image_width = 80
    image_height = 60
    image_array = helper.generate_image_array(image_width, image_height)
    resized_image_array = random_scale(image_array)
    actual_image_height, actual_image_width, _ = resized_image_array.shape
    actual_image_width_scale = actual_image_width / image_width
    actual_image_height_scale = actual_image_height / image_height
    assert actual_image_height_scale >= scale_height_range[0] and actual_image_height_scale <= scale_height_range[1]
    assert helper.compare_float_equal(aspect_ratio[0] / aspect_ratio[1], actual_image_width / actual_image_height, epsilon=0.1)

def test_grayscale():
    grayscale = transform.Grayscale()
    image_array = helper.generate_image_array()
    grayscale_image_array = grayscale(image_array)
    assert helper.check_grayscale(grayscale_image_array)

def test_random_grayscale_1():
    p = 0.1
    random_grayscale = transform.RandomGrayscale(p)
    image_array = helper.generate_image_array(5, 5)

    sample = 0
    positive = 0

    iteration = 500
    for _ in range(iteration):
        sample += 1
        grayscale_image_array = random_grayscale(image_array)
        if helper.check_grayscale(grayscale_image_array):
            positive += 1

    actual_p = positive / sample
    if not helper.compare_float_equal(actual_p, p, epsilon=0.1):
        iteration = 1000
        for _ in range(iteration):
            sample += 1
            grayscale_image_array = random_grayscale(image_array)
            if helper.check_grayscale(grayscale_image_array):
                positive += 1
        
    actual_p = positive / sample
    assert helper.compare_float_equal(actual_p, p, epsilon=0.1)

def test_random_grayscale_2():
    p = 0.85
    random_grayscale = transform.RandomGrayscale(p)
    image_array = helper.generate_image_array(5, 5)

    sample = 0
    positive = 0

    iteration = 500
    for _ in range(iteration):
        sample += 1
        grayscale_image_array = random_grayscale(image_array)
        if helper.check_grayscale(grayscale_image_array):
            positive += 1

    actual_p = positive / sample
    if not helper.compare_float_equal(actual_p, p, epsilon=0.1):
        iteration = 1000
        for _ in range(iteration):
            sample += 1
            grayscale_image_array = random_grayscale(image_array)
            if helper.check_grayscale(grayscale_image_array):
                positive += 1
        
    actual_p = positive / sample
    assert helper.compare_float_equal(actual_p, p, epsilon=0.1)
