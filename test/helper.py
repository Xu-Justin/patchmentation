from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset
from patchmentation.utils.loader import save_image_array_temporary, load_image_array

import random
import string
import sys
import numpy as np
from typing import List, Tuple

DEFAULT_MIN_WIDTH = 20
DEFAULT_MIN_HEIGHT = 20
DEFAULT_MAX_WIDTH = 100
DEFAULT_MAX_HEIGHT = 100
DEFAULT_CLASS_NAME_LENGTH = 6
DEFAULT_MIN_NUMBER_OF_CLASS = 2
DEFAULT_MAX_NUMBER_OF_CLASS = 5
DEFAULT_FILENAME_LENGTH = 8
DEFAULT_IMAGE_EXT = '.jpg'
DEFAULT_MIN_NUMBER_OF_PATCH = 1
DEFAULT_MAX_NUMBER_OF_PATCH = 5
DEFAULT_MIN_NUMBER_OF_IMAGE = 2
DEFAULT_MAX_NUMBER_OF_IMAGE = 5

DEFAULT_EPSILON = 1e-6
FLOAT_MIN = sys.float_info.min
FLOAT_MAX = sys.float_info.max

def generate_width() -> int:
    return random.randint(DEFAULT_MIN_WIDTH, DEFAULT_MAX_WIDTH)

def generate_height() -> int:
    return random.randint(DEFAULT_MIN_HEIGHT, DEFAULT_MAX_HEIGHT)

def generate_class_name(length: int = DEFAULT_CLASS_NAME_LENGTH) -> str:
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def generate_number_of_class() -> int:
    return random.randint(DEFAULT_MIN_NUMBER_OF_CLASS, DEFAULT_MAX_NUMBER_OF_CLASS)

def generate_classes(length: int = DEFAULT_CLASS_NAME_LENGTH, number_of_class: int = None) -> List[str]:
    if number_of_class is None: number_of_class = generate_number_of_class()
    return [generate_class_name(length) for _ in range(number_of_class)]

def generate_filename(ext: str, length: int = DEFAULT_FILENAME_LENGTH) -> str:
    return ''.join(random.choice(string.digits) for _ in range(length)) + ext

def generate_number_of_patch() -> int:
    return random.randint(DEFAULT_MIN_NUMBER_OF_PATCH, DEFAULT_MAX_NUMBER_OF_PATCH)

def generate_number_of_image() -> int:
    return random.randint(DEFAULT_MIN_NUMBER_OF_IMAGE, DEFAULT_MAX_NUMBER_OF_IMAGE)

def generate_x(width: int = None) -> Tuple[int, int]:
    if width is None: width = generate_width()
    xmin = random.randint(0, width-1)
    xmax = random.randint(xmin, width)
    return xmin, xmax

def generate_y(height: int = None) -> Tuple[int, int]:
    if height is None: height = generate_height()
    ymin = random.randint(0, height-1)
    ymax = random.randint(ymin, height)
    return ymin, ymax

def generate_bbox(width: int = None, height: int = None) -> BBox:
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    xmin, xmax = generate_x(width)
    ymin, ymax = generate_y(height)
    return BBox(xmin, ymin, xmax, ymax)

def generate_image_array(width: int = None, height: int = None) -> np.ndarray:
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    channel = 3
    image_array = np.random.rand(height, width, channel)
    image_array = (image_array * 255).astype('uint8')
    return image_array

def generate_image(width: int = None, height: int = None) -> Image:
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    image_array = generate_image_array(width, height)
    return save_image_array_temporary(image_array)
    
def generate_patch(image: Image, width: int = None, height: int = None, classes: List[str] = None) -> Patch:
    if width is None and height is None:
        height, width, _ = load_image_array(image).shape
    assert width is not None
    assert height is not None
    if classes is None: classes = generate_classes(number_of_class=1)
    bbox = generate_bbox(width, height)
    class_name = random.choice(classes)
    return Patch(image, bbox, class_name)

def generate_patches(image: Image, number_of_patch: int = None, width: int = None, height: int = None, classes: List[str] = None) -> List[Patch]:
    if number_of_patch is None: number_of_patch = generate_number_of_patch()
    if width is None and height is None:
        height, width, _ = load_image_array(image).shape
    assert width is not None
    assert height is not None
    if classes is None: classes = generate_classes()
    return [generate_patch(image, width, height, classes) for _ in range(number_of_patch)]

def generate_imagePatch(number_of_patch: int = None, width: int = None, height: int = None, classes: List[str] = None) -> ImagePatch:
    if number_of_patch is None: number_of_patch = generate_number_of_patch()
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    if classes is None: classes = generate_classes()
    image = generate_image(width, height)
    patches = generate_patches(image, number_of_patch, width, height, classes)
    return ImagePatch(image, patches)

def generate_imagePatches(number_of_image: int = None, classes: List[str] = None) -> List[ImagePatch]:
    if number_of_image is None: number_of_image = generate_number_of_image()
    if classes is None: classes = generate_classes()
    return [generate_imagePatch() for _ in range(number_of_image)]

def generate_dataset(number_of_image = None, number_of_class: int = None) -> Dataset:
    if number_of_image is None: number_of_image = generate_number_of_image()
    if number_of_class is None: number_of_class = generate_number_of_class()
    classes = generate_classes(number_of_class=number_of_class)
    imagePatches = generate_imagePatches(number_of_image, classes)
    return Dataset(imagePatches, classes)

# source: https://floating-point-gui.de/errors/comparison/
def compare_float_equal(float_1: float, float_2: float, epsilon: float = DEFAULT_EPSILON) -> bool:
    abs_1 = abs(float_1)
    abs_2 = abs(float_2)
    diff = abs(float_1 - float_2)
    if float_1 == float_2:
        return True
    elif float_1 == 0 or float_2 == 0 or (abs_1 + abs_2) < FLOAT_MIN:
        return diff < (epsilon * FLOAT_MIN)
    else:
        return diff / min(abs_1 + abs_2, FLOAT_MAX) < epsilon
    
def compare_float(float_1: float, float_2: float, epsilon: float = DEFAULT_EPSILON) -> int:
    if compare_float_equal(float_1, float_2, epsilon): return 0
    if float_1 > float_2: return 1
    if float_1 < float_2: return -1
    raise Exception(f'compare_float({float_1}, {float_2}, {epsilon})')

def compare_unordered_list_equal(list_1: list, list_2: list) -> bool:
    if len(list_1) != len(list_2): return False
    list_1 = list_1.copy()
    list_2 = list_2.copy()
    for item in list_1:
        print(item, list_1, list_2)
        if item not in list_2: return False
        list_2.remove(item)
    if len(list_2) == 0: return True
    else: return False