from patchmentation.collections import BBox, Mask, Image, Patch, ImagePatch, Dataset
from patchmentation.utils import loader

import random
import string
import sys
import numpy as np
from typing import List, Tuple, Callable, Union
from inspect import signature

DEFAULT_MIN_WIDTH = 20
DEFAULT_MIN_HEIGHT = 20
DEFAULT_MAX_WIDTH = 100
DEFAULT_MAX_HEIGHT = 100
DEFAULT_CLASS_NAME_LENGTH = 6
DEFAULT_MIN_NUMBER_OF_CLASS = 2
DEFAULT_MAX_NUMBER_OF_CLASS = 5
DEFAULT_FILENAME_LENGTH = 8
DEFAULT_IMAGE_EXT = '.jpg'
DEFAULT_MIN_NUMBER_OF_PATCH = 0
DEFAULT_MAX_NUMBER_OF_PATCH = 5
DEFAULT_MIN_NUMBER_OF_IMAGE = 2
DEFAULT_MAX_NUMBER_OF_IMAGE = 5

DEFAULT_EPSILON = 1e-6
FLOAT_MIN = sys.float_info.min
FLOAT_MAX = sys.float_info.max

def _kwargs(kwargs, func: Callable, var: str):
    return kwargs.get(var, signature(func).parameters[var].default)
    
def generate_width() -> int:
    return random.randint(DEFAULT_MIN_WIDTH, DEFAULT_MAX_WIDTH)

def generate_height() -> int:
    return random.randint(DEFAULT_MIN_HEIGHT, DEFAULT_MAX_HEIGHT)

def generate_class_name(length: int = DEFAULT_CLASS_NAME_LENGTH) -> str:
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def generate_number_of_class() -> int:
    return random.randint(DEFAULT_MIN_NUMBER_OF_CLASS, DEFAULT_MAX_NUMBER_OF_CLASS)

def generate_classes(number_of_class: int = None, **kwargs) -> List[str]:
    if number_of_class is None: number_of_class = generate_number_of_class()
    length = _kwargs(kwargs, generate_class_name, 'length')
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

def generate_mask_image_array(width: int = None, height: int = None) -> np.ndarray:
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    image_array = np.random.rand(height, width)
    image_array = (image_array * 255).astype('uint8')
    return image_array

def generate_image_array(width: int = None, height: int = None, channel: int = 3) -> np.ndarray:
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    image_array = np.random.rand(height, width, channel)
    image_array = (image_array * 255).astype('uint8')
    return image_array

def generate_BBox(width: int = None, height: int = None) -> BBox:
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    xmin, xmax = generate_x(width)
    ymin, ymax = generate_y(height)
    return BBox(xmin, ymin, xmax, ymax)

def generate_Mask(width: int = None, height: int = None) -> Mask:
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    image_array = generate_mask_image_array(width, height)
    return loader.save_mask_image_array_temporary(image_array)

def generate_Image(width: int = None, height: int = None, mask: Union[Mask, bool] = None) -> Image:
    if width is None: width = generate_width()
    if height is None: height = generate_height()
    image_array = generate_image_array(width, height)
    image = loader.save_image_array_temporary(image_array)
    if mask is not None:
        image_width = image.width()
        image_height = image.height()
        if mask is True:
            mask = generate_Mask(image_width, image_height)
        if isinstance(mask, Mask):
            assert image_width == mask.width()
            assert image_height == mask.height()
            image.mask = mask
        else:
            raise TypeError(f'Received unexpected mask {mask}')
    return image
    
def generate_Patch(image: Image, classes: List[str] = None, **kwargs) -> Patch:
    if classes is None: classes = generate_classes(number_of_class=1)
    shape = kwargs.get('shape', image.shape())
    height = shape[0]
    width = shape[1]
    bbox = generate_BBox(width, height)
    class_name = random.choice(classes)
    return Patch(image, bbox, class_name)

def generate_patches(image: Image, number_of_patch: int = None, classes: List[str] = None, **kwargs) -> List[Patch]:
    if number_of_patch is None: number_of_patch = generate_number_of_patch()
    if classes is None: classes = generate_classes()
    shape = kwargs.get('shape', image.shape())
    kwargs['shape'] = shape
    return [generate_Patch(image, classes, **kwargs) for _ in range(number_of_patch)]

def generate_ImagePatch(classes: List[str] = None, **kwargs) -> ImagePatch:
    if classes is None: classes = generate_classes()
    number_of_patch = kwargs.get('number_of_patch', generate_number_of_patch())
    width = kwargs.get('width', generate_width())
    height = kwargs.get('height', generate_height())
    kwargs['shape'] = (height, width)
    with_mask = kwargs.get('with_mask', False)
    mask = generate_Mask(width, height) if with_mask else None
    image = generate_Image(width, height, mask)
    patches = generate_patches(image, number_of_patch, classes, **kwargs)
    return ImagePatch(image, patches)

def generate_image_patches(number_of_image: int = None, classes: List[str] = None, **kwargs) -> List[ImagePatch]:
    if number_of_image is None: number_of_image = generate_number_of_image()
    number_of_class = _kwargs(kwargs, generate_classes, 'number_of_class')
    if classes is None: classes = generate_classes(number_of_class, **kwargs)
    return [generate_ImagePatch(classes, **kwargs) for _ in range(number_of_image)]

def generate_Dataset(number_of_image: int = None, number_of_class: int = None, **kwargs) -> Dataset:
    if number_of_image is None: number_of_image = generate_number_of_image()
    if number_of_class is None: number_of_class = generate_number_of_class()
    classes = generate_classes(number_of_class, **kwargs)
    image_patches = generate_image_patches(number_of_image, classes, **kwargs)
    return Dataset(image_patches, classes)

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
        if item not in list_2: return False
        list_2.remove(item)
    if len(list_2) == 0: return True
    else: return False

def check_grayscale(image: Union[np.ndarray, Image]) -> bool:
    if isinstance(image, Image):
        image = image.image_array()
    height, width, _ = image.shape
    for i in range(height):
        for j in range(width):
            pixel = image[i][j]
            b = pixel[0]
            g = pixel[1]
            r = pixel[2]
            if b != g or b != r:
                return False
    return True