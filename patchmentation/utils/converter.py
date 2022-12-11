from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset
from patchmentation.utils import loader

import numpy as np

def array2image(image_array: np.ndarray) -> Image:
    height, width, _ = image_array.shape
    if height == 0 or width == 0:
        raise ValueError(f'Expected image array width greater than 0 and height greater than 0, but got width {width} heigth {height}')
    return loader.save_image_array_temporary(image_array)

def image2patch(image: Image, class_name: str = None) -> Patch:
    height, width, _ = image.shape
    return Patch(image, BBox(0, 0, width, height), class_name)

def array2patch(image_array: np.ndarray, class_name: str = None) -> Patch:
    image = array2image(image_array)
    return image2patch(image, class_name)

def patch2image(patch: Patch) -> Image:
    image_array = patch.image_array
    return array2image(image_array)