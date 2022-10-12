from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset
from patchmentation.utils.loader import load_image_array

import os
import numpy as np
from typing import List, Tuple

def validate_bbox(xmin: int, ymin: int, xmax: int, ymax: int, width: int = None, height: int = None):
    assert xmin <= xmax, f'Expected xmin <= xmax, but got xmin {xmin} xmax {xmax}'
    assert ymin <= ymax, f'Expected ymin <= ymax, but got ymin {ymin} ymax {ymax}'
    assert xmin >= 0, f'Expected xmin >= 0, but got xmin {xmin}'
    assert ymin >= 0, f'Expected ymin >= 0, but got ymin {ymin}'
    if width is not None:
        assert xmax <= width, f'Expected xmax <= width, but got xmax {xmax} width {width}'
    if height is not None:
        assert ymax <= height, f'Expected ymax <= height, but got ymax {ymax} height {height}'

def validate_BBox(bbox: BBox, width: int = None, height: int = None):
    xmin, ymin, xmax, ymax = bbox
    validate_bbox(xmin, ymin, xmax, ymax, width, height)

def validate_image_path(path: str):
    assert path.endswith(('.jpg', '.png')), f'Expected image path ends with .jpg or .png, but got path {path}'
    assert os.path.exists(path), f'Expected path exists, but got path {path} not exists'

def validate_shape(width, height, channel):
    assert height >= 0, f'Expected height >= 0, but got height {height}'
    assert width >= 0, f'Expected width >= 0, but got width {width}'
    assert channel == 3, f'Expected channel == 3, but got channel {channel}{" [Grayscale image (channel == 1) is not supported, please convert to 3 channels]"if channel == 1 else ""}'
    
def validate_Shape(shape: Tuple[int, int, int], expected_width: int = None, expected_height: int = None, expected_channel: int = None):
    height, width, channel = shape
    validate_shape(width, height, channel)
    if expected_width is not None:
        assert width == expected_width, f'Expected width is {expected_width}, but got height {width}'
    if expected_height is not None:
        assert height == expected_height, f'Expected height is {expected_height}, but got height {height}'
    if expected_channel is not None:
        assert channel == expected_channel, f'Expected channel is {expected_channel}, but got channel {channel}'

def validate_pixel(pixel: int):
    assert pixel >= 0 and pixel <= 255, f'Expected pixel >= 0 and pixel <= 255, but got pixel {pixel}'

def validate_image_array(image_array: np.ndarray, check_value: bool = False, expected_width: int = None, expected_height: int = None, expected_channel: int = None):
    shape = image_array.shape
    validate_Shape(shape, expected_width, expected_height, expected_channel)
    assert image_array.dtype == 'uint8', f'Expected image_array.dtype == uint8, but got image_array.dtype {image_array.dtype}'
    if check_value:
        height, width, channel = image_array.shape
        for i in range(height):
            for j in range(width):
                for k in range(channel):
                    validate_pixel(image_array[i][j][k])

def validate_Image(image: Image, check_image_array: bool = True, check_image_array_value: bool = False, expected_width: int = None, expected_height: int = None, expected_channel: int = None):
    validate_image_path(image.path)
    if check_image_array:
        image_array = load_image_array(image)
        validate_image_array(image_array, check_image_array_value, expected_width, expected_height, expected_channel)
        
def validate_class_name(class_name: str):
    assert class_name != '', f'Expected class_name is not empty string, but got class_name {class_name}'

def validate_classes(classes: List[str]):
    for class_name in classes:
        validate_class_name(class_name)
    assert len(classes) == len(set(classes)), f'Expected classes has all unique class_name, but got classes {classes}'

def validate_Patch(patch: Patch, width: int = None, height: int = None):
    image, bbox, class_name = patch
    validate_Image(image)
    validate_BBox(bbox, width, height)
    validate_class_name(class_name)

def validate_patches(patches: List[Patch], width: int = None, height: int = None):
    for patch in patches:
        validate_Patch(patch, width, height)

def validate_ImagePatch(imagePatch: ImagePatch):
    image, patches = imagePatch
    validate_Image(image)
    height, width, _ = load_image_array(image).shape
    validate_patches(patches, width, height)

def validate_imagePatches(imagePatches: List[ImagePatch]):
    for imagePatch in imagePatches:
        validate_ImagePatch(imagePatch)

def validate_Dataset(dataset: Dataset):
    imagePatches, classes = dataset
    validate_imagePatches(imagePatches)
    validate_classes(classes)

