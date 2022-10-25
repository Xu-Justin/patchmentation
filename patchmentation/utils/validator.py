from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset
from patchmentation.utils.loader import load_image_array

import os
import numpy as np
from typing import List, Tuple, Union, Callable
from inspect import signature

def _kwargs(kwargs, func: Callable, var: str, delete: bool = True):
    value = kwargs.get(var, signature(func).parameters[var].default)
    if var in kwargs.keys() and delete:
        del kwargs[var]
    return value

def validate(collection: Union[np.ndarray, BBox, Image, Patch, ImagePatch, Dataset]):
    if isinstance(collection, BBox):
        validate_BBox(collection)
    elif isinstance(collection, Image):
        validate_Image(collection)
    elif isinstance(collection, Patch):
        validate_Patch(collection)
    elif isinstance(collection, ImagePatch):
        validate_ImagePatch(collection)
    elif isinstance(collection, Dataset):
        validate_Dataset(collection)
    else:
        raise TypeError

def validate_BBox(bbox: BBox, **kwargs):
    xmin, ymin, xmax, ymax = bbox
    assert xmin <= xmax, f'Expected xmin <= xmax, but got xmin {xmin} xmax {xmax}'
    assert ymin <= ymax, f'Expected ymin <= ymax, but got ymin {ymin} ymax {ymax}'
    assert xmin >= 0, f'Expected xmin >= 0, but got xmin {xmin}'
    assert ymin >= 0, f'Expected ymin >= 0, but got ymin {ymin}'
    width = kwargs.get('width', None)
    if width is not None:
        assert xmax <= width, f'Expected xmax <= width, but got xmax {xmax} width {width}'
    height = kwargs.get('height', None)
    if height is not None:
        assert ymax <= height, f'Expected ymax <= height, but got ymax {ymax} height {height}'

def validate_Image(image: Image, check_image_array: bool = True, **kwargs):
    path = image.path
    validate_image_path(path)
    if check_image_array:
        image_array = image.image_array()
        check_value = _kwargs(kwargs, validate_image_array, 'check_value')
        check_shape = _kwargs(kwargs, validate_image_array, 'check_shape')
        validate_image_array(image_array, check_shape, check_value, **kwargs)
    
def validate_Patch(patch: Patch, check_image_bbox: bool = True, **kwargs):
    image, bbox, class_name = patch
    check_image_array = _kwargs(kwargs, validate_Image, 'check_image_array')
    validate_Image(image, check_image_array, **kwargs)
    if check_image_bbox:
        kwargs['height'], kwargs['width'], _ = image.shape()
    validate_BBox(bbox, **kwargs)
    classes = _kwargs(kwargs, validate_class_name, 'classes')
    validate_class_name(class_name, classes)

def validate_ImagePatch(image_patch: ImagePatch, **kwargs):
    image, patches = image_patch
    check_image_array = _kwargs(kwargs, validate_Image, 'check_image_array')
    validate_Image(image, check_image_array, **kwargs)
    for i, patch in enumerate(patches):
        assert patch.image is image, f'Expected all image in patches reference same image in image_patch, but got patches[{i}].image {patch.image} image_patch.image {image}'
    check_image_bbox = _kwargs(kwargs, validate_patches, 'check_image_bbox')
    validate_patches(patches, check_image_bbox, **kwargs)

def validate_Dataset(dataset: Dataset, **kwargs):
    image_patches, classes = dataset
    kwargs['classes'] = classes
    validate_image_patches(image_patches, **kwargs)
    validate_classes(classes)

def validate_image_path(path: str):
    assert path.endswith(('.jpg', '.png')), f'Expected image path ends with .jpg or .png, but got path {path}'
    assert os.path.exists(path), f'Expected path exists, but got path {path} not exists'

def validate_image_array(image_array: np.ndarray, check_shape: bool = True, check_value: bool = False, **kwargs):
    assert image_array.dtype == 'uint8', f'Expected image_array.dtype == uint8, but got image_array.dtype {image_array.dtype}'
    if check_shape:
        shape = image_array.shape
        validate_image_array_shape(shape, **kwargs)
    if check_value:
        validate_image_array_value(image_array)

def validate_image_array_shape(shape: Tuple[int, int, int], **kwargs):
    height, width, channel = shape
    assert height >= 0, f'Expected height >= 0, but got height {height}'
    assert width >= 0, f'Expected width >= 0, but got width {width}'
    assert channel == 3, f'Expected channel == 3, but got channel {channel}{" [Grayscale image (channel == 1) is not supported, please convert to 3 channels]"if channel == 1 else ""}'
    expected_width = kwargs.get('expected_width', None)
    if expected_width is not None:
        assert width == expected_width, f'Expected width is {expected_width}, but got height {width}'
    expected_height = kwargs.get('expected_height', None)
    if expected_height is not None:
        assert height == expected_height, f'Expected height is {expected_height}, but got height {height}'
    expected_channel = kwargs.get('expected_channel', None)
    if expected_channel is not None:
        assert channel == expected_channel, f'Expected channel is {expected_channel}, but got channel {channel}'

def validate_image_array_value(image_array: np.ndarray):
    height, width, channel = image_array.shape
    for i in range(height):
        for j in range(width):
            for k in range(channel):
                pixel = image_array[i, j, k]
                assert pixel >= 0 and pixel <= 255, f'Expected pixel >= 0 and pixel <= 255, but got pixel {pixel} at {(i, j, k)}'

def validate_class_name(class_name: str, classes: List[str] = None):
    assert class_name != '', f'Expected class_name is not empty string, but got class_name {class_name}'
    if classes is not None:
        assert class_name in classes, f'Expected class_name is in classes, but got class_name {class_name} classes {classes}'

def validate_classes(classes: List[str], **kwargs):
    assert len(list(classes)) == len(set(classes)), f'Expected classes contains unique class_name. but got classes {classes}'
    for class_name in classes:
        validate_class_name(class_name)

def validate_patches(patches: List[Patch], check_image_bbox: bool = True, **kwargs):
    for patch in patches:
        validate_Patch(patch, check_image_bbox, **kwargs)
    
def validate_image_patches(image_patches: List[ImagePatch], **kwargs):
    for image_patch in image_patches:
        validate_ImagePatch(image_patch, **kwargs)
