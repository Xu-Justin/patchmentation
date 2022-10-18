from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset

import os
import cv2
import numpy as np
import tempfile
from typing import List, Union

temporary_folder = tempfile.TemporaryDirectory()
ATTR_TEMPORARY_FILE = 'temporary_file'

def load_image_array(image: Union[str, Image, ImagePatch]) -> np.ndarray:
    if isinstance(image, str):
        return Image(image).image_array()
    if isinstance(image, (Image, ImagePatch)):
        return image.image_array()
    raise TypeError

def load_image(path: str) -> Image:
    return Image(path)

def save_image_array(image_array: np.ndarray, path: str) -> Image:
    cv2.imwrite(path, image_array)
    return Image(path)

def save_image_array_temporary(image_array: np.ndarray) -> Image:
    temporary_file = tempfile.NamedTemporaryFile(suffix='.png', dir=temporary_folder.name)
    path = temporary_file.name
    image = save_image_array(image_array, path)
    setattr(image, ATTR_TEMPORARY_FILE, temporary_file)
    return image

def load_yolo_dataset(folder_images: str, folder_annotations: str, file_names: str) -> Dataset:
    classes = load_yolo_names(file_names)
    image_patches = load_yolo_image_patches(folder_images, folder_annotations, classes)
    dataset = Dataset(image_patches, classes)
    return dataset

def load_yolo_names(file_names: str) -> List[str]:
    classes = []
    with open(file_names, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            class_name = line
            if class_name == '': continue
            classes.append(class_name)
    return classes

def load_yolo_image_patches(folder_images: str, folder_annotations: str, classes: List[str]) -> List[ImagePatch]:
    image_patches = []
    for file_name in os.listdir(folder_images):
        if file_name.startswith('.'): continue
        if not(file_name.endswith(('.jpg', '.png'))): continue
        file_image = os.path.join(folder_images, file_name)
        file_annotation = os.path.join(folder_annotations, file_name[:-4] + '.txt')
        image = Image(file_image)
        patches = load_yolo_patches(image, file_annotation, classes)
        image_patch = ImagePatch(image, patches)
        image_patches.append(image_patch)
    return image_patches
    
def load_yolo_patches(image: Image, file_annotation: str, classes: List[str]) -> List[ImagePatch]:
    image_height, image_width, _ = image.image_array().shape
    patches = []
    with open(file_annotation, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            class_id, x_center, y_center, yolo_width, yolo_height = line.split()
            class_id = int(class_id)
            x_center = float(x_center)
            y_center = float(y_center)
            yolo_width = float(yolo_width)
            yolo_height  = float(yolo_height)

            bbox = convert_yolo_bbox(x_center, y_center, yolo_width, yolo_height, image_width, image_height)
            class_name = classes[class_id]
            patch = Patch(image, bbox, class_name)
            patches.append(patch)
    return patches

def convert_yolo_bbox(x_center: float, y_center: float, yolo_width: float, yolo_height: float, image_width: int, image_height: int) -> BBox:
    xmin = int((x_center - (yolo_width / 2)) * image_width)
    xmax = int((x_center + (yolo_width / 2)) * image_width)
    ymin = int((y_center - (yolo_height / 2)) * image_height)
    ymax = int((y_center + (yolo_height / 2)) * image_height)
    bbox = BBox(xmin, ymin, xmax, ymax)
    return bbox

def load_coco_dataset(folder_images: str, file_annotations: str) -> Dataset:
    pass

def load_pascal_voc_dataset(folder_images: str, folder_annotations: str, file_imagesets: str, file_classes: str) -> Dataset:
    pass
