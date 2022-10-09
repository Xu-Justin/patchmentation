from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset

import os
import cv2
import numpy as np
import tempfile
from typing import List, Union

temporary_folder = tempfile.TemporaryDirectory()
ATTR_TEMPORARY_FILE = 'temporary_file'

def load_image_array(path: Union[str, Image]) -> np.array:
    if isinstance(path, Image):
        path = path.path
    image_array = cv2.imread(path)
    return image_array

def load_patch_array(image: Image, bbox: BBox) -> np.array:
    image_array = load_image_array(image)
    xmin, ymin, xmax, ymax = bbox
    patch_array = image_array[ymin:ymax, xmin:xmax]
    return patch_array

def save_image_array(image_array: np.array, path: str) -> Image:
    cv2.imwrite(path, image_array)
    return Image(path)

def save_image_array_temporary(image_array: np.array) -> Image:
    temporary_file = tempfile.NamedTemporaryFile(suffix='.png', dir=temporary_folder.name)
    path = temporary_file.name
    image = save_image_array(image_array, path)
    setattr(image, ATTR_TEMPORARY_FILE, temporary_file)
    return image

def load_yolo_dataset(folder_images: str, folder_annotations: str, file_names: str) -> Dataset:
    classes = load_yolo_names(file_names)
    imagePatches = load_yolo_imagePatches(folder_images, folder_annotations, classes)
    dataset = Dataset(imagePatches, classes)
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

def load_yolo_imagePatches(folder_images: str, folder_annotations: str, classes: List[str]) -> List[ImagePatch]:
    imagePatches = []
    for file_name in os.listdir(folder_images):
        if file_name.startswith('.'): continue
        if not(file_name.endswith(('.jpg', '.png'))): continue
        file_image = os.path.join(folder_images, file_name)
        file_annotation = os.path.join(folder_annotations, file_name[:-4] + '.txt')
        image = Image(file_image)
        patches = load_yolo_patches(image, file_annotation, classes)
        imagePatch = ImagePatch(image, patches)
        imagePatches.append(imagePatch)
    return imagePatches
    
def load_yolo_patches(image: Image, file_annotation: str, classes: List[str]) -> List[ImagePatch]:
    image_height, image_width, _ = load_image_array(image).shape
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

def load_coco_dataset() -> Dataset:
    pass

def load_pascal_voc_dataset() -> Dataset:
    pass
