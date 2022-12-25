from patchmentation.collections import BBox, Mask, EmptyMask, Image, Patch, ImagePatch, Dataset

import os
import cv2
import numpy as np
import tempfile
import json
import functools
import xml.etree.ElementTree as ET
from typing import Dict, List, Union, Tuple
from copy import deepcopy
from tqdm import tqdm

temporary_folder = tempfile.TemporaryDirectory()
ATTR_TEMPORARY_FILE = 'temporary_file'

def copying_lru_cache(maxsize=128, typed=False):
    def decorator(f):
        cached_func = functools.lru_cache(maxsize, typed)(f)
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))
        return wrapper
    return decorator

@copying_lru_cache(maxsize=128)
def _imread(path: str, flags: int = None) -> np.ndarray:
    if flags is None:
        return cv2.imread(path)
    return cv2.imread(path, flags)

def load_image_array(image: Union[str, Image, ImagePatch, Mask, EmptyMask]) -> np.ndarray:
    if isinstance(image, str):
        return Image(image).image_array
    if isinstance(image, (Mask, Image, ImagePatch, Mask, EmptyMask)):
        return image.image_array
    raise TypeError

def load_image(path: str) -> Image:
    return Image(path)

def _save_image_array(image_array: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image_array)
    
def save_mask_image_array(mask_image_array: np.ndarray, path: str) -> Mask:
    _save_image_array(mask_image_array, path)
    return Mask(path)

def save_image_array(image_array: np.ndarray, path: str) -> Image:
    channel = image_array.shape[2]
    if channel == 3:
        _save_image_array(image_array, path)
        return Image(path)
    elif channel == 4:
        array = image_array[:, :, :3]
        _save_image_array(array, path)
        alpha = image_array[:, :, 3]
        mask = save_mask_image_array(alpha, os.path.splitext(path)[0] + '_mask.png')
        return Image(path, mask)
    else:
        raise TypeError(f'Received unexpected image array with channel {channel}')

def save_mask_image_array_temporary(mask_image_array: np.ndarray) -> Mask:
    height, width = mask_image_array.shape
    if (mask_image_array == EmptyMask(width, height).image_array).all():
        return EmptyMask(width, height)
    temporary_file = tempfile.NamedTemporaryFile(suffix='.png', dir=temporary_folder.name)
    path = temporary_file.name
    mask = save_mask_image_array(mask_image_array, path)
    setattr(mask, ATTR_TEMPORARY_FILE, temporary_file)
    return mask

def save_image_array_temporary(image_array: np.ndarray) -> Image:
    channel = image_array.shape[2]
    if channel not in (3, 4):
        raise TypeError(f'Received unexpected image array with channel {channel}')
    temporary_file = tempfile.NamedTemporaryFile(suffix='.png', dir=temporary_folder.name)
    path = temporary_file.name
    image = save_image_array(image_array[:, :, :3], path)
    setattr(image, ATTR_TEMPORARY_FILE, temporary_file)
    if channel == 4:
        mask = save_mask_image_array_temporary(image_array[:, :, 3])
        image.mask = mask
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
        for line in tqdm(lines, desc=f'load_yolo_names'):
            line = line.strip()
            class_name = line
            if class_name == '': continue
            classes.append(class_name)
    return classes

def load_yolo_image_patches(folder_images: str, folder_annotations: str, classes: List[str]) -> List[ImagePatch]:
    image_patches = []
    for file_name in tqdm(os.listdir(folder_images), desc=f'load_yolo_image_patches'):
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
    image_height, image_width, _ = image.image_array.shape
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

def save_yolo_dataset(dataset: Dataset, folder_images: str, folder_annotations: str, file_names: str) -> None:
    save_yolo_names(dataset.classes, file_names)
    save_yolo_image_patches(dataset.image_patches, dataset.classes, folder_images, folder_annotations)

def save_yolo_names(classes: List[str], file_names: str) -> None:
    os.makedirs(os.path.dirname(file_names), exist_ok=True)
    if os.path.exists(file_names):
        raise FileExistsError(file_names)
    with open(file_names, 'w') as f:
        for class_name in classes:
            f.write(f'{class_name}\n')

def save_yolo_image_patches(image_patches: List[ImagePatch], classes: List[str], folder_images: str, folder_annotations: str) -> None:
    os.makedirs(folder_images)
    os.makedirs(folder_annotations)
    for i, image_patch in enumerate(tqdm(image_patches, desc='save_yolo_image_patches')):
        file_image = os.path.join(folder_images, f'{i}.jpg')
        file_annotation = os.path.join(folder_annotations, f'{i}.txt')
        save_yolo_image_patch(image_patch, classes, file_image, file_annotation)

def save_yolo_image_patch(image_patch: ImagePatch, classes: List[str], file_image: str, file_annotation: str) -> None:
    save_yolo_image(image_patch.image, file_image)
    save_yolo_patches(image_patch.patches, classes, file_annotation)

def save_yolo_image(image: Image, file_image: str) -> None:
    os.makedirs(os.path.dirname(file_image), exist_ok=True)
    if os.path.exists(file_image):
        raise FileExistsError(file_image)
    _save_image_array(image.image_array, file_image)
    
def save_yolo_patches(patches: List[Patch], classes: List[str], file_annotation: str) -> None:
    os.makedirs(os.path.dirname(file_annotation), exist_ok=True)
    if os.path.exists(file_annotation):
        raise FileExistsError(file_annotation)
    with open(file_annotation, 'w') as f:
        for patch in patches:
            class_index = classes.index(patch.class_name)
            x_center, y_center, yolo_width, yolo_height = convert_bbox_yolo(patch.bbox, patch.image.width, patch.image.height)
            f.write(f'{class_index} {x_center:.8f} {y_center:.8f} {yolo_width:.8f} {yolo_height:.8f}\n')

def convert_bbox_yolo(bbox: BBox, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    x_center = ((bbox.xmin + bbox.xmax) / 2) / image_width
    y_center = ((bbox.ymin + bbox.ymax) / 2) / image_height
    yolo_width = bbox.width / image_width
    yolo_height = bbox.height / image_height
    return x_center, y_center, yolo_width, yolo_height

def load_coco_dataset(folder_images: str, file_annotations: str) -> Dataset:
    with open(file_annotations, 'r') as file_coco: 
        data = json.load(file_coco)
    classes = load_coco_categories(data)
    image_patches = load_coco_image_patches(data, folder_images, classes)
    dataset = Dataset(image_patches, classes)
    return dataset

def load_coco_categories(data_json: dict) -> List[str]:
    coco_classes = data_json['categories']
    classes = []
    for coco_class in tqdm(coco_classes, desc=f'load_coco_categories'):
        class_name = coco_class['name']
        classes.append(class_name)
    return classes

def load_coco_image_patches(data_json: dict, folder_images: str, classes: List[str]) -> List[ImagePatch]:
    images = load_coco_images(data_json['images'], folder_images)
    annotations = load_coco_annotations(data_json['annotations'], classes)
    image_patches = []
    for image_id in tqdm(images.keys(), desc=f'load_coco_image_patches'):
        image = images[image_id]
        patches = []
        if image_id in annotations.keys():
            for bbox, class_name in annotations[image_id]:
                patch = Patch(image, bbox, class_name)
                patches.append(patch)
        image_patch = ImagePatch(image, patches)
        image_patches.append(image_patch)
    return image_patches

def load_coco_images(coco_images: List, folder_images: str) -> Dict[int, Image]:
    images = {}
    for coco_image in tqdm(coco_images, desc=f'load_coco_images'):
        id = coco_image['id']
        file_name = coco_image['file_name']
        path = os.path.join(folder_images, file_name)
        image = Image(path)
        images[id] = image
    return images

def load_coco_annotations(coco_annotations: List, classes: List[str]) -> Dict[int, List[Tuple[BBox, str]]]:
    annotations = {}
    for coco_annotation in tqdm(coco_annotations, desc=f'load_coco_annotations'):
        image_id = coco_annotation['image_id']
        category_id = coco_annotation['category_id']
        coco_bbox = coco_annotation['bbox']
        class_name = classes[category_id - 1]
        bbox = convert_coco_bbox(coco_bbox[0], coco_bbox[1], coco_bbox[2], coco_bbox[3])
        if image_id not in annotations.keys():
            annotations[image_id] = []
        annotations[image_id].append((bbox, class_name))
    return annotations

def convert_coco_bbox(x: int, y: int, width: int, height: int) -> BBox:
    xmin = int(x)
    ymin = int(y)
    xmax = int(x + width)
    ymax = int(y + height)
    bbox = BBox(xmin, ymin, xmax, ymax)
    return bbox
    
def save_coco_dataset(dataset: Dataset, folder_images: str, file_annotations: str) -> None:
    raise NotImplementedError

def load_pascal_voc_dataset(folder_images: str, folder_annotations: str, file_imagesets: str) -> Dataset:
    imagesets = load_pascal_voc_imagesets(file_imagesets)
    image_patches = load_pascal_voc_image_patches(folder_images, folder_annotations, imagesets)
    classes = set()
    for _, patches in image_patches:
        for _, _, class_name in patches:
            classes.add(class_name)
    classes = list(classes)
    dataset = Dataset(image_patches, classes)
    return dataset

def load_pascal_voc_imagesets(file_imagesets: str) -> List[str]:
    imagesets = []
    with open(file_imagesets, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f'load_pascal_voc_imagesets'):
            line = line.strip()
            imagesets.append(line)
    return imagesets

def load_pascal_voc_image_patches(folder_images: str, folder_annotations: str, imagesets: List[str]) -> List[ImagePatch]:
    images = load_pascal_voc_images(folder_images, imagesets)
    annotations = load_pascal_voc_annotations(folder_annotations, imagesets)
    image_patches = []
    for image, annotation in tqdm(list(zip(images, annotations)), desc=f'load_pascal_voc_image_patches'):
        patches = []
        for bbox, class_name in annotation:
            patch = Patch(image, bbox, class_name)
            patches.append(patch)
        image_patch = ImagePatch(image, patches)
        image_patches.append(image_patch)
    return image_patches
    
def load_pascal_voc_images(folder_images: str, imagesets: List[str]) -> List[Image]:
    images = []
    for file_name in tqdm(imagesets, desc=f'load_pascal_voc_images'):
        path = None
        for ext in ['.jpg', '.png', '.jpeg', '']:
            path = os.path.join(folder_images, file_name + ext)
            if os.path.exists(path):
                break
        assert os.path.exists(path)
        image = Image(path)
        images.append(image)
    return images

def load_pascal_voc_annotations(folder_annotations: str, imagesets: List[str]) -> List[List[Tuple[BBox, str]]]:
    annotations = []
    for file_name in tqdm(imagesets, desc=f'load_pascal_voc_annotations'):
        path = None
        ext = '.xml'
        path = os.path.join(folder_annotations, file_name + ext)
        assert os.path.exists(path)
        annotation = load_pascal_voc_xml(path)
        annotations.append(annotation)
    return annotations
    
def load_pascal_voc_xml(file_xml: str) -> List[Tuple[BBox, str]]:
    annotation = []
    tree = ET.parse(file_xml)
    root = tree.getroot()
    for child in root:
        if child.tag == 'object':
            class_name = xmin = ymin = xmax = ymax = None
            for attribute in child:
                if attribute.tag == 'name':
                    class_name = attribute.text
                if attribute.tag == 'bndbox':
                    for cord in attribute:
                        if cord.tag == 'xmin': xmin = int(float(cord.text))
                        if cord.tag == 'ymin': ymin = int(float(cord.text))
                        if cord.tag == 'xmax': xmax = int(float(cord.text))
                        if cord.tag == 'ymax': ymax = int(float(cord.text))
            bbox = convert_pascal_voc_bbox(xmin, ymin, xmax, ymax)
            annotation.append((bbox, class_name))
    return annotation
    
def convert_pascal_voc_bbox(xmin: int, ymin: int, xmax: int, ymax: int) -> BBox:
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    bbox = BBox(xmin, ymin, xmax, ymax)
    return bbox

def save_pascal_voc_dataset(dataset: Dataset, folder_images: str, folder_annotations: str, file_imagesets: str) -> Dataset:
    raise NotImplementedError
