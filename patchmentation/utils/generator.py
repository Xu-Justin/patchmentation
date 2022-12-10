from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset

import cv2
import os
import random
from tqdm import tqdm
from typing import List, Tuple

def generate_yolo_dataset(dataset: Dataset, n_images: int, folder_images: str, folder_annotations: str, file_names: str, **kwargs) -> None:
    from patchmentation import patch_augmentation
    os.makedirs(folder_images, exist_ok=True)
    os.makedirs(folder_annotations, exist_ok=True)
    save_yolo_names(file_names, dataset.classes)
    patches = []
    for image_patch in dataset.image_patches:
        patches += image_patch.patches
    
    for i in tqdm(range(n_images)):
        image = random.choice(dataset.image_patches)
        result = patch_augmentation(patches, image, **kwargs)
        file_image = os.path.join(folder_images, f'{i}.jpg')
        file_annotation = os.path.join(folder_annotations, f'{i}.txt')
        save_yolo_image_patch(file_image, file_annotation, result, dataset.classes)

def save_yolo_names(file_names: str, classes: List[str]) -> None:
    with open(file_names, 'w') as f:
        for class_name in classes:
            f.write(f'{class_name}\n')

def save_yolo_image_patch(file_image: str, file_annotation: str, image_patch: ImagePatch, classes: List[str]) -> None:
    save_yolo_image(file_image, image_patch.image)
    save_yolo_patches(file_annotation, image_patch.patches, classes)

def save_yolo_image(file_image: str, image: Image) -> None:
    cv2.imwrite(file_image, image.image_array)

def save_yolo_patches(file_annotation: str, patches: List[Patch], classes: List[str]) -> None:
    with open(file_annotation, 'w') as f:
        for patch in patches:
            class_index = classes.index(patch.class_name)
            x_center, y_center, width, height = convert_yolo_bbox(patch.bbox, patch.image.width, patch.image.height)
            f.write(f'{class_index} {x_center} {y_center} {width} {height}\n')

def convert_yolo_bbox(bbox: BBox, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    x_center = ((bbox.xmax + bbox.xmin) / 2) / image_width
    y_center = ((bbox.ymax + bbox.ymin) / 2) / image_height
    width = bbox.width / image_width
    height = bbox.height / image_height
    return x_center, y_center, width, height
