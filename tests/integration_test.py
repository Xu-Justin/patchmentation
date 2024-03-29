import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.collections import BBox, Image, Patch, ImagePatch, Mask
from patchmentation.utils import loader
from patchmentation.utils import filter
from patchmentation.utils import transform
from patchmentation.utils import Comparator
from patchmentation.utils import functional as F
from patchmentation import patch_augmentation

from typing import List
import numpy as np
import patchmentation
import setup

SAMPLE_PATCHMENTATION_FOLDER_IMAGES = 'dataset/sample_patchmentation/source/obj_train_data/'
SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS = 'dataset/sample_patchmentation/source/obj_train_data/'
SAMPLE_PATCHMENTATION_FILE_NAMES = 'dataset/sample_patchmentation/source/obj.names'

SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_1 = 'dataset/sample_patchmentation/background/background_1.jpg'
SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2 = 'dataset/sample_patchmentation/background/background_2.jpg'

def test_version():
    assert patchmentation.__version__ == setup.version

def test_patch_augmentation_1():
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_1)
    
    negative_patches = []
    negative_patches.append(F.get_negative_patch(dataset.image_patches[0], 1.0))
    negative_patches.append(F.get_negative_patch(dataset.image_patches[1], 0.5))
    negative_patches.append(F.get_negative_patch(dataset.image_patches[2], 0.0))
    negative_patches.append(F.get_negative_patch(background_image, 0.5))

    patches = dataset.image_patches[0].patches + dataset.image_patches[1].patches + dataset.image_patches[2].patches + negative_patches
    max_n_patches = 5

    assert len(patches) > max_n_patches
    image_patch = patch_augmentation(patches, background_image, max_n_patches=max_n_patches)
    assert isinstance(image_patch, ImagePatch)
    assert len(image_patch.patches) <= max_n_patches

def test_patch_augmentation_2():
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2)
    
    max_n_patches = 3

    expected_max_n_patches = 0 + max_n_patches
    patches = dataset.image_patches[0].patches
    image_patch = patch_augmentation(patches, background_image, max_n_patches=max_n_patches)
    actual_n_patches = len(image_patch.patches)
    assert actual_n_patches <= expected_max_n_patches

    expected_max_n_patches = actual_n_patches + max_n_patches
    patches = dataset.image_patches[1].patches
    image_patch = patch_augmentation(patches, image_patch, max_n_patches=max_n_patches)
    actual_n_patches = len(image_patch.patches)
    assert actual_n_patches <= expected_max_n_patches

    expected_max_n_patches = actual_n_patches + max_n_patches
    patches = dataset.image_patches[2].patches
    image_patch = patch_augmentation(patches, image_patch, max_n_patches=max_n_patches)
    actual_n_patches = len(image_patch.patches)
    assert actual_n_patches <= expected_max_n_patches

    assert isinstance(image_patch, ImagePatch)

def test_patch_augmentation_3():
    actions = [
        transform.Resize(width=200, aspect_ratio='auto'),
        transform.RandomScale([0.8, 1.2], [0.9, 1.1]),
        filter.FilterHeight(400, Comparator.LessEqual)
    ]
    
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_1)
    background_image_2 = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2)

    zero_patches = [
        Patch(dataset.image_patches[0].image, BBox(1, 1, 1, 2), None),
        Patch(dataset.image_patches[1].image, BBox(1, 1, 2, 1), None)
    ]

    overflow_patches = [
        Patch(background_image_2, BBox(0, 0, background_image_2.width, background_image_2.height), None)
    ]

    patches = dataset.image_patches[0].patches + dataset.image_patches[1].patches + dataset.image_patches[2].patches + zero_patches + overflow_patches
    image_patch = patch_augmentation(patches, background_image, actions=actions)
    assert isinstance(image_patch, ImagePatch)

def test_patch_augmentation_4():
    actions = [
        transform.Resize(height=200, aspect_ratio='auto'),
        transform.HardEdge(),
        transform.SoftEdge(5, 1),
        transform.Scale(scale_width=0.8, aspect_ratio=(1, 1))
    ]
    
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2)
    
    patch_distribution = np.full((background_image.height, background_image.width), 0)
    patch_distribution[250:300, :] = 255
    patch_distribution[:, 400:450] = 255

    visibility_threshold = 0.2

    patches = dataset.image_patches[0].patches
    image_patch = patch_augmentation(patches, background_image, visibility_threshold, actions, True, patch_distribution)

    patches = dataset.image_patches[1].patches
    image_patch = patch_augmentation(patches, image_patch, visibility_threshold, actions, False, patch_distribution)
    zero_background_patches = [
        Patch(image_patch.image, BBox(1, 1, 1, 2), None),
        Patch(image_patch.image, BBox(1, 1, 2, 1), None)
    ]
    image_patch.patches += zero_background_patches

    patches = dataset.image_patches[2].patches
    image_patch = patch_augmentation(patches, image_patch, visibility_threshold, actions, True, loader.save_mask_image_array_temporary(patch_distribution))

    assert isinstance(image_patch, ImagePatch)

def test_patch_augmentation_5():
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_1)
    background_image_2 = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2)

    zero_patches = [
        Patch(dataset.image_patches[0].image, BBox(1, 1, 1, 2), None),
        Patch(dataset.image_patches[1].image, BBox(1, 1, 2, 1), None)
    ]

    overflow_patches = [
        Patch(background_image_2, BBox(0, 0, background_image_2.width, background_image_2.height), None)
    ]

    patches = zero_patches + overflow_patches
    image_patch = patch_augmentation(patches, background_image)
    assert isinstance(image_patch, ImagePatch)

def test_demo():
    penn_fudan_ped = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    patches = []
    for image_patch in penn_fudan_ped.image_patches:
        patches += image_patch.patches
        if len(patches) > 5:
            break

    assert isinstance(patches, List)
    assert len(patches) > 0
    assert isinstance(patches[0], Patch)
    
    # Load Background Image
    path_image = 'dataset/campus_garden1_frame1/images/contour2.jpg'
    path_annotation = 'dataset/campus_garden1_frame1/labels/contour2.txt'
    path_classes = 'dataset/campus_garden1_frame1/obj.names'

    classes = loader.load_yolo_names(path_classes)
    background_image = Image(path_image)
    background_patches = loader.load_yolo_patches(background_image, path_annotation, classes)
    background_image_patch = ImagePatch(background_image, background_patches)

    assert isinstance(classes, List)
    assert len(classes) > 0
    assert isinstance(classes[0], str)
    assert isinstance(background_image, Image)
    assert isinstance(background_patches, List)
    assert len(background_patches) > 0
    assert isinstance(background_patches[0], Patch)
    assert isinstance(background_image_patch, ImagePatch)

    # Patch Augmentation
    result = patchmentation.patch_augmentation(
        patches, 
        background_image_patch,
        max_n_patches=10
    )

    assert isinstance(result, ImagePatch)

    # Soft-edge Blending
    result = patchmentation.patch_augmentation(
        patches, 
        background_image_patch, 
        actions=[
            filter.FilterWidth(10, Comparator.GreaterThan),
            filter.FilterHeight(10, Comparator.GreaterThan),
            transform.SoftEdge(3)
        ],
        max_n_patches=10
    )

    assert isinstance(result, ImagePatch)

    # Negative Patching
    negative_patches = []
    for _ in range(5):
        negative_patch = F.get_negative_patch(background_image, 0.5)
        if negative_patch is not None:
            negative_patches.append(negative_patch)
    
    assert isinstance(negative_patches, List)
    assert len(negative_patches) > 0
    assert isinstance(negative_patches[0], Patch)

    result = patchmentation.patch_augmentation(
        patches + negative_patches, 
        background_image_patch,
        max_n_patches=10
    )

    assert isinstance(result, ImagePatch)

    # Distribution Mask
    path_distribution_mask = 'dataset/campus_garden1_frame1/distribution_mask/contour2.jpg'
    distribution_mask = Mask(path_distribution_mask)
    
    result = patchmentation.patch_augmentation(
        patches, 
        background_image_patch, 
        patch_distribution=distribution_mask,
        max_n_patches=10
    )

    assert isinstance(result, ImagePatch)
