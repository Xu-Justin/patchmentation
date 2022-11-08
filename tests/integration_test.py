import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.collections import Image
from patchmentation.utils import validator
from patchmentation.utils import loader
from patchmentation.utils import filter
from patchmentation.utils import transform
from patchmentation.utils import Comparator
from patchmentation import patch_augmentation

SAMPLE_PATCHMENTATION_FOLDER_IMAGES = 'dataset/sample_patchmentation/source/obj_train_data/'
SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS = 'dataset/sample_patchmentation/source/obj_train_data/'
SAMPLE_PATCHMENTATION_FILE_NAMES = 'dataset/sample_patchmentation/source/obj.names'

SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_1 = 'dataset/sample_patchmentation/background/background_1.jpg'
SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2 = 'dataset/sample_patchmentation/background/background_2.jpg'

def test_patch_augmentation_1():
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_1)
    
    patches = dataset.image_patches[0].patches + dataset.image_patches[1].patches + dataset.image_patches[2].patches
    image_patch = patch_augmentation(patches, background_image)
    validator.validate_ImagePatch(image_patch)

def test_patch_augmentation_2():
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2)
    
    patches = dataset.image_patches[0].patches
    image_patch = patch_augmentation(patches, background_image)

    patches = dataset.image_patches[1].patches
    image_patch = patch_augmentation(patches, image_patch)

    patches = dataset.image_patches[2].patches
    image_patch = patch_augmentation(patches, image_patch)

    validator.validate_ImagePatch(image_patch)

def test_patch_augmentation_3():
    actions = [
        transform.Resize(width=200, aspect_ratio='auto'),
        transform.RandomScale([0.8, 1.2], [0.9, 1.1]),
        filter.FilterHeight(400, Comparator.LessEqual)
    ]
    
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_1)
    
    patches = dataset.image_patches[0].patches + dataset.image_patches[1].patches + dataset.image_patches[2].patches
    image_patch = patch_augmentation(patches, background_image, actions=actions)
    validator.validate_ImagePatch(image_patch)

def test_patch_augmentation_4():
    actions = [
        transform.Resize(height=200, aspect_ratio='auto'),
        transform.HardEdge(),
        transform.SoftEdge(5, 1),
        transform.Scale(scale_width=0.8, aspect_ratio=(1, 1))
    ]
    
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2)
    
    visibility_threshold = 0.2

    patches = dataset.image_patches[0].patches
    image_patch = patch_augmentation(patches, background_image, visibility_threshold, actions, True)

    patches = dataset.image_patches[1].patches
    image_patch = patch_augmentation(patches, image_patch, visibility_threshold, actions, False)

    patches = dataset.image_patches[2].patches
    image_patch = patch_augmentation(patches, image_patch, visibility_threshold, actions, True)

    validator.validate_ImagePatch(image_patch)
