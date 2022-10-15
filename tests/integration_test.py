import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.collections import Image
from patchmentation.utils import validator
from patchmentation.utils import loader
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
    imagePatch = patch_augmentation(patches, background_image)
    validator.validate_ImagePatch(imagePatch)

def test_patch_augmentation_2():
    dataset = loader.load_yolo_dataset(SAMPLE_PATCHMENTATION_FOLDER_IMAGES, SAMPLE_PATCHMENTATION_FOLDER_ANNOTATIONS, SAMPLE_PATCHMENTATION_FILE_NAMES)
    background_image = Image(SAMPLE_PATCHMENTATION_BACKGROUND_IMAGE_2)
    
    patches = dataset.image_patches[0].patches
    imagePatch = patch_augmentation(patches, background_image)

    patches = dataset.image_patches[1].patches
    imagePatch = patch_augmentation(patches, imagePatch)

    patches = dataset.image_patches[2].patches
    imagePatch = patch_augmentation(patches, background_image)

    validator.validate_ImagePatch(imagePatch)
