import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.collections import Dataset
from tests import helper

def test_dataset():
    image_patches = helper.generate_image_patches()
    classes = helper.generate_classes()
    dataset = Dataset(image_patches, classes)
    assert dataset.image_patches is image_patches
    assert dataset.classes is classes
    assert (image_patches, classes) == tuple(dataset)
    dataset.summary()
