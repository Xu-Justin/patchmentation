import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import os
import tempfile
from tests import helper

from patchmentation.utils import generator
from patchmentation.utils import transform
from patchmentation.utils import loader

temporary_folder = tempfile.TemporaryDirectory()

def get_temporary_folder():
    return tempfile.TemporaryDirectory(dir=temporary_folder.name)

def get_temporary_file(suffix: str):
    return tempfile.NamedTemporaryFile(suffix=suffix, dir=temporary_folder.name)

def test_generate_yolo_dataset():
    n_images = 5
    dataset = helper.generate_Dataset()
    actions = [transform.RandomResize((2, 5), (2, 5))]

    folder_images = get_temporary_folder()
    folder_annotations = get_temporary_folder()
    file_names = get_temporary_file('.txt')

    generator.generate_yolo_dataset(dataset, n_images, folder_images.name, folder_annotations.name, file_names.name, actions=actions)
    generated_dataset = loader.load_yolo_dataset(folder_images.name, folder_annotations.name, file_names.name)
    assert generated_dataset.classes == dataset.classes
    assert generated_dataset.n_image_patches == n_images
