import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.collections import BBox, Image, Patch, ImagePatch, Dataset
from test import helper

def test_bbox():
    xmin, xmax = helper.generate_x()
    ymin, ymax = helper.generate_y()
    bbox = BBox(xmin, ymin, xmax, ymax)
    
    # attribute
    assert bbox.xmin == xmin
    assert bbox.ymin == ymin
    assert bbox.xmax == xmax
    assert bbox.ymax == ymax
    
    # __iter__
    assert (xmin, ymin, xmax, ymax) == tuple(bbox)

def test_image():
    path = helper.generate_filename('.jpg')
    image = Image(path)
    
    # attribute
    assert image.path == path

def test_patch():
    image = helper.generate_image()
    bbox = helper.generate_bbox()
    class_name = helper.generate_class_name()
    patch = Patch(image, bbox, class_name)

    # attribute
    assert patch.image is image
    assert patch.bbox is bbox
    assert patch.class_name == class_name
    
    # __iter__
    assert (image, bbox, class_name) == tuple(patch)

def test_imagePatch():
    image = helper.generate_image()
    patches = helper.generate_patches(image)
    imagePatch = ImagePatch(image, patches)

    # attribute
    assert imagePatch.image is image
    assert imagePatch.patches is patches

    # __iter__
    assert (image, patches) == tuple(imagePatch)

def test_dataset():
    classes = helper.generate_classes()
    imagePatches = helper.generate_imagePatches(classes=classes)
    dataset = Dataset(imagePatches, classes)

    # attribute
    assert dataset.imagePatches is imagePatches
    assert dataset.classes is classes

    # __iter__
    assert (imagePatches, classes) == tuple(dataset)
