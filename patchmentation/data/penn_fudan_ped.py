from patchmentation.collections import Dataset, ImagePatch, Image, Patch, BBox, Mask
from patchmentation.utils import loader
from . import datautils
from .data import Data

import os
import re
import numpy as np
from typing import Dict, List
from tqdm import tqdm

DOWNLOAD_URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'

FOLDER_PENN_FUDAN_PED = os.path.join('PennFudanPed')
FOLDER_IMAGES = os.path.join(FOLDER_PENN_FUDAN_PED, 'PNGImages')
FOLDER_ANNOTATIONS = os.path.join(FOLDER_PENN_FUDAN_PED, 'Annotation')
FOLDER_MASKS = os.path.join(FOLDER_PENN_FUDAN_PED, 'PedMasks')

EXT_IMAGE = '.png'
EXT_ANNOTATION = '.txt'
EXT_MASK = '_mask.png'

class PennFudanPed(Data):

    @property
    def name(self) -> str:
        return 'penn-fudan-ped'

    @property
    def folder_images(self) -> str:
        return os.path.join(self.folder, FOLDER_IMAGES)

    @property
    def folder_annotations(self) -> str:
        return os.path.join(self.folder, FOLDER_ANNOTATIONS)

    @property
    def folder_masks(self) -> str:
        return os.path.join(self.folder, FOLDER_MASKS)

    def load(self) -> Dataset:
        self.initialize()
        return load_penn_fudan_ped_dataset(self.folder_images, self.folder_annotations, self.folder_masks)

    def download(self, overwrite: bool = False) -> None:
        datautils.download(DOWNLOAD_URL, self.file_archive, overwrite)

    def extract(self, overwrite: bool = False) -> None:
        datautils.extract_zip(self.file_archive, self.folder, overwrite)

def load_penn_fudan_ped_dataset(folder_images: str, folder_annotations: str, folder_masks: str) -> Dataset:
    images = load_penn_fudan_ped_images(folder_images)
    annotations = load_penn_fudan_ped_annotations(folder_annotations)
    masks = load_penn_fudan_ped_masks(folder_masks)
    image_patches = load_penn_fudan_ped_image_patches(images, annotations, masks)
    return Dataset(image_patches)

def load_penn_fudan_ped_images(folder_images: str) -> Dict[str, Image]:
    images = {}
    for file_name in tqdm(os.listdir(folder_images), desc='load_penn_fudan_ped_images'):
        if file_name.startswith('.'): continue
        if not file_name.endswith(EXT_IMAGE): continue
        path = os.path.join(folder_images, file_name)
        image = load_penn_fudan_ped_image(path)
        images[file_name] = image
    return images

def load_penn_fudan_ped_image(file_image: str) -> Image:
    return Image(file_image)

def load_penn_fudan_ped_annotations(folder_annotations: str) -> Dict[str, Dict[int, Patch]]:
    annotations = {}
    for file_name in tqdm(os.listdir(folder_annotations), desc='load_penn_fudan_ped_annotations'):
        if file_name.startswith('.'): continue
        if not file_name.endswith(EXT_ANNOTATION): continue
        path = os.path.join(folder_annotations, file_name)
        annotation = load_penn_fudan_ped_annotation(path)
        annotations[file_name] = annotation
    return annotations
        
PATTERN = r'Bounding box for object (?P<index>[0-9]+) "(?P<class_name>[a-zA-Z0-9]+)" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((?P<xmin>[0-9]+), (?P<ymin>[0-9]+)\) - \((?P<xmax>[0-9]+), (?P<ymax>[0-9]+)\)'

def load_penn_fudan_ped_annotation(file_annotation: str) -> Dict[int, Patch]:
    annotation = {}
    with open(file_annotation, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            match = re.match(PATTERN, line)
            if match is None: continue
            groupdict = match.groupdict()
            index =int(groupdict['index'])
            class_name = str(groupdict['class_name'])
            xmin = int(groupdict['xmin'])
            ymin = int(groupdict['ymin'])
            xmax = int(groupdict['xmax'])
            ymax = int(groupdict['ymax'])
            bbox = convert_penn_fudan_ped_bbox(xmin, ymin, xmax, ymax)
            patch = Patch(None, bbox, class_name)
            annotation[index] = patch
    return annotation
                
def convert_penn_fudan_ped_bbox(xmin: int, ymin: int, xmax: int, ymax: int) -> BBox:
    xmin = int(xmin) - 1
    ymin = int(ymin) - 1
    xmax = int(xmax)
    ymax = int(ymax)
    return BBox(xmin, ymin, xmax, ymax)

def load_penn_fudan_ped_masks(folder_masks: str) -> Dict[str, Dict[int, Mask]]:
    masks = {}
    for file_name in tqdm(os.listdir(folder_masks), desc='load_penn_fudan_ped_masks'):
        if file_name.startswith('.'): continue
        if not file_name.endswith(EXT_MASK): continue
        path = os.path.join(folder_masks, file_name)
        mask = load_penn_fudan_ped_mask(path)
        masks[file_name] = mask
    return masks

def load_penn_fudan_ped_mask(file_mask: str) -> Dict[int, Mask]:
    dict_mask = {}
    array = Mask(file_mask).image_array
    for i in range(1, array.max() + 1):
        mask_image_array = (array == i).astype(np.uint8) * 255
        mask = loader.save_mask_image_array_temporary(mask_image_array)
        dict_mask[i] = mask
    return dict_mask

def load_penn_fudan_ped_image_patches(images: Dict[str, Image], annotations: Dict[str, Dict[int, Patch]], masks: Dict[str, Dict[int, Mask]]) -> List[ImagePatch]:
    image_patches = []
    for image_file_name, image in images.items():
        annotation_file_name = datautils.remove_ext(image_file_name) + '.txt'
        mask_file_name = datautils.remove_ext(image_file_name) + '_mask.png'
        annotation = annotations[annotation_file_name]
        mask = masks[mask_file_name]
        patches = load_penn_fudan_ped_patches(annotation, mask)
        for patch in patches:
            patch.image = image
        image_patch = ImagePatch(image, patches)
        image_patches.append(image_patch)
    return image_patches

def load_penn_fudan_ped_patches(dict_patch: Dict[int, Patch], dict_mask: Dict[int, Mask]) -> List[Patch]:
    patches = []
    for index, patch in dict_patch.items():
        mask = dict_mask[index]
        _patch = Patch(None, patch.bbox, patch.class_name, mask)
        patches.append(_patch)
    return patches
