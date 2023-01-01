from patchmentation.collections import Dataset, ImagePatch, Image, Patch, BBox
from patchmentation.utils import loader
from . import datautils
from .data import Data

import cv2
import os
import numpy as np
from typing import Dict, List, Any
from abc import abstractmethod
from tqdm import tqdm

class Campus(Data):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def folder_images(self) -> str:
        return os.path.join(self.folder, 'images')

    @property
    def folder_annotations(self) -> str:
        return os.path.join(self.folder, 'labels')

    @property
    def file_names(self) -> str:
        return os.path.join(self.folder, 'obj.names')
    
    @property
    @abstractmethod
    def url_file_video(self) -> str:
        pass
    
    @property
    @abstractmethod
    def url_file_annotation(self) -> str:
        pass

    @property
    def name_file_video(self) -> str:
        return self.name + '.mp4'

    @property
    def name_file_annotation(self) -> str:
        return self.name + '.txt'

    @property
    def file_video(self) -> str:
        return os.path.join(self.root, self.name_file_video)

    @property
    def file_annotation(self) -> str:
        return os.path.join(self.root, self.name_file_annotation)

    def load(self) -> Dataset:
        self.initialize()
        return loader.load_yolo_dataset(self.folder_images, self.folder_annotations, self.file_names)

    def download(self, overwrite: bool = False) -> None:
        self.download_file_video(overwrite)
        self.download_file_annotation(overwrite)

    def download_file_video(self, overwrite: bool = False) -> None:
        datautils.download(self.url_file_video, self.file_video, overwrite)

    def download_file_annotation(self, overwrite: bool = False) -> None:
        datautils.download(self.url_file_annotation, self.file_annotation, overwrite)

    def exists_file_video(self) -> bool:
        return os.path.exists(self.file_video)

    def exists_file_annotation(self) -> bool:
        return os.path.exists(self.file_annotation)

    def initialize(self) -> None:
        if not self.exists():
            if not self.exists_file_video():
                self.download_file_video(overwrite=True)
            if not self.exists_file_annotation():
                self.download_file_annotation(overwrite=True)
            self.extract_yolo(overwrite=True)

    def extract(self, overwrite: bool = False) -> None:
        raise NotImplementedError

    def extract_yolo(self, overwrite: bool = False) -> None:
        save_video_yolo(self.file_video, self.file_annotation, self.folder_images, self.folder_annotations, self.file_names, overwrite=overwrite)

class Garden1:
    class IP1(Campus):
        @property
        def name(self) -> str:
            return 'campus-garden1-ip1'

        @property
        def url_file_video(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden1/view-IP1.mp4'

        @property
        def url_file_annotation(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden1/view-IP1.txt'

    class Contour2(Campus):
        @property
        def name(self) -> str:
            return 'campus-garden1-contour2'

        @property
        def url_file_video(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden1/view-Contour2.mp4'

        @property
        def url_file_annotation(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden1/view-Contour2.txt'

    class HC2(Campus):
        @property
        def name(self) -> str:
            return 'campus-garden1-hc2'

        @property
        def url_file_video(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden1/view-HC2.mp4'

        @property
        def url_file_annotation(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden1/view-HC2.txt'

    class HC3(Campus):
        @property
        def name(self) -> str:
            return 'campus-garden1-hc3'

        @property
        def url_file_video(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden1/view-HC3.mp4'

        @property
        def url_file_annotation(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden1/view-HC3.txt'

class Garden2:
    class HC1(Campus):
        @property
        def name(self) -> str:
            return 'campus-garden2-hc1'

        @property
        def url_file_video(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden2/view-HC1.mp4'

        @property
        def url_file_annotation(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden2/view-HC1.txt'

    class HC2(Campus):
        @property
        def name(self) -> str:
            return 'campus-garden2-hc2'

        @property
        def url_file_video(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden2/view-HC2.mp4'

        @property
        def url_file_annotation(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden2/view-HC2.txt'

    class HC3(Campus):
        @property
        def name(self) -> str:
            return 'campus-garden2-hc3'

        @property
        def url_file_video(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden2/view-HC3.mp4'

        @property
        def url_file_annotation(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden2/view-HC3.txt'

    class HC4(Campus):
        @property
        def name(self) -> str:
            return 'campus-garden2-hc4'

        @property
        def url_file_video(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden2/view-HC4.mp4'

        @property
        def url_file_annotation(self) -> str:
            return 'https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/023d64c36f073dbba371d31b76f6f20ab46aeaf2/CAMPUS/Garden2/view-HC4.txt'

def save_video_yolo(file_video: str, file_annotation: str, save_folder_images: str, save_folder_annotations: str, save_file_names: str, overwrite: bool = False) -> None:
    datautils.validate_not_exists(save_folder_images, overwrite)
    datautils.validate_not_exists(save_folder_annotations, overwrite)
    datautils.validate_not_exists(save_file_names, overwrite)
    
    annotations = read_file_annotation(file_annotation)
    annotations = organize_annotations_by_frame_number(annotations)

    classes = set()
    for _, annotation in annotations.items():
        for patch in annotation:
            classes.add(patch['label'])
    classes = sorted(list(classes))
    loader.save_yolo_names(classes, save_file_names)

    for frame_index, frame in tqdm(enumerate(video_generator(file_video)), desc=f'save_video_yolo {os.path.basename(file_video)}'):
        frame_annotations = annotations.get(frame_index, [])
        image_patch = construct_image_patch(frame, frame_annotations)
        save_file_image = os.path.join(save_folder_images, f'{str(frame_index).zfill(5)}.jpg')
        save_file_annotation = os.path.join(save_folder_annotations, f'{str(frame_index).zfill(5)}.txt')
        loader.save_yolo_image_patch(image_patch, classes, save_file_image, save_file_annotation)
        
def video_generator(file_video):
    video = cv2.VideoCapture(file_video)
    while True:
        success, frame = video.read()
        if not success:
            return
        yield frame

def read_file_annotation(file_annotation: str) -> List[Dict[str, Any]]:
    annotations = []
    with open(file_annotation, 'r') as f:
        lines = f.readlines()
        for line in lines:
            annotation = read_line_annotation(line)
            annotations.append(annotation)
    return annotations

def read_line_annotation(line: str) -> Dict[str, Any]:
    line = line.strip().replace('"', '').split(' ')
    return {
        'track_id': int(line[0]),
        'xmin': int(line[1]),
        'ymin': int(line[2]),
        'xmax': int(line[3]),
        'ymax': int(line[4]),
        'frame_number': int(line[5]),
        'lost': bool(int(line[6])),
        'occluded': bool(int(line[7])),
        'generated': bool(int(line[8])),
        'label': str(line[9])
    }

def organize_annotations_by_frame_number(annotations: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    dict_annotations = {}
    for annotation in annotations:
        frame_number = annotation['frame_number']
        if frame_number not in dict_annotations.keys():
            dict_annotations[frame_number] = []
        dict_annotations[frame_number].append(annotation)
    return dict_annotations

def construct_image_patch(frame: np.ndarray, annotations: List[Dict[str, Any]]) -> ImagePatch:
    image = loader.save_image_array_temporary(frame)
    patches = construct_patches(image, annotations)
    return ImagePatch(image, patches)
    
def construct_patches(image: Image, annotations: List[Dict[str, Any]]) -> List[Patch]:
    patches = []
    for annotation in annotations:
        if annotation['lost']: continue
        if annotation['occluded']: continue
        patch = construct_patch(image, annotation)
        patches.append(patch)
    return patches

def construct_patch(image: Image, annotation: Dict[str, Any]) -> Patch:
    bbox = construct_bbox(annotation)
    class_name = annotation['label']
    return Patch(image, bbox, class_name)

def construct_bbox(annotation: Dict[str, Any]) -> BBox:
    xmin = int(annotation['xmin'])
    ymin = int(annotation['ymin'])
    xmax = int(annotation['xmax'])
    ymax = int(annotation['ymax'])
    return BBox(xmin, ymin, xmax, ymax)