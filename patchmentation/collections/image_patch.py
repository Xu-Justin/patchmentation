from .patch import Patch
from .image import Image

import cv2
import numpy as np
from typing import List, Tuple

class ImagePatch:
    def __init__(self, image: Image, patches: List[Patch]):
        self.image = image
        self.patches = patches

    def __iter__(self) -> Tuple[Image, List[Patch]]:
        return iter((self.image, self.patches))

    def __repr__(self) -> str:
        return f'ImagePatch(image={self.image}, n_patches={self.n_patches})'
    
    @property
    def image(self) -> Image:
        return self._image
    
    @image.setter
    def image(self, value: Image):
        self._image = value

    @property
    def patches(self) -> List[Patch]:
        if self._patches == None:
            self.patches = []
        return self._patches

    @patches.setter
    def patches(self, value: List[Patch]):
        if value is not None:
            for patch in value:
                if patch.image != self.image:
                    raise ValueError(f'all patch image must equal to image-patch image, but got patch image {patch.image}, image-patch image {self.image}')
        self._patches = value
    
    def image_array(self, classes: List[str] = None, **kwargs) -> np.ndarray:
        rectangle_color = kwargs.get('rectangle_color', (255, 0, 0, 255))
        rectangle_thickness = kwargs.get('rectangle_thickness', 1)
        font = kwargs.get('font', cv2.FONT_HERSHEY_SIMPLEX)
        font_scale = kwargs.get('font_scale', 0.5)
        font_color = kwargs.get('font_color', rectangle_color)
        line_thickness = kwargs.get('line_thickness', 1)
        line_type = kwargs.get('line_type', cv2.LINE_8)
        
        array = self.image.image_array
        for patch in self.patches:
            if classes is not None and patch.class_name not in classes: continue
            xmin, ymin, xmax, ymax = patch.bbox
            class_name = patch.class_name
            array = cv2.rectangle(array, (xmin, ymin), (xmax, ymax), rectangle_color, rectangle_thickness)
            array = cv2.putText(array, class_name, (xmin, ymin), font, font_scale, font_color, line_thickness, line_type)
        return array
    
    @property
    def n_patches(self) -> int:
        return len(self.patches)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def channel(self) -> int:
        return self.image.channel
    