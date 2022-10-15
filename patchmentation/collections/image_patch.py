from .patch import Patch
from .image import Image

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ImagePatch:
    image: Image
    patches: List[Patch]
    
    def __iter__(self) -> Tuple[Image, List[Patch]]:
        return iter((self.image, self.patches))

    def summary(self) -> None:
        print(
            f'Image: {self.image}\n'
            f'Number of patches: {len(self.patches)}\n'
        )
    
    def image_array(self, **kwargs) -> np.ndarray:
        rectangle_color = kwargs.get('rectangle_color', (255, 0, 0))
        rectangle_thickness = kwargs.get('rectangle_thickness', 1)
        font = kwargs.get('font', cv2.FONT_HERSHEY_SIMPLEX)
        font_scale = kwargs.get('font_scale', 0.5)
        font_color = kwargs.get('font_color', rectangle_color)
        line_thickness = kwargs.get('line_thickness', 1)
        line_type = kwargs.get('line_type', cv2.LINE_8)
        
        array = self.image.image_array()
        for patch in self.patches:
            xmin, ymin, xmax, ymax = patch.bbox
            class_name = patch.class_name
            array = cv2.rectangle(array, (xmin, ymin), (xmax, ymax), rectangle_color, rectangle_thickness)
            array = cv2.putText(array, class_name, (xmin, ymin), font, font_scale, font_color, line_thickness, line_type)
        return array

