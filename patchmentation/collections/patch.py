from .bbox import BBox
from .image import Image

import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Patch:
    image: Image
    bbox: BBox
    class_name: str = None

    def __iter__(self) -> Tuple[Image, BBox, str]:
        return iter((self.image, self.bbox, self.class_name))

    def summary(self) -> None:
        print(
            f'Image: {self.image}\n'
            f'bbox: {self.bbox}\n'
            f'class_name : {self.class_name}\n'
        )

    def image_array(self) -> np.ndarray:
        from patchmentation.utils import functional as F
        array = self.image.image_array()
        array = F.crop_image_array(array, self.bbox)
        return array
