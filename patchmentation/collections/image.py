from .mask import Mask

import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Image:
    path: str
    mask: Mask = None

    def summary(self) -> None:
        print(
            f'image path: {self.path}\n'
            f'Mask: {self.mask}\n'
        )

    def image_array(self) -> np.ndarray:
        array = cv2.imread(self.path)
        if self.mask is not None:
            alpha = self.mask.image_array()
            array = np.dstack((array, alpha))
        return array

    def shape(self) -> Tuple[int, int, int]:
        return self.image_array().shape

    def width(self):
        return self.shape()[1]
    
    def height(self):
        return self.shape()[0]
    
    def channel(self):
        return self.shape()[2]
