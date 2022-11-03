import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Mask:
    path: str

    def summary(self) -> None:
        print(
            f'mask path: {self.path}\n'
        )

    def image_array(self) -> np.ndarray:
        return cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

    def shape(self) -> Tuple[int, int]:
        return self.image_array().shape

    def width(self):
        return self.shape()[1]
    
    def height(self):
        return self.shape()[0]
