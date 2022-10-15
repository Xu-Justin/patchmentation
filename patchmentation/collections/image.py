import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Image:
    path: str

    def summary(self) -> None:
        print(
            f'image path: {self.path}\n'
        )

    def image_array(self) -> np.ndarray:
        return cv2.imread(self.path)

    def shape(self) -> Tuple[int, int, int]:
        return self.image_array().shape

    def width(self):
        return self.shape()[1]
    
    def height(self):
        return self.shape()[0]
    
    def channel(self):
        return self.shape()[2]
