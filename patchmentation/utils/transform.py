from abc import ABC, abstractmethod

import cv2
import numpy as np
import random
from typing import Tuple, Union

class Transform(ABC):
    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        self.transform(image_array)

    @abstractmethod
    def transform(self, image_array: np.ndarray) -> np.ndarray:
        pass
        
class Resize(Transform):
    def __init__(self, width: int = None, height: int = None, aspect_ratio: Union[Tuple[int, int], str] = None):
        if width is None and height is None:
            raise TypeError(f'Expected both width and height is not None, but got width {width} height {height}')
        self.width = width
        self.height = height
        self.aspect_ratio = aspect_ratio
    
    def transform(self, image_array: np.ndarray) -> np.ndarray:
        image_height, image_width, _ = image_array.shape
        width, height = Resize._generate_width_height(self.width, self.height, image_width, image_height, self.aspect_ratio)
        return Resize.resize(image_array, width, height)

    AUTO_ASPECT_RATIO = 'auto'

    @staticmethod
    def resize(image_array: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(image_array, (width, height), interpolation = cv2.INTER_AREA)

    @staticmethod
    def _generate_width_height(width: int, height: int, image_width: int, image_height: int, aspect_ratio: Union[Tuple[int, int], str]) -> Tuple[int, int]:
        if width is None:
            if aspect_ratio is None:
                width = image_width
            elif aspect_ratio == Resize.AUTO_ASPECT_RATIO:
                ratio = image_width / image_height
                width = height * ratio
            else:
                ratio = aspect_ratio[0] / aspect_ratio[1]
                width = height * ratio
            
        if height is None:
            if aspect_ratio is None:
                height = image_height
            elif aspect_ratio == Resize.AUTO_ASPECT_RATIO:
                ratio = image_width / image_height
                height = width / ratio
            else:
                ratio = aspect_ratio[0] / aspect_ratio[1]
                height = width / ratio
        
        return width, height

class RandomResize(Transform):
    def __init__(self, width_range: Tuple[float, float] = None, height_range: Tuple[float, float] = None, aspect_ratio: Union[Tuple[int, int], str] = None):
        if width_range is None and height_range is None:
            raise TypeError(f'Expected both width_range and height_range is not None, but got width_range {width_range} height_range {height_range}')
        self.width_range = width_range
        self.height_range = height_range
        self.aspect_ratio = aspect_ratio

    def transform(self, image_array: np.ndarray) -> np.ndarray:
        width = None if self.width_range is None else random.randint(self.width_range[0], self.width_range[1])
        height = None if self.height_range is None else random.randint(self.height_range[0], self.height_range[1])
        return Resize(width, height, self.aspect_ratio).transform(image_array)

class Scale(Transform):
    def __init__(self, scale_width: float = None, scale_height: float = None, aspect_ratio: Union[Tuple[int, int], str] = None):
        if scale_width is None and scale_height is None:
            raise TypeError(f'Expected both scale_width and scale_height is not None, but got scale_width {scale_width} scale_height {scale_height}')
        self.scale_width = scale_width
        self.scale_height = scale_height
        self.aspect_ratio = aspect_ratio
    
    def transform(self, image_array: np.ndarray) -> np.ndarray:
        image_height, image_width, _ = image_array.shape
        scale_width, scale_height = Resize._generate_width_height(self.scale_width, self.scale_height, image_width, image_height, self.aspect_ratio)
        return Scale.scale(image_array, scale_width, scale_height)

    @staticmethod
    def scale_length(length: int, scale: float) -> int:
        return int(length * scale)

    @staticmethod
    def scale(image_array: np.ndarray, scale_width: float, scale_height: float) -> np.ndarray:
        image_height, image_width, _ = image_array.shape
        scaled_image_width = Scale.scale_length(image_width, scale_width)
        scaled_image_height = Scale.scale_length(image_height, scale_height)
        return Resize.resize(image_array, scaled_image_width, scaled_image_height)
    
class RandomScale(Transform):
    def __init__(self, scale_width_range: Tuple[float, float] = None, scale_height_range: Tuple[float, float] = None, aspect_ratio: Union[Tuple[int, int], str] = None):
        if scale_width_range is None and scale_height_range is None:
            raise TypeError(f'Expected both scale_width_range and scale_height_range is not None, but got scale_width_range {scale_width_range} scale_height_range {scale_height_range}')
        self.scale_width_range = scale_width_range
        self.scale_height_range = scale_height_range
        self.aspect_ratio = aspect_ratio

    def transform(self, image_array: np.ndarray) -> np.ndarray:
        scale_width = None if self.scale_width_range is None else random.uniform(self.scale_width_range[0], self.scale_width_range[1])
        scale_height = None if self.scale_height_range is None else random.uniform(self.scale_height_range[0], self.scale_height_range[1])
        return Scale(scale_width, scale_height, self.aspect_ratio).transform(image_array)

class Grayscale(Transform):
    def __init__(self):
        pass
    
    def transform(self, image_array: np.ndarray) -> np.ndarray:
        return Grayscale.grayscale(image_array)
        
    @staticmethod
    def grayscale(image_array: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

class RandomGrayscale(Transform):
    def __init__(self, p: float = 0.5):
        self.probability = p
    
    def transform(self, image_array: np.ndarray) -> np.ndarray:
        if random.random() < self.probability:
            return Grayscale().transform(image_array)
        else:
            return image_array
