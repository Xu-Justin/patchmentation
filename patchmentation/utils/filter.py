from patchmentation.utils import Comparator

from abc import ABC, abstractmethod

import numpy as np
from typing import List

class Filter(ABC):
    def __call__(self, list_image_array: List[np.ndarray]) -> List[np.ndarray]:
        self.filter(list_image_array)

    @abstractmethod
    def filter(self, list_image_array: List[np.ndarray]) -> List[np.ndarray]:
        pass

class FilterWidth(Filter):
    def __init__(self, width: int, comparator: Comparator):
        self.width = width
        self.comparator = comparator
    
    def filter(self, list_image_array: List[np.ndarray]) -> List[np.ndarray]:
        result = []
        for image_array in list_image_array:
            _, width, _ = image_array.shape
            if self.comparator(width, self.width):
                result.append(image_array)
        return result
    
class FilterHeight(Filter):
    def __init__(self, height: int, comparator: Comparator):
        self.height = height
        self.comparator = comparator
    
    def filter(self, list_image_array: List[np.ndarray]) -> List[np.ndarray]:
        result = []
        for image_array in list_image_array:
            height, _, _ = image_array.shape
            if self.comparator(height, self.height):
                result.append(image_array)
        return result

