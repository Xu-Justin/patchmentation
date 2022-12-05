from patchmentation.utils.comparator import Comparator

from abc import ABC, abstractmethod

import numpy as np
from typing import List

from patchmentation.collections import Patch

class Filter(ABC):
    def __call__(self, patches: List[Patch]) -> List[Patch]:
        return self.filter(patches)

    @abstractmethod
    def filter(self, patches: List[Patch]) -> List[Patch]:
        pass

class FilterWidth(Filter):
    def __init__(self, width: int, comparator: Comparator):
        self.width = width
        self.comparator = comparator
    
    def filter(self, patches: List[Patch]) -> List[Patch]:
        result = []
        for patch in patches:
            width = patch.width
            if self.comparator(width, self.width):
                result.append(patch)
        return result
    
class FilterHeight(Filter):
    def __init__(self, height: int, comparator: Comparator):
        self.height = height
        self.comparator = comparator
    
    def filter(self, patches: List[Patch]) -> List[Patch]:
        result = []
        for patch in patches:
            height = patch.height
            if self.comparator(height, self.height):
                result.append(patch)
        return result

class FilterAspectRatio(Filter):
    def __init__(self, width: int, height: int, comparator: Comparator):
        self.width = width
        self.height = height
        self.comparator = comparator

    def filter(self, patches: List[Patch]) -> List[Patch]:
        threshold_aspect_ratio = self.width / self.height
        result = []
        for patch in patches:
            aspect_ratio = patch.width / patch.height
            if self.comparator(aspect_ratio, threshold_aspect_ratio):
                result.append(patch)
        return result
