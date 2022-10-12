from .ImagePatch import ImagePatch

from dataclasses import dataclass
from typing import List
from pprint import pprint

@dataclass
class Dataset:
    imagePatches: List[ImagePatch]
    classes: List[str]

    def __iter__(self):
        return iter((self.imagePatches, self.classes))

    def summary(self):
        print(
            f'Number of images: {len(self.imagePatches)}\n'
            f'Number of classes: {len(self.classes)}\n'
        )
