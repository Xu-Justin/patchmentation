from .image_patch import ImagePatch

from typing import List
from dataclasses import dataclass

@dataclass
class Dataset:
    image_patches: List[ImagePatch]
    classes: List[str]

    def __iter__(self):
        return iter((self.image_patches, self.classes))

    def summary(self):
        print(
            f'Number of images: {len(self.image_patches)}\n'
            f'Number of classes: {len(self.classes)}\n'
        )
    