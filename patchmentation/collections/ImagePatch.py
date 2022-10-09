from .Patch import Patch
from .Image import Image

from dataclasses import dataclass
from typing import List

@dataclass
class ImagePatch:
    image: Image
    patches: List[Patch]
    
    def __iter__(self):
        return iter((self.image, self.patches))

    def summary(self):
        print(
            f'Image: {self.image}\n'
            f'Number of patches: {len(self.patches)}\n'
        )
