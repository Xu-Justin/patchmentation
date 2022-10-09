from .BBox import BBox
from .Image import Image

from dataclasses import dataclass

@dataclass
class Patch:
    image: Image
    bbox: BBox
    class_name: str = None

    def __iter__(self):
        return iter((self.image, self.bbox, self.class_name))

    def summary(self):
        print(
            f'Image: {self.image}\n'
            f'bbox: {self.bbox}\n'
            f'class_name : {self.class_name}\n'
        )
