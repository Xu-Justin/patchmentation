from .image_patch import ImagePatch

from typing import List

class Dataset:
    def __init__(self, image_patches: List[ImagePatch], classes: List[str] = None):
        self.image_patches = image_patches
        if classes is None:
            classes = self.generate_classes()
        self.classes = classes

    def __iter__(self):
        return iter((self.image_patches, self.classes))

    def __repr__(self) -> str:
        return f'Dataset(n_image_patches={self.n_image_patches}, n_classes={self.n_classes})'

    @property
    def image_patches(self) -> List[ImagePatch]:
        return self._image_patches

    @image_patches.setter
    def image_patches(self, value: List[ImagePatch]):
        self._image_patches = value

    @property
    def n_image_patches(self) -> int:
        return len(self.image_patches)

    @property
    def classes(self) -> List[str]:
        return self._classes

    @classes.setter
    def classes(self, value: List[str]):
        self._classes = sorted(value)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def generate_classes(self) -> List[str]:
        classes = set()
        for image_patch in self.image_patches:
            for patch in image_patch.patches:
                classes.add(patch.class_name)
        return list(classes)
