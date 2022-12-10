from .mask import Mask, EmptyMask

import numpy as np
from typing import Tuple
from functools import cached_property

class Image:
    def __init__(self, path: str, mask: Mask = None):
        self.path = path
        self.mask = mask

    def __iter__(self) -> Tuple[str, Mask]:
        return iter((self.path, self.mask))

    def __repr__(self) -> str:
        return f'Image(path={self.path}, mask={self.mask})'

    def __eq__(self, image: 'Image') -> bool:
        return (self.path == image.path) and (self.mask == image.mask)

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str):
        self.shape_cache_clear()
        self.shape_without_mask_cache_clear()
        self._path = value
        if self.width_without_mask != self.mask.width:
            raise ValueError(f'image width must equal to mask width, but got image width {self.width_without_mask}, mask width {self.mask.width}')
        if self.height_without_mask != self.mask.height:
            raise ValueError(f'image height must equal to mask height, but got image height {self.height_without_mask}, mask heigth {self.mask.height}')

    @property
    def mask(self) -> Mask:
        if getattr(self, '_mask', None) is None:
            self.mask = self.empty_mask
        return self._mask

    @mask.setter
    def mask(self, value: Mask):
        if value is not None:
            if value.width != self.width_without_mask:
                raise ValueError(f'mask width must equal to image width, but got mask width {value.width}, image width {self.width_without_mask}')
            if value.height != self.height_without_mask:
                raise ValueError(f'mask height must equal to image height, but got mask heigth {value.height}, image height {self.height_without_mask}')
        self._mask = value

    @property
    def image_array(self) -> np.ndarray:
        array = self.image_array_without_mask
        alpha = self.mask.image_array
        array = np.dstack((array, alpha))
        return array

    @cached_property
    def shape(self) -> Tuple[int, int, int]:
        return self.image_array.shape

    def shape_cache_clear(self) -> None:
        key = 'shape'
        if key in self.__dict__:
            del self.__dict__[key]

    @property
    def width(self) -> int:
        return self.shape[1]
    
    @property
    def height(self) -> int:
        return self.shape[0]
    
    @property
    def channel(self) -> int:
        return self.shape[2]

    @property
    def image_array_without_mask(self) -> np.ndarray:
        from patchmentation.utils.loader import _imread
        array = _imread(self.path)
        return array

    @cached_property
    def shape_without_mask(self) -> Tuple[int, int, int]:
        return self.image_array_without_mask.shape

    def shape_without_mask_cache_clear(self) -> None:
        key = 'shape_without_mask'
        if key in self.__dict__:
            del self.__dict__[key]

    @property
    def width_without_mask(self) -> int:
        return self.shape_without_mask[1]
    
    @property
    def height_without_mask(self) -> int:
        return self.shape_without_mask[0]
    
    @property
    def channel_without_mask(self) -> int:
        return self.shape_without_mask[2]
    
    @property
    def empty_mask(self) -> EmptyMask:
        return EmptyMask(self.width_without_mask, self.height_without_mask)
        