import cv2
import numpy as np
from typing import Tuple, Union
from functools import cached_property

class Mask:
    def __init__(self, path: str):
        self.path = path

    def __repr__(self) -> str:
        return f'Mask(path={self.path})'

    def __eq__(self, mask: Union['Mask', 'EmptyMask']) -> bool:
        if not isinstance(mask, (EmptyMask, Mask)):
            return False
        return equal(self, mask)

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str):
        self.shape_cache_clear()
        self._path = value

    @property
    def image_array(self) -> np.ndarray:
        from patchmentation.utils.loader import _imread
        return _imread(self.path, cv2.IMREAD_GRAYSCALE)

    @cached_property
    def shape(self) -> Tuple[int, int]:
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

class EmptyMask:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f'EmptyMask(width={self.width}, height={self.height})'
    
    def __eq__(self, mask: Union['EmptyMask', Mask]) -> bool:
        if not isinstance(mask, (EmptyMask, Mask)):
            return False
        return equal(self, mask)

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        if value is not None:
            if value < 0:
                raise ValueError(f'empty-mask width cannot less than 0, empty-mask width : {value}')
        self._width = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int):
        if value is not None:
            if value < 0:
                raise ValueError(f'empty-mask height cannot less than 0, empty-mask height : {value}')
        self._height = value

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width
    
    @property
    def image_array(self) -> np.ndarray:
        return np.full((self.height, self.width), 255, np.uint8)
    
def equal(mask1: Union[Mask, EmptyMask], mask2: Union[Mask, EmptyMask]) -> bool:
    if not isinstance(mask1, (Mask, EmptyMask)):
        raise TypeError(f'mask1 is not instance of Mask or EmptyMask, type mask1 {type(mask1)}')
    if not isinstance(mask2, (Mask, EmptyMask)):
        raise TypeError(f'mask2 is not instance of Mask or EmptyMask, type mask2 {type(mask2)}')
    if isinstance(mask1, Mask) and isinstance(mask2, Mask):
        return (mask1.path == mask2.path) or ((mask1.shape == mask2.shape) and (mask1.image_array == mask2.image_array).all())
    if isinstance(mask1, EmptyMask) and isinstance(mask2, EmptyMask):
        return mask1.shape == mask2.shape
    if isinstance(mask1, Mask) and isinstance(mask2, EmptyMask):
        return (mask1.shape == mask2.shape) and (mask1.image_array == mask2.image_array).all()
    if isinstance(mask1, EmptyMask) and isinstance(mask2, Mask):
        return equal(mask2, mask1)
    raise Exception(f'Unknown condition for type mask1 {type(mask1)}, type mask2 {type(mask2)}')
