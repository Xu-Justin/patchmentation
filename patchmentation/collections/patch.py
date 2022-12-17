from .bbox import BBox
from .image import Image
from .mask import Mask, EmptyMask

import numpy as np
from typing import Tuple
from functools import cached_property

class Patch:
    def __init__(self, image: Image, bbox: BBox, class_name: str = None, mask: Mask = None):
        self.image = image
        self.bbox = bbox
        self.class_name = class_name
        self.mask = mask

    def __iter__(self) -> Tuple[Image, BBox, str]:
        return iter((self.image, self.bbox, self.class_name))

    def __repr__(self) -> str:
        return f'Patch(image={self.image}, bbox={self.bbox}, class_name={self.class_name})'
    
    @property
    def image(self) -> Image:
        return getattr(self, '_image', None)

    @image.setter
    def image(self, value: Image):
        self.shape_cache_clear()
        if self.bbox is not None:
            if value.width < self.bbox.width:
                raise ValueError(f'image width cannot smaller than bbox width, image width : {value.width}, bbox width : {self.bbox.width}')
            if value.height < self.bbox.height:
                raise ValueError(f'image height cannot smaller than bbox height, image height : {value.height}, bbox height : {self.bbox.height}')
        self._image = value

    @property
    def bbox(self) -> BBox:
        return getattr(self, '_bbox', None)

    @bbox.setter
    def bbox(self, value: BBox):
        self.shape_cache_clear()
        if self.image is not None:
            if value.width > self.image.width:
                raise ValueError(f'bbox width cannot greater than image width, bbox width : {value.width}, image width : {self.image.width}')
            if value.height > self.image.height:
                raise ValueError(f'bbox height cannot greater than image height, bbox height : {value.height}, image height : {self.image.height}')
        self._bbox = value

    @property
    def class_name(self) -> str:
        return self._class_name

    @class_name.setter
    def class_name(self, value):
        self._class_name = value

    @property
    def mask(self) -> Mask:
        if getattr(self, '_mask', None) is None:
            self.mask = self.empty_mask
        return self._mask

    @mask.setter
    def mask(self, value: Mask):
        if value is not None and self.image is not None:
            if value.width != self.image.width:
                raise ValueError(f'mask width must equal to patch image width, but got mask width {value.width}, patch image width {self.image.width}')
            if value.height != self.image.height:
                raise ValueError(f'mask height must equal to patch image height, but got mask heigth {value.height}, patch image height {self.image.height}')
        self._mask = value
    
    @property
    def empty_mask(self) -> EmptyMask:
        return EmptyMask(self.image.width, self.image.height)
    
    @property
    def image_array(self) -> np.ndarray:
        from patchmentation.utils import functional as F
        array = self.image.image_array_without_mask
        alpha = F.overlay_mask(self.image.mask, self.mask).image_array
        array = np.dstack((array, alpha))
        array = F.crop_image_array(array, self.bbox)
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
        return self.bbox.width
    
    @property
    def height(self) -> int:
        return self.bbox.height
    
    @property
    def channel(self) -> int:
        return self.shape[2]
