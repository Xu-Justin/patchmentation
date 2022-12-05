from abc import ABC, abstractmethod

import numpy as np
import random
from typing import Tuple, Union
from scipy.signal import convolve2d
from copy import copy

from patchmentation.collections import Image, EmptyMask
from patchmentation.utils import functional as F
from patchmentation.utils import loader

class Transform(ABC):
    def __call__(self, image: Image) -> Image:
        return self.transform(image)

    @abstractmethod
    def transform(self, image: Image) -> Image:
        pass
        
class Resize(Transform):
    def __init__(self, width: int = None, height: int = None, aspect_ratio: Union[Tuple[int, int], str] = None):
        if width is None and height is None:
            raise TypeError(f'Expected both width and height is not None, but got width {width} height {height}')
        self.width = width
        self.height = height
        self.aspect_ratio = aspect_ratio
    
    def transform(self, image: Image) -> Image:
        image_height = image.height
        image_width = image.width
        width, height = Resize._generate_width_height(self.width, self.height, image_width, image_height, self.aspect_ratio)
        return Resize.resize(image, width, height)

    AUTO_ASPECT_RATIO = 'auto'

    @staticmethod
    def resize(image: Image, width: int, height: int) -> Image:
        image_array = image.image_array
        resized_image_array = F.resize_image_array(image_array, width, height)
        resized_image = loader.save_image_array_temporary(resized_image_array)
        return resized_image
    
    @staticmethod
    def _generate_width_height(width: int, height: int, image_width: int, image_height: int, aspect_ratio: Union[Tuple[int, int], str]) -> Tuple[int, int]:
        if width is None:
            if aspect_ratio is None:
                width = image_width
            elif aspect_ratio == Resize.AUTO_ASPECT_RATIO:
                ratio = image_width / image_height
                width = height * ratio
            else:
                ratio = aspect_ratio[0] / aspect_ratio[1]
                width = height * ratio
            
        if height is None:
            if aspect_ratio is None:
                height = image_height
            elif aspect_ratio == Resize.AUTO_ASPECT_RATIO:
                ratio = image_width / image_height
                height = width / ratio
            else:
                ratio = aspect_ratio[0] / aspect_ratio[1]
                height = width / ratio
        
        return int(width), int(height)

class RandomResize(Transform):
    def __init__(self, width_range: Tuple[float, float] = None, height_range: Tuple[float, float] = None, aspect_ratio: Union[Tuple[int, int], str] = None):
        if width_range is None and height_range is None:
            raise TypeError(f'Expected both width_range and height_range is not None, but got width_range {width_range} height_range {height_range}')
        self.width_range = width_range
        self.height_range = height_range
        self.aspect_ratio = aspect_ratio

    def transform(self, image: Image) -> Image:
        width = None if self.width_range is None else random.randint(self.width_range[0], self.width_range[1])
        height = None if self.height_range is None else random.randint(self.height_range[0], self.height_range[1])
        return Resize(width, height, self.aspect_ratio).transform(image)

class Scale(Transform):
    def __init__(self, scale_width: float = None, scale_height: float = None, aspect_ratio: Union[Tuple[int, int], str] = None):
        if scale_width is None and scale_height is None:
            raise TypeError(f'Expected both scale_width and scale_height is not None, but got scale_width {scale_width} scale_height {scale_height}')
        self.scale_width = scale_width
        self.scale_height = scale_height
        self.aspect_ratio = aspect_ratio
    
    def transform(self, image: Image) -> Image:
        image_height = image.height
        image_width = image.width
        scale_width, scale_height = Scale._generate_scale_width_height(self.scale_width, self.scale_height, image_width, image_height, self.aspect_ratio)
        return Scale.scale(image, scale_width, scale_height)

    AUTO_ASPECT_RATIO = 'auto'

    @staticmethod
    def scale_length(length: int, scale: float) -> int:
        return int(length * scale)

    @staticmethod
    def scale(image: Image, scale_width: float, scale_height: float) -> Image:
        image_height = image.height
        image_width = image.width
        scaled_image_width = Scale.scale_length(image_width, scale_width)
        scaled_image_height = Scale.scale_length(image_height, scale_height)
        return Resize.resize(image, scaled_image_width, scaled_image_height)

    @staticmethod
    def _generate_scale_width_height(scale_width: int, scale_height: int, image_width: int, image_height: int, aspect_ratio: Union[Tuple[int, int], str]) -> Tuple[float, float]:
        if scale_width is None:
            if aspect_ratio is None:
                scale_width = 1
            elif aspect_ratio == Scale.AUTO_ASPECT_RATIO:
                scale_width = scale_height
            else:
                width, _ = Resize._generate_width_height(None, image_height * scale_height, image_width, image_height, aspect_ratio)
                scale_width = width / image_width

        if scale_height is None:
            if aspect_ratio is None:
                scale_height = 1
            elif aspect_ratio == Resize.AUTO_ASPECT_RATIO:
                scale_height = scale_width
            else:
                _, height = Resize._generate_width_height(image_width * scale_width, None, image_width, image_height, aspect_ratio)
                scale_height = height / image_height
        
        return float(scale_width), float(scale_height)
    
class RandomScale(Transform):
    def __init__(self, scale_width_range: Tuple[float, float] = None, scale_height_range: Tuple[float, float] = None, aspect_ratio: Union[Tuple[int, int], str] = None):
        if scale_width_range is None and scale_height_range is None:
            raise TypeError(f'Expected both scale_width_range and scale_height_range is not None, but got scale_width_range {scale_width_range} scale_height_range {scale_height_range}')
        self.scale_width_range = scale_width_range
        self.scale_height_range = scale_height_range
        self.aspect_ratio = aspect_ratio

    def transform(self, image: Image) -> Image:
        scale_width = None if self.scale_width_range is None else random.uniform(self.scale_width_range[0], self.scale_width_range[1])
        scale_height = None if self.scale_height_range is None else random.uniform(self.scale_height_range[0], self.scale_height_range[1])
        return Scale(scale_width, scale_height, self.aspect_ratio).transform(image)

class Grayscale(Transform):
    def __init__(self):
        pass
    
    def transform(self, image: Image) -> Image:
        return Grayscale.grayscale(image)
        
    @staticmethod
    def grayscale(image: Image) -> Image:
        image_array = image.image_array
        channel = image.channel
        if channel == 3:
            grayscale_image_array = F.convert_BGR2Grayscale(image_array)
        elif channel == 4:
            grayscale_image_array = F.convert_BGRA2Grayscale(image_array)
        else:
            raise TypeError(f'Received unexpected image channel {channel}')
        grayscale = loader.save_image_array_temporary(grayscale_image_array)
        return grayscale

class RandomGrayscale(Transform):
    def __init__(self, p: float = 0.5):
        self.probability = p
    
    def transform(self, image: Image) -> Image:
        if random.random() < self.probability:
            return Grayscale().transform(image)
        else:
            return image

class SoftEdge(Transform):
    def __init__(self, kernel_size: int, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = F.gaussian_kernel_2d(self.kernel_size, self.sigma)
    
    def transform(self, image: Image) -> Image:
        return SoftEdge.softedge(image, self.kernel)
        
    @staticmethod
    def softedge(image: Image, kernel: np.ndarray) -> Image:
        kernel_height, kernel_width = kernel.shape
        if kernel_height % 2 == 0:
            raise ValueError(f'Expected kernel_height is odd, but received kernel_height {kernel_height}')
        if kernel_width % 2 == 0:
            raise ValueError(f'Expected kernel_width is odd, but received kernel_width {kernel_width}')
        mask_image_array = image.mask.image_array
        mask_height, mask_width = mask_image_array.shape
        pad_up = pad_down = int((kernel_height - 1) / 2)
        pad_left = pad_right = int((kernel_width - 1) / 2)
        mask_image_array = F.resize_image_array(mask_image_array, mask_width - pad_left - pad_right, mask_height - pad_up - pad_down)
        mask_image_array = np.pad(mask_image_array, ((pad_up, pad_down), (pad_left, pad_right)))
        mask_image_array = convolve2d(mask_image_array, kernel, mode='same').astype(np.uint8)
        mask = loader.save_mask_image_array_temporary(mask_image_array)
        result_image = copy(image)
        result_image.mask = mask
        return result_image

class HardEdge(Transform):
    def __init__(self):
        pass

    def transform(self, image: Image) -> Image:
        result_image = copy(image)
        result_image.mask = EmptyMask(image.width, image.height)
        return result_image
