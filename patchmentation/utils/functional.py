from patchmentation.collections import BBox, Image, Mask, Patch, ImagePatch
from patchmentation.utils import loader

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

def intersection(bbox_1: BBox, bbox_2: BBox) -> BBox:
    xmin = max(bbox_1.xmin, bbox_2.xmin)
    ymin = max(bbox_1.ymin, bbox_2.ymin)
    xmax = min(bbox_1.xmax, bbox_2.xmax)
    ymax = min(bbox_1.ymax, bbox_2.ymax)
    
    if xmin > xmax or ymin > ymax:
        return BBox(0, 0, 0, 0)
    return BBox(xmin, ymin, xmax, ymax)

def intersection_over_union(bbox_1: BBox, bbox_2: BBox) -> float:
    bbox_intersection = intersection(bbox_1, bbox_2)
    intersection_area = bbox_intersection.area
    union_area = bbox_1.area + bbox_2.area - intersection_area
    if union_area == 0 and intersection_area == 0:
        return 0
    return intersection_area / union_area

IOU = intersection_over_union

def scale_length(length: int, scale: float) -> int:
    return int(length * scale)

def scale_dimension(width: int, height: int, scale: float) -> Tuple[int, int]:
    scaled_width = scale_length(width, scale)
    scaled_height = scale_length(height, scale)
    return scaled_width, scaled_height

def scale_bbox(bbox: BBox, scale: float) -> BBox:
    width = bbox.width
    height = bbox.height
    scaled_width, scaled_height = scale_dimension(width, height, scale)
    xmin, ymin, _, _ = bbox
    xmax = xmin + scaled_width
    ymax = ymin + scaled_height
    return BBox(xmin, ymin, xmax, ymax)

def visibility_thresholding(list_patch_bbox: List[Tuple[Patch, BBox]], visibility_threshold: float, list_non_removal_patch_bbox: List[Tuple[Patch, BBox]] = None) -> List[Tuple[Patch, BBox]]:
    if len(list_patch_bbox) == 0: return []
    
    min_x, min_y, max_x, max_y = list_patch_bbox[0][1]
    for _, bbox in list_patch_bbox:
        xmin, ymin, xmax, ymax = bbox
        min_x = min(min_x, xmin)
        min_y = min(min_y, ymin)
        max_x = max(max_x, xmax)
        max_y = max(max_y, ymax)

    EMPTY_CELL = -1
    grid_width = max_x - min_x
    grid_height = max_y - min_y
    grid = np.full((grid_height, grid_width), EMPTY_CELL)
    
    total_area = [0] * len(list_patch_bbox)

    for i, (_, bbox) in enumerate(list_patch_bbox):
        xmin, ymin, xmax, ymax = bbox
        xmin -= min_x
        xmax -= min_x
        ymin -= min_y
        ymax -= min_y
        grid[ymin:ymax, xmin:xmax] = i
        total_area[i] = bbox.area
    
    if list_non_removal_patch_bbox is not None:
        for _, bbox in list_non_removal_patch_bbox:
            xmin, ymin, xmax, ymax = bbox
            xmin -= min_x
            xmax -= min_x
            ymin -= min_y
            ymax -= min_y
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, grid_width)
            ymax = min(ymax, grid_height)
            if xmin >= grid_width or ymin >= grid_height or xmax < 0 or ymax < 0 : continue
            grid[ymin:ymax, xmin:xmax] = EMPTY_CELL
    
    visible_area = dict(zip(*np.unique(grid, return_counts=True)))

    visibility = [0] * len(list_patch_bbox)
    for i in range(len(list_patch_bbox)):
        if total_area[i] != 0 and i in visible_area:
            visibility[i] = visible_area[i] / total_area[i]

    result_patch_bbox = []
    for i, patch_bbox in enumerate(list_patch_bbox):
        if visibility[i] >= visibility_threshold:
            result_patch_bbox.append(patch_bbox)
    
    return result_patch_bbox

def resize_image_array(image_array: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image_array, (width, height), interpolation = cv2.INTER_AREA)

def overlay_image(image_a: Image, image_b: Image, bbox: BBox) -> Image:
    assert bbox.width == image_b.width
    assert bbox.height == image_b.height

    background_image_array = image_a.image_array[:,:,:3]
    background_mask_image_array = image_a.mask.image_array

    overlay_image_array = image_b.image_array[:,:,:3]
    overlay_mask_image_array = image_b.mask.image_array

    background_color = background_image_array[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax, :] / 255.0
    background_alpha = background_mask_image_array[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax] / 255.0
    background_alpha = np.dstack((background_alpha, background_alpha, background_alpha))
    overlay_color = overlay_image_array / 255.0
    overlay_alpha = overlay_mask_image_array / 255.0
    overlay_alpha = np.dstack((overlay_alpha, overlay_alpha, overlay_alpha))

    composite_alpha = overlay_alpha + background_alpha * (1 - overlay_alpha)
    composite_color = overlay_color * overlay_alpha + background_color * background_alpha * (1 - overlay_alpha)
    composite_alpha = (composite_alpha[:,:,0] * 255).astype(np.uint8)
    composite_color = (composite_color * 255).astype(np.uint8)

    background_image_array[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax, :] = composite_color
    background_mask_image_array[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax] = composite_alpha
    image_array = np.dstack((background_image_array, background_mask_image_array))
    return loader.save_image_array_temporary(image_array)

def overlay_mask(mask_a: Mask, mask_b: Mask) -> Mask:
    assert mask_a.width == mask_b.width
    assert mask_a.height == mask_b.height

    array_a = mask_a.image_array / 255.0
    array_b = mask_b.image_array / 255.0
    array = ((array_a * array_b) * 255).astype(np.uint8)
    return loader.save_mask_image_array_temporary(array)

def display_image_array(image_array: np.ndarray, block: bool = True) -> None:
    if len(image_array.shape) == 2:
        plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
        plt.show(block=block)
    elif image_array.shape[2] == 3:
        image_array = convert_BGR2RGB(image_array)
        plt.imshow(image_array, vmin=0, vmax=255)
        plt.show(block=block)
    elif image_array.shape[2] == 4:
        image_array = convert_BGRA2RGBA(image_array)
        plt.imshow(image_array, vmin=0, vmax=255)
        plt.show(block=block)
    else:
        raise TypeError(f'Received unexpected image_array with shape {image_array.shape}')

def convert_BGR2RGB(image_array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

def convert_BGRA2RGBA(image_array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGBA)

def convert_BGRA2Grayscale(image_array: np.ndarray) -> np.ndarray:
    alpha = image_array[:,:,3]
    bgr = image_array[:,:,:3]
    grayscale = convert_BGR2Grayscale(bgr)
    return np.dstack((grayscale, alpha))

def convert_BGR2Grayscale(image_array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def crop_image_array(image_array: np.ndarray, bbox: BBox) -> np.ndarray:
    xmin, ymin, xmax, ymax = bbox
    return image_array[ymin:ymax, xmin:xmax]

def gaussian_kernel_2d(kernel_size: int, sigma: float = 1.0) -> np.ndarray:
    xdir_gauss = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.multiply(xdir_gauss.T, xdir_gauss)
    return kernel

NEGATIVE_PATCH_CLASS_NAME = 'NEGATIVE_PATCH'
def get_negative_patch(image_patch: Union[Image, ImagePatch], iou_threshold: float, max_iteration: int = 300) -> Patch:
    if isinstance(image_patch, Image):
        image_patch = ImagePatch(image_patch, [])
    image = image_patch.image
    image_width = image.width
    image_height = image.height
    for _ in range(max_iteration):
        xmin = random.randint(0, image_width - 1)
        ymin = random.randint(0, image_height - 1)
        xmax = random.randint(xmin + 1, image_width)
        ymax = random.randint(ymin + 1, image_height)
        bbox = BBox(xmin, ymin, xmax, ymax)
        valid = True
        for positive_patch in image_patch.patches:
            if intersection_over_union(bbox, positive_patch.bbox) > iou_threshold:
                valid = False
                break
        if valid:
            return Patch(image, bbox, NEGATIVE_PATCH_CLASS_NAME)
    return None

def get_overpatch(image_patch: ImagePatch, iou_threshold: float, max_iteration: int = 300) -> ImagePatch:
    image_width = image_patch.width
    image_height = image_patch.height
    for _ in range(max_iteration):
        xmin = random.randint(0, image_width - 1)
        ymin = random.randint(0, image_height - 1)
        xmax = random.randint(xmin + 1, image_width)
        ymax = random.randint(ymin + 1, image_height)
        bbox = BBox(xmin, ymin, xmax, ymax)
        positive_patches: List[Patch] = []
        for positive_patch in image_patch.patches:
            intersection_bbox = intersection(positive_patch.bbox, bbox)
            if intersection_over_union(positive_patch.bbox, intersection_bbox) > iou_threshold:
                positive_patches.append(positive_patch)
        if len(positive_patches) > 0:
            image = loader.save_image_array_temporary(Patch(image_patch.image, bbox, None).image_array)
            patches = []
            for positive_patch in positive_patches:
                intersection_bbox = intersection(positive_patch.bbox, bbox)
                if intersection_over_union(positive_patch.bbox, intersection_bbox) > iou_threshold:
                    intersection_bbox.xmin -= bbox.xmin
                    intersection_bbox.ymin -= bbox.ymin
                    intersection_bbox.xmax -= bbox.xmin
                    intersection_bbox.ymax -= bbox.ymin
                    patch = Patch(image, intersection_bbox, positive_patch.class_name)
                    patches.append(patch)
            return ImagePatch(image, patches)
    return None

def get_weighted_random_2d(weight: np.ndarray, k: int = 1) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    height, width = weight.shape
    indexes = np.arange(0, height * width)
    flatten_weight = weight.flatten().astype(np.float32)
    flatten_weight = flatten_weight / flatten_weight.sum()
    selected_indexes = np.random.choice(indexes, size=k, p=flatten_weight)
    y = selected_indexes // width
    x = selected_indexes % width
    return np.c_[y, x]
