from patchmentation.collections import BBox, Patch

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple

def calculate_width(bbox: BBox) -> int:
    return bbox.xmax - bbox.xmin

def calculate_height(bbox: BBox) -> int:
    return bbox.ymax - bbox.ymin

def calculate_area(bbox: BBox) -> int:
    return calculate_width(bbox) * calculate_height(bbox)

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
    intersection_area = calculate_area(bbox_intersection)
    union_area = calculate_area(bbox_1) + calculate_area(bbox_2) - intersection_area
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
    width = calculate_width(bbox)
    height = calculate_height(bbox)
    scaled_width, scaled_height = scale_dimension(width, height, scale)
    xmin, ymin, _, _ = bbox
    xmax = xmin + scaled_width
    ymax = ymin + scaled_height
    return BBox(xmin, ymin, xmax, ymax)

def visibility_suppression(
        patches: List[Patch],
        visibility_threshold: float,
        non_removal_patches: List[Patch] = None,
        attr_bbox: str = 'bbox',
        attr_non_removal_patches_bbox: str = 'bbox'
    ) -> List[Patch]:
    
    min_x, min_y, max_x, max_y = getattr(patches[0], attr_bbox)
    for patch in patches:
        xmin, ymin, xmax, ymax = getattr(patch, attr_bbox)
        min_x = min(min_x, xmin)
        min_y = min(min_y, ymin)
        max_x = max(max_x, xmax)
        max_y = max(max_y, ymax)
    
    EMPTY_CELL = -1
    grid_width = max_x - min_x
    grid_height = max_y - min_y
    grid = np.full((grid_height, grid_width), EMPTY_CELL)
    
    total_area = [0] * len(patches)
    visible_area = [0] * len(patches)

    for i, patch in enumerate(patches):
        xmin, ymin, xmax, ymax = getattr(patch, attr_bbox)
        xmin -= min_x
        xmax -= min_x
        ymin -= min_y
        ymax -= min_y
        grid[ymin:ymax, xmin:xmax] = i
        total_area[i] = calculate_area(getattr(patch, attr_bbox))
    
    if non_removal_patches is not None:
        for patch in non_removal_patches:
            xmin, ymin, xmax, ymax = getattr(patch, attr_non_removal_patches_bbox)
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
    
    for i in range(grid_height):
        for j in range(grid_width):
            if grid[i][j] != EMPTY_CELL:
                visible_area[grid[i][j]] += 1

    visibility = [0] * len(patches)
    for i in range(len(patches)):
        if total_area[i] != 0:
            visibility[i] = visible_area[i] / total_area[i]

    result_patches = []
    for i, patch in enumerate(patches):
        if visibility[i] > visibility_threshold:
            result_patches.append(patch)
    
    return result_patches

def resize_image_array(image_array: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image_array, (width, height), interpolation = cv2.INTER_AREA)

def place_image_array(patch_array: np.ndarray, image_array: np.ndarray, bbox: BBox) -> BBox:
    xmin, ymin, xmax, ymax = bbox
    image_height, image_width, _ = image_array.shape
    patch_height, patch_width, _ = patch_array.shape

    dif_xmin = max(0, 0 - xmin)
    dif_ymin = max(0, 0 - ymin)
    dif_xmax = min(0, image_width - xmax)
    dif_ymax = min(0, image_height - ymax)

    patch_xmin = dif_xmin
    patch_ymin = dif_ymin
    patch_xmax = patch_width + dif_xmax
    patch_ymax = patch_height + dif_ymax

    image_xmin = xmin + dif_xmin
    image_ymin = ymin + dif_ymin
    image_xmax = xmax + dif_xmax
    image_ymax = ymax + dif_ymax

    image_array[image_ymin:image_ymax, image_xmin: image_xmax] = patch_array[patch_ymin:patch_ymax, patch_xmin:patch_xmax]
    return BBox(image_xmin, image_ymin, image_xmax, image_ymax)

def display_image_array(image_array: np.ndarray, block: bool = True) -> None:
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    plt.imshow(image_array)
    plt.show(block=block)
