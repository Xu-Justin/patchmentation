from patchmentation.collections import BBox, OverflowBBox, Mask, Image, Patch, ImagePatch
from patchmentation.utils import functional as F
from patchmentation.utils.transform import Transform
from patchmentation.utils.filter import Filter
from patchmentation.utils import converter

import numpy as np
import random
from typing import List, Union

def patch_augmentation(
        patches: List[Patch],
        image: Union[Image, ImagePatch],
        visibility_threshold: float = 0.5,
        actions: List[Union[Transform, Filter]] = None,
        preserve_background_patch: bool = True,
        patch_distribution: Union[np.ndarray, Mask] = None,
        max_n_patches: int = 100
    ) -> ImagePatch:
    
    if max_n_patches < len(patches):
        patches = random.sample(patches, k=max_n_patches)

    background_patches = []

    if isinstance(image, ImagePatch):
        if preserve_background_patch:
            background_patches = image.patches
        image = image.image
    
    background_image_width = image.width
    background_image_height = image.height

    if actions is not None:
        for action in actions:
            if isinstance(action, Transform):
                transformed_patches = []
                for patch in patches:
                    if patch.width == 0 or patch.height == 0: continue
                    transformed_image = action.transform(converter.patch2image(patch))
                    transformed_patch = converter.image2patch(transformed_image, patch.class_name)
                    transformed_patches.append(transformed_patch)
                patches = transformed_patches
            if isinstance(action, Filter):
                patches = action.filter(patches)

    if patch_distribution is None:
        patch_distribution = np.full((background_image_height, background_image_width), 1)
    
    if isinstance(patch_distribution, Mask):
        patch_distribution = patch_distribution.image_array

    patch_distribution = patch_distribution.astype(np.float32)
    patch_distribution -= patch_distribution.min()
    patch_distribution += 1e-6 # smoothing
    patch_distribution /= patch_distribution.max()

    list_patch_bbox = []
    for patch in patches:
        if patch.width == 0 or patch.height == 0: continue
        if patch.width > background_image_width or patch.height > background_image_height: continue
        width = patch.width
        height = patch.height
        weight = patch_distribution.copy()
        weight[background_image_height - height + 1:, :] = 0
        weight[:, background_image_width - width + 1:] = 0
        ymin, xmin = F.get_weighted_random_2d(weight, 1)[0]
        xmax = xmin + width
        ymax = ymin + height
        bbox = BBox(xmin, ymin, xmax, ymax)
        list_patch_bbox.append((patch, bbox))

    list_background_patch_bbox = []
    for patch in background_patches:
        if patch.width == 0 or patch.height == 0: continue
        if patch.width > background_image_width or patch.height > background_image_height: continue
        bbox = patch.bbox
        list_background_patch_bbox.append((patch, bbox))
            
    INF = float('inf')
    list_patch_bbox = F.visibility_thresholding(
        list_patch_bbox,
        visibility_threshold,
        list_non_removal_patch_bbox = [
            (None, OverflowBBox(-INF, -INF, 0, INF)),
            (None, OverflowBBox(-INF, -INF, INF, 0)),
            (None, OverflowBBox(background_image_width, -INF, INF, INF)),
            (None, OverflowBBox(-INF, background_image_height, INF, INF))
        ] + list_background_patch_bbox
    )

    result_image: Image = image
    result_patches: List[Patch] = []
    
    for patch, bbox in list_patch_bbox:
        if patch.width == 0 or patch.height == 0: continue
        if patch.width > background_image_width or patch.height > background_image_height: continue
        patch_image = converter.patch2image(patch)
        result_image = F.overlay_image(result_image, patch_image, bbox)
        result_patch = Patch(None, bbox, patch.class_name)
        result_patches.append(result_patch)

    for patch, bbox in list_background_patch_bbox:
        if patch.width == 0 or patch.height == 0: continue
        if patch.width > background_image_width or patch.height > background_image_height: continue
        patch_image = converter.patch2image(patch)
        result_image = F.overlay_image(result_image, patch_image, bbox)
        result_patch = Patch(None, bbox, patch.class_name)
        result_patches.append(result_patch)
    
    for result_patch in result_patches:
        result_patch.image = result_image
    
    return ImagePatch(result_image, result_patches)
