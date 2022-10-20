from patchmentation.collections import BBox, Image, Patch, ImagePatch
from patchmentation.utils import loader
from patchmentation.utils import functional as F
from patchmentation.utils.transform import Transform
from patchmentation.utils.filter import Filter

import random
from typing import List, Tuple, Union


def patch_augmentation(patches: List[Patch], image: Union[Image, ImagePatch], visibility_threshold: float = 0.5, actions: List[Union[Transform, Filter]] = None) -> ImagePatch:
    if isinstance(image, ImagePatch):
        image = image.image
    
    background_image_array = loader.load_image_array(image)
    background_image_height, background_image_width, _ = background_image_array.shape

    list_image_array = []
    for patch in patches:
        image_array = patch.image_array()
        list_image_array.append(image_array)
        
    if actions is not None:
        for action in actions:
            if isinstance(action, Transform):
                list_image_array = [action.transform(image_array) for image_array in list_image_array]
            if isinstance(action, Filter):
                list_image_array = action.filter(list_image_array)

    patches = []
    for image_array in list_image_array:
        image = loader.save_image_array_temporary(image_array)
        height, width, _ = image_array.shape
        patch = Patch(image, BBox(0, 0, width, height))
        patches.append(patch)

    ATTR_TARGET_BBOX = 'target_bbox'

    for patch in patches:
        width = patch.bbox.width()
        height = patch.bbox.height()
        xmin = random.randint(0, background_image_width - 1)
        ymin = random.randint(0, background_image_height - 1)
        xmax = xmin + width
        ymax = ymin + height
        target_bbox = BBox(xmin, ymin, xmax, ymax)
        setattr(patch, ATTR_TARGET_BBOX, target_bbox)
            
    INF = float('inf')
    patches = F.visibility_suppression(
        patches=patches,
        visibility_threshold=visibility_threshold,
        non_removal_patches=[
            Patch(None, BBox(-INF, -INF, 0, INF), None),
            Patch(None, BBox(-INF, -INF, INF, 0), None),
            Patch(None, BBox(background_image_width + 1, -INF, INF, INF), None),
            Patch(None, BBox(-INF, background_image_height + 1, INF, INF), None),
        ],
        attr_bbox=ATTR_TARGET_BBOX,
        attr_non_removal_patches_bbox='bbox'
    )

    result_image_array = background_image_array
    result_patches: List[Patch] = []
    for patch in patches:
        target_bbox = getattr(patch, ATTR_TARGET_BBOX)
        width = target_bbox.width()
        height = target_bbox.height()
        
        patch_array = patch.image_array()
        patch_array = F.resize_image_array(patch_array, width, height)

        result_bbox = F.place_image_array(patch_array, result_image_array, target_bbox)
        result_patch = Patch('', result_bbox, patch.class_name)
        result_patches.append(result_patch)
    
    result_image = loader.save_image_array_temporary(result_image_array)
    for patch in result_patches:
        patch.image = result_image
    
    for patch in patches:
        delattr(patch, ATTR_TARGET_BBOX)

    return ImagePatch(result_image, result_patches)
