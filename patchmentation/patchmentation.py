from patchmentation.collections import BBox, Image, Patch, ImagePatch
from patchmentation.utils import loader
from patchmentation.utils import functional as F
from patchmentation.utils.transform import Transform
from patchmentation.utils.filter import Filter
from patchmentation.utils import converter
import random
from typing import List, Tuple, Union


def patch_augmentation(patches: List[Patch], image: Union[Image, ImagePatch], visibility_threshold: float = 0.5, actions: List[Union[Transform, Filter]] = None) -> ImagePatch:
    if isinstance(image, ImagePatch):
        image = image.image
    
    background_image_width = image.width()
    background_image_height = image.height()

    if actions is not None:
        for action in actions:
            if isinstance(action, Transform):
                transformed_patches = []
                for patch in patches:
                    transformed_image = action.transform(converter.patch2image(patch))
                    transformed_patch = converter.image2patch(transformed_image, patch.class_name)
                    transformed_patches.append(transformed_patch)
                patches = transformed_patches
            if isinstance(action, Filter):
                patches = action.filter(patches)

    list_patch_bbox = []
    for patch in patches:
        width = patch.width()
        height = patch.height()
        xmin = random.randint(0, background_image_width - width)
        ymin = random.randint(0, background_image_height - height)
        xmax = xmin + width
        ymax = ymin + height
        bbox = BBox(xmin, ymin, xmax, ymax)
        list_patch_bbox.append((patch, bbox))
            
    INF = float('inf')
    list_patch_bbox = F.visibility_thresholding(
        list_patch_bbox,
        visibility_threshold,
        list_non_removal_patch_bbox=[
            (None, BBox(-INF, -INF, 0, INF)),
            (None, BBox(-INF, -INF, INF, 0)),
            (None, BBox(background_image_width, -INF, INF, INF)),
            (None, BBox(-INF, background_image_height, INF, INF))
        ]
    )

    result_image: Image = image
    result_patches: List[Patch] = []
    
    for patch, bbox in list_patch_bbox:
        patch_image = converter.patch2image(patch)
        result_image = F.overlay_image(result_image, patch_image, bbox)
        result_patch = Patch(None, bbox, patch.class_name)
        result_patches.append(result_patch)

    for result_patch in result_patches:
        result_patch.image = result_image
    
    return ImagePatch(result_image, result_patches)
