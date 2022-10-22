import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.collections import Image, Dataset, ImagePatch
from patchmentation.utils import loader
from patchmentation.utils import transform
from patchmentation.utils import filter
from patchmentation.utils import Comparator

import streamlit as st
import extra_streamlit_components as stx
import numpy as np
from typing import List, Tuple, Union

SAMPLE_FOLDER_IMAGES = 'dataset/sample_patchmentation/source/obj_train_data/'
SAMPLE_FOLDER_ANNOTATIONS = 'dataset/sample_patchmentation/source/obj_train_data/'
SAMPLE_FILE_NAMES = 'dataset/sample_patchmentation/source/obj.names'

SAMPLE_BACKGROUND_IMAGE = 'dataset/sample_patchmentation/background/background_1.jpg'

YOLO_FOLDER_IMAGES = 'dataset/sample_format_yolo/obj_train_data/'
YOLO_FOLDER_ANNOTATIONS = 'dataset/sample_format_yolo/obj_train_data/'
YOLO_FILE_NAMES = 'dataset/sample_format_yolo/obj.names'

COCO_FOLDER_IMAGES = 'dataset/sample_format_coco/images/'
COCO_FILE_ANNOTATIONS = 'dataset/sample_format_coco/annotations/instances_default.json'

PASCAL_VOC_FOLDER_IMAGES = 'dataset/sample_format_pascal_voc/JPEGImages/'
PASCAL_VOC_FOLDER_ANNOTATIONS = 'dataset/sample_format_pascal_voc/Annotations/'
PASCAL_VOC_FILE_IMAGESETS = 'dataset/sample_format_pascal_voc/ImageSets/Main/default.txt'
PASCAL_VOC_FILE_CLASSES = 'dataset/sample_format_pascal_voc/classes.txt'

DATASET_SAMPLE = 'Sample'
DATASET_FORMAT_YOLO = 'YOLO'
DATASET_FORMAT_COCO = 'COCO'
DATASET_FORMAT_PASCAL_VOC = 'Pascal VOC'

TRANSFORM_RESIZE = 'Resize'
TRANSFORM_RANDOM_RESIZE = 'Random Resize'
TRANSFORM_SCALE = 'Scale'
TRANSFORM_RANDOM_SCALE = 'Random Scale'
TRANSFORM_GRAYSCALE = 'Grayscale'
TRANSFORM_RANDOM_GRAYSCALE = 'Random Grayscale'
FILTER_WIDTH = 'Filter by Width'
FILTER_HEIGHT = 'Filter by Height'

ASPECT_RATIO_NONE = None
ASPECT_RATIO_AUTO = 'auto'
ASPECT_RATIO_CUSTOM = 'custom'

COMPARATOR_EQUAL = 'Equal'
COMPARATOR_LESS_THAN = 'Less Than'
COMPARATOR_LESS_EQUAL = 'Less Equal'
COMPARATOR_GREATER_THAN = 'Greater Than'
COMPARATOR_GREATER_EQUAL = 'Greater Equal'

def dataset(key: str) -> Dataset:
    st.subheader('Dataset')
    tab = stx.tab_bar(data=[
        stx.TabBarItemData(id=DATASET_SAMPLE, title=DATASET_SAMPLE, description=None),
        stx.TabBarItemData(id=DATASET_FORMAT_YOLO, title=DATASET_FORMAT_YOLO, description=None),
        stx.TabBarItemData(id=DATASET_FORMAT_COCO, title=DATASET_FORMAT_COCO, description=None),
        stx.TabBarItemData(id=DATASET_FORMAT_PASCAL_VOC, title=DATASET_FORMAT_PASCAL_VOC, description=None)
    ])

    if tab == DATASET_SAMPLE:
        dataset = dataset_sample(f'{key}-tab_dataset_sample')

    elif tab == DATASET_FORMAT_YOLO:
        dataset = dataset_yolo(f'{key}-tab_dataset_format_yolo')

    elif tab == DATASET_FORMAT_COCO:
        dataset = dataset_coco(f'{key}-tab_dataset_format_coco')
    
    elif tab == DATASET_FORMAT_PASCAL_VOC:
        dataset = dataset_pascal_voc(f'{key}-tab_dataset_format_pascal_voc')

    else:
        dataset = dataset_sample(f'{key}-tab_dataset_sample_default')

    if dataset is None:
        st.error(f'ERROR: Invalid dataset')
        return None
    else:
        return dataset

def dataset_sample(key: str) -> Dataset:
    folder_images = st.text_input('Path to YOLO Images', SAMPLE_FOLDER_IMAGES, disabled=True, key=f'{key}-sample-folder_images')
    folder_annotations = st.text_input('Path to YOLO Annotations', SAMPLE_FOLDER_ANNOTATIONS, disabled=True, key=f'{key}-sample-folder_annotations')
    file_names = st.text_input('Path to YOLO Names', SAMPLE_FILE_NAMES, disabled=True, key=f'{key}-sample-file_names')
    return loader.load_yolo_dataset(folder_images, folder_annotations, file_names)
    
def dataset_yolo(key: str) -> Dataset:
    folder_images = st.text_input('Path to YOLO Images', YOLO_FOLDER_IMAGES, key=f'{key}-yolo-folder_images')
    folder_annotations = st.text_input('Path to YOLO Annotations', YOLO_FOLDER_ANNOTATIONS, key=f'{key}-yolo-folder_annotations')
    file_names = st.text_input('Path to YOLO Names', YOLO_FILE_NAMES, key=f'{key}-yolo-file_names')
    return loader.load_yolo_dataset(folder_images, folder_annotations, file_names)

def dataset_coco(key: str) -> Dataset:
    folder_images = st.text_input('Path to COCO Images', COCO_FOLDER_IMAGES, key=f'{key}-coco-folder_images')
    file_annotations = st.text_input('Path to COCO Annotations', COCO_FILE_ANNOTATIONS, key=f'{key}-coco-file_annotations')
    return loader.load_coco_dataset(folder_images, file_annotations)

def dataset_pascal_voc(key: str) -> Dataset:
    folder_images = st.text_input('Path to Pascal VOC Images', PASCAL_VOC_FOLDER_IMAGES, key=f'{key}-pascal-voc-folder_images')
    folder_annotations = st.text_input('Path to Pascal VOC Annotations', PASCAL_VOC_FOLDER_ANNOTATIONS, key=f'{key}-pascal-voc-folder_annotations')
    file_imagesets = st.text_input('Path to Pascal VOC Image Sets', PASCAL_VOC_FILE_IMAGESETS, key=f'{key}-pascal-voc-file_imagesets')
    file_classes = st.text_input('Path to Pascal VOC Classes', PASCAL_VOC_FILE_CLASSES, key=f'{key}-pascal-voc-file_classes')
    return loader.load_pascal_voc_dataset(folder_images, folder_annotations, file_imagesets, file_classes)

def background_image(key: str) -> Image:
    st.subheader('Background Image')
    path = st.text_input('Path to Background Image', SAMPLE_BACKGROUND_IMAGE, key=f'{key}-background_image_path')
    return loader.load_image(path)

def display_dataset(dataset: Dataset, key: str) -> None:
    st.subheader('Display Dataset')
    col_images, col_right = st.columns([5, 1])
    
    with col_right:
        classes = input_classes(dataset.classes, key=f'{key}-classes')
        index = st.number_input('Index', min_value=0, max_value=len(dataset.image_patches)-1, step=1, key=f'{key}-index_input')
        
    with col_images:
        image_array = dataset.image_patches[index].image_array(classes)
        display_image(image_array)

def display_image(image_array: np.ndarray) -> None:
    st.image(image_array, channels='BGR')

def input_classes(classes: List[str], key: str) -> List[str]:
    st.write('Classes')
    checked_classes = []
    for class_name in classes:
        if class_name is None: continue
        checked = st.checkbox(class_name, True, key=f'{key}-{class_name}')
        if checked:
            checked_classes.append(class_name)
    return checked_classes

def display_image_patch(image_patch: ImagePatch, key: str) -> None:
    st.subheader('Display Image Patch')
    col_images, col_right = st.columns([5, 1])
    
    with col_right:
        classes = {patch.class_name for patch in image_patch.patches}
        classes = input_classes(classes, key=f'{key}-classes')
        
    with col_images:
        image_array = image_patch.image_array(classes)
        display_image(image_array)

def input_actions(key: str) -> List[Union[transform.Transform, filter.Filter]]:
    st.subheader('Actions')

    actions_placeholder = st.empty()
    
    NUMBER_OF_ACTION = f'{key}-NUMBER-OF-ACTION'
    if NUMBER_OF_ACTION not in st.session_state:
        st.session_state[NUMBER_OF_ACTION] = 0
    
    col = st.columns([1, 5])

    with col[0]:
        add_action = st.button('Add Action', key=f'{key}-add-action')
        if add_action:
            st.session_state[NUMBER_OF_ACTION] += 1
    
    with col[1]:
        remove_action = st.button('Remove Action', key=f'{key}-remove-action')
        if remove_action:
            if st.session_state[NUMBER_OF_ACTION] > 0:
                st.session_state[NUMBER_OF_ACTION] -= 1

    with actions_placeholder.container():
        actions = []
        number_of_action = int(st.session_state[NUMBER_OF_ACTION])
        for i in range(number_of_action):
            action = input_action(key=f'{key}-action-{i}')
            if action is not None:
                actions.append(action)
    
    return actions

def input_action(key: str) -> Union[transform.Transform, filter.Filter]:

    options = [
        None,
        TRANSFORM_RESIZE, TRANSFORM_RANDOM_RESIZE,
        TRANSFORM_SCALE, TRANSFORM_RANDOM_SCALE,
        TRANSFORM_GRAYSCALE, TRANSFORM_RANDOM_GRAYSCALE,
        FILTER_WIDTH, FILTER_HEIGHT
    ]
    action = st.selectbox('Action', options, key=f'{key}-selectbox-action')
    
    if action == TRANSFORM_RESIZE:
       return input_transform_resize(key=f'{key}-transform-resize')
    
    if action == TRANSFORM_RANDOM_RESIZE:
       return input_transform_random_resize(key=f'{key}-transform-random-resize')
    
    if action == TRANSFORM_SCALE:
       return input_transform_scale(key=f'{key}-transform-scale')

    if action == TRANSFORM_RANDOM_SCALE:
       return input_transform_random_scale(key=f'{key}-transform-random-scale')
    
    if action == TRANSFORM_GRAYSCALE:
       return input_transform_grayscale(key=f'{key}-transform-grayscale')

    if action == TRANSFORM_RANDOM_GRAYSCALE:
       return input_transform_random_grayscale(key=f'{key}-transform-random-grayscale')
    
    if action == FILTER_WIDTH:
        return input_filter_width(key=f'{key}-filter-width')

    if action == FILTER_HEIGHT:
        return input_filter_height(key=f'{key}-filter-height')

    return None

def input_transform_resize(key: str) -> transform.Resize:
    col = st.columns([1, 1, 1, 1, 1])
    
    with col[0]:
        width = _number_input(st.text_input('Width', None, key=f'{key}-width'))
     
    with col[1]:
        height = _number_input(st.text_input('Height', None, key=f'{key}-height'))

    with col[2]:
        disabled = False
        if width is not None and height is not None: disabled = True
        aspect_ratio = st.selectbox('Aspect Ratio', [ASPECT_RATIO_NONE, ASPECT_RATIO_AUTO, ASPECT_RATIO_CUSTOM], disabled=disabled, key=f'{key}-aspect-ratio')

    with col[3]:
        disabled = False
        if width is not None and height is not None: disabled = True
        if aspect_ratio != ASPECT_RATIO_CUSTOM: disabled = True
        aspect_ratio_width = st.number_input('Aspect Ratio - Width', min_value=1, max_value=None, step=1, disabled=disabled, key=f'{key}-aspect-ratio-width')
    
    with col[4]:
        disabled = False
        if width is not None and height is not None: disabled = True
        if aspect_ratio != ASPECT_RATIO_CUSTOM: disabled = True
        aspect_ratio_height = st.number_input('Aspect Ratio - Height', min_value=1, max_value=None, step=1, disabled=disabled, key=f'{key}-aspect-ratio-height')

    if aspect_ratio == ASPECT_RATIO_CUSTOM:
        aspect_ratio = (aspect_ratio_width, aspect_ratio_height)
    
    if width is None and height is None:
        st.error(f'ERROR: Both width and height could not be None')
        return None
    else:
        return transform.Resize(width, height, aspect_ratio)
    
def input_transform_random_resize(key: str) -> transform.RandomResize:
    col_1 = st.columns([1, 1, 1, 1])
    
    with col_1[0]:
        min_width = _number_input(st.text_input('Min. Width', None, key=f'{key}-min-width'), key='width')
    
    with col_1[1]:
        max_width = _number_input(st.text_input('Max. Width', None, key=f'{key}-max-width'), key='width')

    with col_1[2]:
        min_height = _number_input(st.text_input('Min. Height', None, key=f'{key}-min-height'), key='height')
    
    with col_1[3]:
        max_height = _number_input(st.text_input('Max. Height', None, key=f'{key}-max-height'), key='height')

    width = _range_input(min_width, max_width, key='width')
    height = _range_input(min_height, max_height, key='height')

    if width is None and height is None:
        st.error(f'ERROR: Both width and height could not be None')

    col_2 = st.columns([1, 1, 1])
    
    with col_2[0]:
        disabled = False
        if width is not None and height is not None: disabled = True
        aspect_ratio = st.selectbox('Aspect Ratio', [ASPECT_RATIO_NONE, ASPECT_RATIO_AUTO, ASPECT_RATIO_CUSTOM], disabled=disabled, key=f'{key}-aspect-ratio')

    with col_2[1]:
        disabled = False
        if width is not None and height is not None: disabled = True
        if aspect_ratio != ASPECT_RATIO_CUSTOM: disabled = True
        aspect_ratio_width = st.number_input('Aspect Ratio - Width', min_value=1, max_value=None, step=1, disabled=disabled, key=f'{key}-aspect-ratio-width')
    
    with col_2[2]:
        disabled = False
        if width is not None and height is not None: disabled = True
        if aspect_ratio != ASPECT_RATIO_CUSTOM: disabled = True
        aspect_ratio_height = st.number_input('Aspect Ratio - Height', min_value=1, max_value=None, step=1, disabled=disabled, key=f'{key}-aspect-ratio-height')

    if aspect_ratio == ASPECT_RATIO_CUSTOM:
        aspect_ratio = (aspect_ratio_width, aspect_ratio_height)
    
    if width is None and height is None:
        return None
    else:
        return transform.RandomResize(width, height, aspect_ratio)

def input_transform_scale(key: str) -> transform.Scale:
    col = st.columns([1, 1, 1, 1, 1])
    
    with col[0]:
        width = _float_input(st.text_input('Scale Width', None, key=f'{key}-width'))
     
    with col[1]:
        height = _float_input(st.text_input('Scale Height', None, key=f'{key}-height'))

    with col[2]:
        disabled = False
        if width is not None and height is not None: disabled = True
        aspect_ratio = st.selectbox('Aspect Ratio', [ASPECT_RATIO_NONE, ASPECT_RATIO_AUTO, ASPECT_RATIO_CUSTOM], disabled=disabled, key=f'{key}-aspect-ratio')

    with col[3]:
        disabled = False
        if width is not None and height is not None: disabled = True
        if aspect_ratio != ASPECT_RATIO_CUSTOM: disabled = True
        aspect_ratio_width = st.number_input('Aspect Ratio - Width', min_value=1, max_value=None, step=1, disabled=disabled, key=f'{key}-aspect-ratio-width')
    
    with col[4]:
        disabled = False
        if width is not None and height is not None: disabled = True
        if aspect_ratio != ASPECT_RATIO_CUSTOM: disabled = True
        aspect_ratio_height = st.number_input('Aspect Ratio - Height', min_value=1, max_value=None, step=1, disabled=disabled, key=f'{key}-aspect-ratio-height')

    if aspect_ratio == ASPECT_RATIO_CUSTOM:
        aspect_ratio = (aspect_ratio_width, aspect_ratio_height)
    
    if width is None and height is None:
        st.error(f'ERROR: Both width and height could not be None')
        return None
    else:
        return transform.Scale(width, height, aspect_ratio)

def input_transform_random_scale(key: str) -> transform.RandomScale:
    col_1 = st.columns([1, 1, 1, 1])
    
    with col_1[0]:
        min_width = _float_input(st.text_input('Min. Scale Width', None, key=f'{key}-min-width'), key='scale width')
    
    with col_1[1]:
        max_width = _float_input(st.text_input('Max. Scale Width', None, key=f'{key}-max-width'), key='scale width')

    with col_1[2]:
        min_height = _float_input(st.text_input('Min. Scale Height', None, key=f'{key}-min-height'), key='scale height')
    
    with col_1[3]:
        max_height = _float_input(st.text_input('Max. Scale Height', None, key=f'{key}-max-height'), key='scale height')

    width = _range_input(min_width, max_width, key='scale width')
    height = _range_input(min_height, max_height, key='scale height')

    if width is None and height is None:
        st.error(f'ERROR: Both width and height could not be None')

    col_2 = st.columns([1, 1, 1])
    
    with col_2[0]:
        disabled = False
        if width is not None and height is not None: disabled = True
        aspect_ratio = st.selectbox('Aspect Ratio', [ASPECT_RATIO_NONE, ASPECT_RATIO_AUTO, ASPECT_RATIO_CUSTOM], disabled=disabled, key=f'{key}-aspect-ratio')

    with col_2[1]:
        disabled = False
        if width is not None and height is not None: disabled = True
        if aspect_ratio != ASPECT_RATIO_CUSTOM: disabled = True
        aspect_ratio_width = st.number_input('Aspect Ratio - Width', min_value=1, max_value=None, step=1, disabled=disabled, key=f'{key}-aspect-ratio-width')
    
    with col_2[2]:
        disabled = False
        if width is not None and height is not None: disabled = True
        if aspect_ratio != ASPECT_RATIO_CUSTOM: disabled = True
        aspect_ratio_height = st.number_input('Aspect Ratio - Height', min_value=1, max_value=None, step=1, disabled=disabled, key=f'{key}-aspect-ratio-height')

    if aspect_ratio == ASPECT_RATIO_CUSTOM:
        aspect_ratio = (aspect_ratio_width, aspect_ratio_height)
    
    if width is None and height is None:
        return None
    else:
        return transform.RandomScale(width, height, aspect_ratio)

def input_transform_grayscale(key: str) -> transform.Grayscale:
    return transform.Grayscale()

def input_transform_random_grayscale(key: str) -> transform.RandomGrayscale:
    p = st.number_input('Probability', min_value=0.0, max_value=1.0, value=0.5, step=0.05, key=f'{key}-p')
    return transform.RandomGrayscale(p)

def input_filter_width(key: str) -> filter.FilterWidth:
    col = st.columns([3, 1])
    
    with col[0]:
        width = st.number_input('Width', min_value=0, value=0, step=50, key=f'{key}-width')
    
    with col[1]:
        comparator = input_comparator(key=f'{key}-comparator')
    
    return filter.FilterWidth(width, comparator)

def input_filter_height(key: str) -> filter.FilterHeight:
    col = st.columns([3, 1])
    
    with col[0]:
        height = st.number_input('Height', min_value=0, value=0, step=50, key=f'{key}-height')
    
    with col[1]:
        comparator = input_comparator(key=f'{key}-comparator')
    
    return filter.FilterHeight(height, comparator)


def _number_input(number: int, key: str = 'number') -> Union[int, None]:
    if number == 'None' or number == '':
            return None
    elif number.isdecimal():
        return int(number)
    else:
        st.error(f'ERROR: Value of {key} must be integer or None')
        return None

def _float_input(real: float, key: str = 'number') -> Union[float, None]:
    if real == 'None' or real == '':
        return None
    elif _isfloat(real):
        return float(real)
    else:
        st.error(f'ERROR: Value of {key} must be float or None')
        return None

def _isfloat(real: float) -> bool:
    try:
        float(real)
        return True
    except ValueError:
        return False

def _range_input(min_number: int, max_number: int, key: str = 'number') -> Union[Tuple[int, int], None]:
    if min_number is None and max_number is None:
        return None
    if min_number is None and max_number is not None:
        st.warning(f'Value of max. {key} is ignored becuse min. {key} is None')
        return None
    if min_number is not None and max_number is None:
        st.warning(f'Value of min. {key} is ignored becuse max. {key} is None')
        return None
    if min_number > max_number:
        st.error(f'ERROR: Value of min. {key} > max. {key}')
        return None
    return (min_number, max_number)

def input_comparator(key: str) -> Comparator:
    options = [
        COMPARATOR_EQUAL,
        COMPARATOR_LESS_THAN,
        COMPARATOR_LESS_EQUAL,
        COMPARATOR_GREATER_THAN,
        COMPARATOR_GREATER_EQUAL
    ]
    
    comparator = st.selectbox('Comparator', options, key=f'{key}-comparator')
    
    if comparator == COMPARATOR_EQUAL:
        return Comparator.Equal
    
    if comparator == COMPARATOR_LESS_THAN:
        return Comparator.LessThan

    if comparator == COMPARATOR_LESS_EQUAL:
        return Comparator.LessEqual

    if comparator == COMPARATOR_GREATER_THAN:
        return Comparator.GreaterThan
    
    if comparator == COMPARATOR_GREATER_EQUAL:
        return Comparator.GreaterEqual

    raise RuntimeError(f'Unexpected comparator {comparator}')

def patchmentation_configuration(key: str):
    st.subheader('Configuration')
    conf = dict()
    visibility_threshold = st.number_input('Visibility Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.05, key=f'{key}-visibility-threshold')
    conf['visibility_threshold'] = visibility_threshold
    return conf