import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.utils import loader
from patchmentation.utils import functional as F
from patchmentation import patch_augmentation
from patchmentation.utils import transform, filter, Comparator

import os
import streamlit as st
import extra_streamlit_components as stx

SAMPLE_PATCHMENTATION = 'dataset/sample_patchmentation/'
SAMPLE_PATCHMENTATION_BACKGROUND = os.path.join(SAMPLE_PATCHMENTATION, 'background/background_1.jpg')
SAMPLE_PATCHMENTATION_YOLO_IMAGES = os.path.join(SAMPLE_PATCHMENTATION, 'source/obj_train_data/')
SAMPLE_PATCHMENTATION_YOLO_ANNOTATIONS = os.path.join(SAMPLE_PATCHMENTATION, 'source/obj_train_data/')
SAMPLE_PATCHMENTATION_YOLO_NAMES = os.path.join(SAMPLE_PATCHMENTATION, 'source/obj.names')

DATASET_FORMAT_YOLO = 'YOLO'
DATASET_FORMAT_COCO = 'COCO'
DATASET_FORMAT_PASCAL_VOC = 'Pascal VOC'

def section_dataset_yolo():
    folder_images = st.text_input('Path to YOLO Images', SAMPLE_PATCHMENTATION_YOLO_IMAGES)
    folder_annotations = st.text_input('Path to YOLO Annotations', SAMPLE_PATCHMENTATION_YOLO_ANNOTATIONS)
    file_names = st.text_input('Path to YOLO Names', SAMPLE_PATCHMENTATION_YOLO_NAMES)
    return loader.load_yolo_dataset(folder_images, folder_annotations, file_names)

def section_dataset_coco():
    st.write('Not implemented yet.')

def section_dataset_pascal_voc():
    st.write('Not implemented yet.')

def section_dataset():
    st.subheader('Dataset')

    tab = stx.tab_bar(data=[
        stx.TabBarItemData(id=DATASET_FORMAT_YOLO, title=DATASET_FORMAT_YOLO, description=None),
        stx.TabBarItemData(id=DATASET_FORMAT_COCO, title=DATASET_FORMAT_COCO, description=None),
        stx.TabBarItemData(id=DATASET_FORMAT_PASCAL_VOC, title=DATASET_FORMAT_PASCAL_VOC, description=None)
    ])

    if tab == DATASET_FORMAT_YOLO:
        return section_dataset_yolo()

    if tab == DATASET_FORMAT_COCO:
        return section_dataset_coco()
    
    if tab == DATASET_FORMAT_PASCAL_VOC:
        return section_dataset_pascal_voc()

    return section_dataset_yolo()

def section_background_image():
    st.subheader('Background Image')
    file_background = st.text_input('Path to Background Image', SAMPLE_PATCHMENTATION_BACKGROUND)
    return loader.load_image(file_background)

def section_patch_augmentation(dataset, background_image):
    st.subheader('Configruation')
    cols = st.columns([2, 1, 1])
    with cols[0]:
        visibility_threshold = st.number_input('Visibility Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    patches = []
    for image_patch in dataset.image_patches:
        patches += image_patch.patches
    actions = [
        transform.Resize(width=50, aspect_ratio='auto'),
        transform.RandomScale([0.8, 1.2], [0.9, 1.1]),
        filter.FilterHeight(400, Comparator.LessEqual)
    ]
    return patch_augmentation(patches, background_image, visibility_threshold, actions=actions)
    
def section_display_result(result):
    st.subheader('Result')
    image = loader.load_image_array(result)
    image = F.convert_BGR2RGB(image)
    st.image(image, use_column_width='always')

def main():
    dataset = section_dataset()
    background_image = section_background_image()
    result = section_patch_augmentation(dataset, background_image)
    section_display_result(result)
        
if __name__ == '__main__':
    main()
