import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from patchmentation.utils import loader
from patchmentation.utils import functional as F
from patchmentation import patch_augmentation

import os
import streamlit as st

SAMPLE_PATCHMENTATION = 'dataset/sample_patchmentation/'
SAMPLE_PATCHMENTATION_BACKGROUND = os.path.join(SAMPLE_PATCHMENTATION, 'background/background_1.jpg')
SAMPLE_PATCHMENTATION_YOLO_IMAGES = os.path.join(SAMPLE_PATCHMENTATION, 'source/obj_train_data/')
SAMPLE_PATCHMENTATION_YOLO_ANNOTATIONS = os.path.join(SAMPLE_PATCHMENTATION, 'source/obj_train_data/')
SAMPLE_PATCHMENTATION_YOLO_NAMES = os.path.join(SAMPLE_PATCHMENTATION, 'source/obj.names')

DATASET_FORMAT_YOLO = 'YOLO'
DATASET_FORMAT_COCO = 'COCO'
DATASET_FORMAT_PASCAL_VOC = 'Pascal VOC'

def section_dataset():
    st.subheader('Dataset')
    dataset_format = st.selectbox('Dataset Format', (DATASET_FORMAT_YOLO, DATASET_FORMAT_COCO, DATASET_FORMAT_PASCAL_VOC))
    
    if dataset_format == DATASET_FORMAT_YOLO:
        folder_images = st.text_input('Path to YOLO Images', SAMPLE_PATCHMENTATION_YOLO_IMAGES)
        folder_annotations = st.text_input('Path to YOLO Annotations', SAMPLE_PATCHMENTATION_YOLO_ANNOTATIONS)
        file_names = st.text_input('Path to YOLO Names', SAMPLE_PATCHMENTATION_YOLO_NAMES)
        return loader.load_yolo_dataset(folder_images, folder_annotations, file_names)
    
    if dataset_format == DATASET_FORMAT_COCO:
        st.write('Not implemented yet.')
        return None
    
    if dataset_format == DATASET_FORMAT_PASCAL_VOC:
        st.write('Not implemented yet.')
        return None

def section_background_image():
    st.subheader('Background Image')
    file_background = st.text_input('Path to Background Image', SAMPLE_PATCHMENTATION_BACKGROUND)
    return loader.load_image(file_background)

def section_patch_augmentation(dataset, background_image):
    st.subheader('Configruation')
    cols = st.columns([2, 1, 1])
    with cols[0]:
        visibility_threshold = st.number_input('Visibility Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    with cols[1]:
        min_scale_range = st.number_input('Min. Scale Range', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    with cols[2]:
        max_scale_range = st.number_input('Max. Scale Range', min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    patches = []
    for imagePatch in dataset.imagePatches:
        patches += imagePatch.patches
    return patch_augmentation(patches, background_image, visibility_threshold, scale_range=(min_scale_range, max_scale_range))
    
def section_display_result(result):
    st.subheader('Result')
    image = result.image
    image = loader.load_image_array(image)
    image = F.convert_BGR2RGB(image)
    st.image(image, use_column_width='always')

def main():
    dataset = section_dataset()
    background_image = section_background_image()
    results = section_patch_augmentation(dataset, background_image)
    section_display_result(results)
    
if __name__ == '__main__':
    main()
