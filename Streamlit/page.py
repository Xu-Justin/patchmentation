import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import section
from patchmentation import patch_augmentation

import streamlit as st

def dataset(key: str = 'page-dataset'):
    dataset = section.dataset(key=f'{key}-dataset')
    section.display_dataset(dataset, key=f'{key}-display_dataset')

def patchmentation(key: str = 'page-patchmentation'):
    dataset = section.dataset(key=f'{key}-dataset')
    background_image = section.background_image(key=f'{key}-background_image')
    actions = section.input_actions(key=f'{key}-input_actions')
    conf = section.patchmentation_configuration(key=f'{key}-patchmentation_conf')

    if dataset is None:
        st.error(f'ERROR: Invalid dataset')
        return
    
    patches = []
    for image_patch in dataset.image_patches:
        for patch in image_patch.patches:
            patches.append(patch)
    
    result_image_patch = patch_augmentation(patches, background_image, conf.get('visibility_threshold', 0.5), actions)
    section.display_image_patch(result_image_patch, key=f'{key}-display_result_image_patch')
    