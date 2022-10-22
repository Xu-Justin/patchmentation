import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import page

import streamlit as st
from streamlit_option_menu import option_menu

def main():
    MENU_DATASET = 'Dataset'
    MENU_PATCHMENTATION = 'Patchmentation'

    with st.sidebar:
        menu = option_menu(
            'Main Menu',
            [
                MENU_DATASET,
                MENU_PATCHMENTATION
            ],
            default_index=0
        )

    if menu == MENU_DATASET:
        page.dataset()

    if menu == MENU_PATCHMENTATION:
        page.patchmentation()

if __name__ == '__main__':
    main()
