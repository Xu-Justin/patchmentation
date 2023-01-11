# [`patchmentation`](https://pypi.org/project/patchmentation/)

Patchmentation is a python library to perform patch augmentation, a data augmentation technique for object detection, that allows for the synthesis of new images by combining objects from one or more source images into a background image.

<p align="center">
  <img src="assets/patch-augmentation-flow.jpg" height="400" />
</p>

The process of patch augmentation involves extracting objects of interest from the source images, transforming them, and then pasting them onto the background image to create a composite image, therefore increasing diversity at the object level. The resulting dataset offers a greater variety of object combinations within a single image, making it more robust and accurate when training object detection models.

## Installation

The easiest way to install patchmentation is through pip.

```bash
pip install patchmentation
```

**Note: Some functionality of patchmentation might not be working on non-Linux systems.**

## External Links

* GitHub Repository: https://github.com/Xu-Justin/patchmentation

* PyPI: https://pypi.org/project/patchmentation

* Research Paper: TBA

* Benchmarking Results: https://github.com/Xu-Justin/patchmentation-yolov5

---

This project was developed as part of thesis project, Computer Science, BINUS University.
