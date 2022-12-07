# Mask-RCNN-TF2
This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on TensorFlow 2.2 and Keras 2.3.1. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample](assets/street.png)

The repository includes:
* Example of training on the metal plate object dataset
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training

Please consider downloading the following versions: 

pip uninstall tensorflow -y
pip uninstall keras -y
pip install tensorflow-gpu==2.2.0
pip install keras==2.3.1
pip3 install scikit-image==0.16.2
pip3 install opencv-python
