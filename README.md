# Mask-RCNN-TF2
This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on TensorFlow 2.2 and Keras 2.3.1. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The repository includes:
* Example of training on the metal plate object dataset
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training

# Training on Your Own Dataset - Google Colab Version 

`mask_rcnn_tf2.ipynb` shows how to train Mask R-CNN on our own dataset.

Please consider downloading the following versions: 

* pip uninstall tensorflow -y
* pip uninstall keras -y
* pip install tensorflow-gpu==2.2.0
* pip install keras==2.3.1
* pip3 install scikit-image==0.16.2
* pip3 install opencv-python

# Training on Your Own Dataset - ML Cluster Version
 ## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

