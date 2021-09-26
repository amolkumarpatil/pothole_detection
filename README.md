# Pothole Detection 
## Overview

Detecting potholes on road using RGB and Depth images.
2 Neural Neteorks are used to accomplish the results, one is to propose the region of pothole and second is to classify whether proposed region is pothole or road/different object.
Model is trained on cpu due to less computation power. 

## Libraries
* `tensorflow`
* `opencv-python`
* `numpy`
* `matplotlob`
* `tqdm`
* `shutil`
* `pandas`
* `Pillow`
* `torch`
* `torchvision==0.10.0+cpu`

## Repo Overview
This repo contains all the files to train and test both neural networks.
* `train_depth.py` - This script is used to train model using RGB and Depth images using RGBD Net
* `train_classify` - This script is used to train CNN for classifying potholes
* `eval.py` - Used to test the model
* `preprocess.py` - Preprocess data for classification CNN
* `dataset.py` -  Load dataset for classification CNN
* `dataloader_rgbdsod.py` - Load dataset for RGBD Network

## Inference

Run `python eval.py` to run inference on test images

## Training

Run `python train_depth.py` to train rgbd network

Run `pythin train_classify.py` to train classification CNN

## ToDo
* Training on GPU
* ??????


