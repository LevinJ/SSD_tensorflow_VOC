# Self-Driving Car Engineer Nanodegree
# Computer Vision/Deep Learning
## Side Project: PASCAL VOC Object Recognition and Detection


Levin Jian, June 2017



### Overview
[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) is a publicly available benchmark dataset used for object recognition and detection. There are about 17k images in the dataset (VOC 2007 and VOC 2012), and contains 20 labelled classes like person, car, cat, bottle bicycle, sheep, sofa,and etc. The detectors we develooped can be used to determine what kind of objects an image contains, and where those objects are.

We used the excellent work from [here](https://github.com/balancap/SSD-Tensorflow) as our baseline. It successfully converted the original SSD detector from caffe implementation to tensorflow implementation. The goal of our project is to focus on the trainig part of the problem. Specifically, We first load the VGG16 weights trained from ImageNET into our VGG 16  part of SSD model, train SSD modle on PASCAL VOC training dataset (VOC 2007 taineval and VOC 2012 train_eval), and evaluat SSD model on PASCAL VOC testig dataset (VOC 2007 test). Evaluation metric used is mAP.

Techncially, tensorflow and slim are used as the neural network framework, and all the development is done in Python.

### Final Result

Our SSD detecotrs achieves 0.65 mAP accuracy on VOC 2007 test dataset, at the speed of 8 rames/second. Below are a few examples of detection outputs.



Here is the training/evaluation chart,

and here the loss chart.


#SSD architecture