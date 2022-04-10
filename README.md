# Comma10k Challenge

Let's tackle the [comma10k](https://github.com/commaai/comma10k) segmentation and have some fun.

First I will try to implement the following [paper](https://arxiv.org/pdf/2111.09957v2.pdf) and see how this is going to work out. I have chosen this one because it is small (and hopefully fast) and it has good results on common benchmarks for semantic segmentation like [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). 

Moreover I will compare it to some of the older approaches like [UNet](https://arxiv.org/abs/1505.04597) or [DeepLabV3](https://arxiv.org/abs/1706.05587) using the [PyTorch Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch) framework.


## RegSeg - Predictions

Down below you can see a few example predictions of the current RegSeg model, trained for 50 epochs without any augmentations. Results look pretty good.

<p align="center">
<img src="assets/example-0.jpg" width="100%">
</p>

<p align="center">
<img src="assets/example-1.jpg" width="100%">
</p>

<p align="center">
<img src="assets/example-2.jpg" width="100%">
</p>


## TODOs

* mixed precision training
* ~~evaluation (pixel accuracy, IoU, F1 Score)~~
* more augmentations
* ~~visualization methods~~