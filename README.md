# OpenMix
Project for "OpenMix+: Revisiting Data Augmentation for Open Set Recognition". The published paper can be found in \[[link](https://ieeexplore.ieee.org/abstract/document/10106029)\]. <br>
* Highlights：<br>
1. Contrary to the popular view of "a good closed-set classifier is all you need for open set recognition" \[[link](https://arxiv.org/abs/2110.06207)\], we propose a new perspective that open set recognition requires the balance between structural risk and open space risk.
2. We theoretically and experimentally show that recent mix-based data augmentation methods (Mixup, Cutout, Cutmix) promote the closed set accuracy by sacrificing open space risk.
3. We propose **a simple but efficient strategy** termed "OpenMix" seeking a better balance between structural risk and open space risk.
* Implimentation：<br>
This repository provides a pytorch-version implementation of OpenMix which includes three data augmentation methods, _i.e_., OpenMixup, OpenCutout, OpenCutmix. All tools for implementing OpenMix are included in **util.py**
* Notes：<br>
Generally, OpenMixup is robust to foregrounds without rotational symmetry, and OpenCutout is robust to images with diverse backgrounds. Tuning the ratio of augmentation is helpful for achieving better AUROC results on your customized datasets. In our experiments, #Known:#OpenMixup:#OpenCutout:#OpenCutmix=1:4:1:1 for natural datasets like CIFAR and Imagenet, #Known:#OpenMixup:#OpenCutout:#OpenCutmix=1:0:0:1 for MNIST.<br>

Have fun and may it inspire your own idea :-)
