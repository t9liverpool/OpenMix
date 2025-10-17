# OpenMix
Codes for "OpenMix+: Revisiting Data Augmentation for Open Set Recognition". The corresponding paper can be found in \[[link](https://ieeexplore.ieee.org/abstract/document/10106029)\]. <br>
* Highlights：<br>
1. Contrary to the view of "a good closed-set classifier is all you need for open set recognition" \[[link](https://arxiv.org/abs/2110.06207)\], we propose a new perspective that open set recognition requires the balance between classification and unknowns detection.
2. We propose data augmentation strategies termed "OpenMix", which could promote the model's performance on unknowns detection.
* Implimentation：<br>
This repository provides a pytorch-version implementation of OpenMix which includes three data augmentation methods, _i.e_., OpenMixup, OpenCutout, OpenCutmix. The scripts are included in **utils.py** which can be directly called in your dataloader.

* Notes：<br>
Generally, OpenMixup is robust to foregrounds without rotational symmetry, and OpenCutout is robust to images with diverse backgrounds. FineTune the sample ratio of augmentation is helpful for achieving better AUROC results. In our experiments:<br>  For CIFAR10 and Imagenet, Known:OpenMixup:OpenCutout:OpenCutmix=1:4:1:1;<br>   For MNIST, Known:OpenMixup:OpenCutout:OpenCutmix=1:0:0:1 .<br>

If you have any problems, feel free to contact me. Have fun and may it inspire your own idea :-)
