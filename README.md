# HBONet
Official implementation of our HBONet architecture as described in [HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions](https://arxiv.org/abs/1908.03888) (ICCV'19) by [Duo Li](https://d-li14.github.io), Aojun Zhou and Anbang Yao(https://yaoanbang.github.io) on ILSVRC2012 benchmark with [PyTorch](pytorch.org) framework.

We integrate our HBO modules into the state-of-the-art MobileNetV2 backbone as a reference case. Baseline models of MobileNetV2 counterparts are available in my repository [mobilenetv2.pytorch](https://github.com/d-li14/mobilenetv2.pytorch).

# Requirements
## Dependencies
* PyTorch 1.0+
* [NVIDIA-DALI](https://github.com/NVIDIA/DALI) (in development, not recommended)
## Dataset
Download the ImageNet dataset and move validation images to labeled subfolders.
To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# Pretrained models
The following statistics are reported on the ILSVRC2012 validation set with single center crop testing.

### HBONet with a spectrum of width multipliers (Table 2)
| Architecture      | MFLOPs | Top-1 / Top-5 Acc. (%) |
| ----------------- | ------ | -------------------------- |
| [HBONet 1.0](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_1_0.pth)    | 305 | 73.1 / 91.0 |
| [HBONet 0.8](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_8.pth)    | 205 | 71.3 / 89.7 |
| [HBONet 0.5](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_5.pth)    |  96 | 67.0 / 86.9 |
| [HBONet 0.35](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_35.pth)  |  61 | 62.4 / 83.7 |
| [HBONet 0.25](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_25.pth)  |  37 | 57.3 / 79.8 |
| [HBONet 0.1](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_1.pth)    |  14 | 41.5 / 65.7 |

### HBONet 0.8 with a spectrum of input resolutions (Table 3)
| Architecture      | MFLOPs | Top-1 / Top-5 Acc. (%) |
| ----------------- | ------ | -------------------------- |
| [HBONet 0.8 224x224](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_8.pth)        | 205 | 71.3 / 89.7 |
| [HBONet 0.8 192x192](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_8_192x192.pth)| 150 | 70.0 / 89.2 |
| [HBONet 0.8 160x160](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_8_160x160.pth)| 105 | 68.3 / 87.8 |
| [HBONet 0.8 128x128](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_8_128x128.pth)|  68 | 65.5 / 85.9 |
| [HBONet 0.8 96x96](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_8_96x96.pth)    |  39 | 61.4 / 83.0 |

### HBONet 0.35 with a spectrum of input resolutions (Table 4)
| Architecture      | MFLOPs | Top-1 / Top-5 Acc. (%) |
| ----------------- | ------ | -------------------------- |
| [HBONet 0.35 224x224](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_35.pth)        | 61 | 62.4 / 83.7 |
| [HBONet 0.35 192x192](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_35_192x192.pth)| 45 | 60.9 / 82.6 |
| [HBONet 0.35 160x160](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_35_160x160.pth)| 31 | 58.6 / 80.7 |
| [HBONet 0.35 128x128](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_35_128x128.pth)| 21 | 55.2 / 78.0 |
| [HBONet 0.35 96x96](https://github.com/d-li14/HBONet/blob/master/pretrained/hbonet_0_35_96x96.pth)    | 12 | 50.3 / 73.8 |

### HBONet with different width multipliers and different input resolutions (Table 5)
| Architecture      | MFLOPs | Top-1 / Top-5 Acc. (%) |
| ----------------- | ------ | -------------------------- |
| [HBONet 0.5 224x224](https://github.com/d-li14/HBONet/blob/master/pretrained/variant/hbonet_0_5_8_divisible.pth) |  98 | 67.7 / 87.4 |
| [HBONet 0.6 192x192](https://github.com/d-li14/HBONet/blob/master/pretrained/variant/hbonet_0_6_192x192.pth)     | 108 | 67.3 / 87.3 |

### HBONet 0.25 variants with different down-sampling and up-sampling rates (Table 6)
| Architecture      | MFLOPs | Top-1 / Top-5 Acc. (%) |
| ----------------- | ------ | -------------------------- |
| [HBONet(2x) 0.25](https://github.com/d-li14/HBONet/blob/master/pretrained/variant/hbonet_0_25_8_divisible.pth)    | 44 | 58.3 / 80.6 |
| [HBONet(4x) 0.25](https://github.com/d-li14/HBONet/blob/master/pretrained/variant/hbonet_0_25_4x_8_divisible.pth) | 45 | **59.3 / 81.4** |
| [HBONet(8x) 0.25](https://github.com/d-li14/HBONet/blob/master/pretrained/variant/hbonet_0_25_8x_8_divisible.pth) | 45 | 58.2 / 80.4 |

Taking HBONet 1.0 as an example, pretrained models can be easily imported using the following lines and then finetuned for other vision tasks or utilized in resource-aware platforms. (To create variant models in Table 5 & 6, it is necessary to make slight modifications following the instructions in the docstrings of the [model file](https://github.com/d-li14/HBONet/blob/master/models/imagenet/hbonet.py) in advance.)

```python
from models.imagenet import hbonet

net = hbonet()
net.load_state_dict(torch.load('pretrained/hbonet_1_0.pth'))
```

# Usage
## Training
Configuration to reproduce our reported results, totally the same as [mobilenetv2.pytorch](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/README.md#training) for fair comparison.

* *batch size* 256
* *epoch* 300
* *learning rate* 0.05
* *LR decay strategy* cosine
* *weight decay* 0.00004

```shell
python imagenet.py \
    -a hbonet \
    -d <path-to-ILSVRC2012-data> \
    --epochs 150 \
    --lr-decay cos \
    --lr 0.05 \
    --wd 4e-5 \
    -c <path-to-save-checkpoints> \
    --width-mult <width-multiplier> \
    --input-size <input-resolution> \
    -j <num-workers>
```

## Test
```shell
python imagenet.py \
    -a hbonet \
    -d <path-to-ILSVRC2012-data> \
    --weight <pretrained-pth-file> \
    --width-mult <width-multiplier> \
    --input-size <input-resolution> \
    -e
```

# Citations
If you find our work useful in your research, please consider citing:
```
@InProceedings{Li_2019_ICCV,
author = {Li, Duo and Zhou, Aojun and Yao, Anbang},
title = {HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```
