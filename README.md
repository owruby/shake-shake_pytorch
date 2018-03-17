# Shake-Shake regularization
PyTorch implementation of shake-shake regularization.
Author implementations is [here](https://github.com/xgastaldi/shake-shake).

## Dependencies
- python 3.5
- PyTorch 0.3.1

## Implemented Accuracy

### CIFAR-10
|Model|Method|This implemnetaion |Paper|
|:---:|:---:|:---:|:---:|
|ResNet26-2x64|S-S-I|97.07|97.02|

![CIFAR-10](checkpoint/cifar10.png)

### CIFAR-100
|Model|Method|This implemnetaion |Paper|
|:---:|:---:|:---:|:---:|
|ResNeXt29-2x4x64d|S-S-I|WIP|84.03|

## Train ResNet26-2x64d for CIFAR-10
```
python train.py --label 10 --depth 26 --w_base 64 --lr 0.1 --epochs 1800 --batch_size 64
```
