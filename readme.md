# Model Compression（Pytorch）

This repository provides a toy tutorial of model compression including network pruning, knowledge distillation and quantization (MNN). Most code are originally from other repositories, while i modified on my experiment.

Base Model: VGG16, ResNet34

Dataset: CIFAR10

## Dependencies

torch v1.7	torchvision v0.4.2

## Attention

Before running, you need to modify the model_path (such paths) in train.py, resprune.py and so on.

## Network Pruning（BN Pruning）

Example: ResNet34（VGG same）

### Step1.  Sparse Training

```python
python train.py --arch resnet --depth 36 -sr --s 0.001 --lr 0.01			resnet34 sparse train 
```

### Step2. Pruning

```python
python resprune.py --depth 36 --percent 0.5 			resnet34, prune ratio 0.7
```

### Result

|                                         | Accuracy | params (million / MB) | Infer Time (ms) |
| :-------------------------------------: | :------: | :-------------------: | :-------------: |
|                ResNet34                 |  90.89%  |    406.76K / 3.2MB    |     13.6ms      |
| ResNet34 - Sparse Train (scale = 0.001) |  89.88%  |    406.76K / 3.2MB    |     14.7ms      |
|   ResNet34 - Sparse Train - prune0.7    |  24.3%   |   145.81K / 633.9KB   |     11.8ms      |
|   ResNet34 - Sparse Train - prune0.5    |  89.76%  |   225.91K / 946.9KB   |     12.5ms      |



## Knowledge Distillation（Hinton）

implement follow [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

Teacher Model: ResNet 151
Student Model: ResNet34 after pruning

### Step1. train a teacher model 

```python
python train.py --arch resnet --depth 92 --lr 0.01
```

### Step2. KD Train

```python
python resnet_kd_train.py --lr 0.01
```

### Result

|                                                              | Accuracy |
| :----------------------------------------------------------: | :------: |
|                           ResNet92                           |  92.04%  |
|                           ResNet34                           |  90.89%  |
| ResNet34 - Sparse Train - prune0.5（alpha=0.9, lr=0.01, temperature=3） |  91.85%  |
| ResNet34 - Sparse Train - prune0.7（alpha=0.9, lr=0.01, temperature=3） |  91.72%  |



## Quantization

Tools: MNN

### Step1. Convert Model（pth => onnx => mnn）

```
python convert_model.py
YOUR_PATH/MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz
```

### Step2. Quantize MNN Model

```
YOUR_PATH/quantized.out mnn_model_path quant_model_path ./model/weight/quant_config.json
```

|                     | Accuracy | Infer Time | Params  |
| :-----------------: | :------: | :--------: | :-----: |
|    ResNet（pth）    |  91.85%  |   12.5ms   | 946.9KB |
|    ResNet（mnn）    |  91.85%  |   2.04ms   |  926KB  |
| ResNet（quant mnn） |  90.0%   |   4.9ms    |  315KB  |

### TO SOLVE

1. quant_model Accuracy decrease about 2%（not find reason yet）
2. quant_model infer time longer than mnn model（only test on mac, maybe platform problem not support int8 operation?）



## Reference

[network slimming](https://github.com/Eric-mingjie/network-slimming)

[Knowledge Distillation](https://github.com/peterliht/knowledge-distillation-pytorch)

