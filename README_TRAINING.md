# 模型训练指南

本项目提供了两个训练脚本，用于训练不同的深度学习模型：

1. `train_universal.py` - 通用训练脚本，支持大多数模型
2. `train_inceptionv3.py` - 专门用于训练 InceptionV3 模型的脚本

## 通用训练脚本 (train_universal.py)

这个脚本支持训练多种不同的模型架构，包括 ResNet、VGG、MobileNet 等。

### 支持的模型

- CNN (简单卷积神经网络)
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19 with BatchNorm)
- MobileNet, MobileNetV2
- ResNeXt (50, 101, 152)
- DenseNet (121, 161, 169, 201)
- GoogleNet
- InceptionV4
- SEResNet (18, 34, 50, 101, 152)
- ShuffleNet, ShuffleNetV2
- SqueezeNet
- WideResNet (28-10, 40-10)

### 使用方法

基本用法：

```bash
python train_universal.py --model resnet18 --dataset cifar10
```

完整参数列表：

```
--model: 模型架构 (默认: cnn)
--dataset: 数据集 (mnist 或 cifar10, 默认: mnist)
--epochs: 训练轮数 (默认: 30)
--batch_size: 批量大小 (默认: 64)
--lr: 学习率 (默认: 0.001)
--weight_decay: 权重衰减 (默认: 0.00001)
--optimizer: 优化器 (adam 或 sgd, 默认: adam)
--scheduler: 学习率调度器 (exp, step, cosine, 默认: exp)
--output_dir: 输出目录 (默认: ./results)
--num_workers: 数据加载线程数 (默认: 4)
--seed: 随机种子 (默认: 42)
```

### 示例

训练 ResNet-18 模型在 CIFAR10 数据集上：

```bash
python train_universal.py --model resnet18 --dataset cifar10 --epochs 100 --batch_size 128 --optimizer sgd --lr 0.1 --scheduler cosine
```

训练 MobileNetV2 模型在 MNIST 数据集上：

```bash
python train_universal.py --model mobilenetv2 --dataset mnist --epochs 50
```

## InceptionV3 训练脚本 (train_inceptionv3.py)

这个脚本专门用于训练 InceptionV3 模型，因为它需要特殊的输入尺寸 (299x299)。

### 使用方法

基本用法：

```bash
python train_inceptionv3.py --dataset cifar10
```

完整参数列表：

```
--dataset: 数据集 (目前仅支持 cifar10)
--epochs: 训练轮数 (默认: 30)
--batch_size: 批量大小 (默认: 32)
--lr: 学习率 (默认: 0.001)
--weight_decay: 权重衰减 (默认: 0.00001)
--optimizer: 优化器 (adam 或 sgd, 默认: adam)
--scheduler: 学习率调度器 (exp, step, cosine, 默认: exp)
--output_dir: 输出目录 (默认: ./results/inceptionv3)
--num_workers: 数据加载线程数 (默认: 4)
--seed: 随机种子 (默认: 42)
```

### 示例

使用 SGD 优化器训练 InceptionV3 模型：

```bash
python train_inceptionv3.py --epochs 100 --optimizer sgd --lr 0.01 --scheduler cosine
```

## 输出文件

训练完成后，脚本会在指定的输出目录中生成以下文件：

1. `{model_name}_best.pth` - 验证集上性能最好的模型权重
2. `{model_name}_final.pth` - 最终训练完成的模型权重
3. `training_history.csv` - 包含每个 epoch 的训练和验证损失、准确率的 CSV 文件

## 注意事项

1. 对于 CNN 模型，脚本会自动处理输入通道数的转换（MNIST 为单通道，CIFAR10 为三通道）
2. 对于其他模型，当在 MNIST 上训练时，脚本会自动将单通道图像转换为三通道
3. 建议在 CIFAR10 数据集上使用 SGD 优化器和余弦退火学习率调度器获得更好的性能
4. 对于较大的模型（如 ResNet-50 及以上），建议使用较小的批量大小以避免内存不足
