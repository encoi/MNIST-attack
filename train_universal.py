import os
import torch
import argparse
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入所有模型
from models.cnn import CNN
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from models.mobilenet import mobilenet
from models.mobilenetv2 import mobilenetv2
from models.resnext import resnext50, resnext101, resnext152
from models.densenet import densenet121, densenet161, densenet169, densenet201
from models.googlenet import googlenet
from models.inceptionv4 import inceptionv4
from models.senet import seresnet18, seresnet34, seresnet50, seresnet101, seresnet152
from models.shufflenet import shufflenet
from models.shufflenetv2 import shufflenetv2
from models.squeezenet import squeezenet
from models.wideresidual_functions import wideresnet28_10, wideresnet40_10


def parse_opt():
    parser = argparse.ArgumentParser(description='Universal Model Training')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                                'mobilenet', 'mobilenetv2',
                                'resnext50', 'resnext101', 'resnext152',
                                'densenet121', 'densenet161', 'densenet169', 'densenet201',
                                'googlenet', 'inceptionv4',
                                'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
                                'shufflenet', 'shufflenetv2', 'squeezenet',
                                'wideresnet28', 'wideresnet40'],
                        help='model architecture to use')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='dataset to use for training')
    parser.add_argument('--epochs', type=int, default=30, help='input total epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='dataloader batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='optimizer weight_decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='optimizer to use (adam, sgd)')
    parser.add_argument('--scheduler', type=str, default='exp', choices=['exp', 'step', 'cosine'],
                        help='learning rate scheduler (exp, step, cosine)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='directory to save results')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args


def get_model(model_name, num_classes=10, in_channels=3):
    """获取指定的模型"""
    model_name = model_name.lower()

    # 特殊处理CNN模型，它需要1个输入通道
    if model_name == 'cnn':
        return CNN()

    # 其他模型
    if model_name == 'resnet18':
        return resnet18()
    elif model_name == 'resnet34':
        return resnet34()
    elif model_name == 'resnet50':
        return resnet50()
    elif model_name == 'resnet101':
        return resnet101()
    elif model_name == 'resnet152':
        return resnet152()
    elif model_name == 'vgg11_bn':
        return vgg11_bn()
    elif model_name == 'vgg13_bn':
        return vgg13_bn()
    elif model_name == 'vgg16_bn':
        return vgg16_bn()
    elif model_name == 'vgg19_bn':
        return vgg19_bn()
    elif model_name == 'mobilenet':
        return mobilenet()
    elif model_name == 'mobilenetv2':
        return mobilenetv2()
    elif model_name == 'resnext50':
        return resnext50()
    elif model_name == 'resnext101':
        return resnext101()
    elif model_name == 'resnext152':
        return resnext152()
    elif model_name == 'densenet121':
        return densenet121()
    elif model_name == 'densenet161':
        return densenet161()
    elif model_name == 'densenet169':
        return densenet169()
    elif model_name == 'densenet201':
        return densenet201()
    elif model_name == 'googlenet':
        return googlenet()
    elif model_name == 'inceptionv4':
        return inceptionv4()
    elif model_name == 'seresnet18':
        return seresnet18()
    elif model_name == 'seresnet34':
        return seresnet34()
    elif model_name == 'seresnet50':
        return seresnet50()
    elif model_name == 'seresnet101':
        return seresnet101()
    elif model_name == 'seresnet152':
        return seresnet152()
    elif model_name == 'shufflenet':
        return shufflenet()
    elif model_name == 'shufflenetv2':
        return shufflenetv2()
    elif model_name == 'squeezenet':
        return squeezenet()
    elif model_name == 'wideresnet28':
        return wideresnet28_10()
    elif model_name == 'wideresnet40':
        return wideresnet40_10()
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def get_optimizer(optimizer_name, model_params, lr, weight_decay):
    """获取指定的优化器"""
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def get_scheduler(scheduler_name, optimizer, epochs):
    """获取指定的学习率调度器"""
    if scheduler_name.lower() == 'exp':
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_name.lower() == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name.lower() == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")


def train(epoch, model, train_loader, optimizer, criterion, device, args):
    """训练一个epoch"""
    total_loss = 0
    total_correct = 0
    total_data = 0
    global iteration

    model.train()
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()
        # 正向传播
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total_correct += torch.eq(predicted, labels).sum().item()
        # 计算损失
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        # 反向传播
        loss.backward()
        # 权重更新
        optimizer.step()

        total_data += labels.size(0)
        iteration = iteration + 1

        train_bar.desc = f"训练 epoch[{epoch+1}/{args.epochs}] loss:{loss:.3f} iteration:{iteration}"

    # 计算平均损失和准确率
    loss = total_loss / len(train_loader)
    acc = 100 * total_correct / total_data
    train_loss.append(loss)
    train_acc.append(acc)

    print(f'训练集准确率: {acc:.2f}%')
    return loss, acc


def validate(epoch, model, test_loader, criterion, device, args):
    """验证模型"""
    total_loss = 0
    total_correct = 0
    total_data = 0

    model.eval()
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # 正向传播
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += torch.eq(predicted, labels).sum().item()

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            total_data += labels.size(0)

            # 进度条描述
            test_bar.desc = f"验证 epoch[{epoch+1}/{args.epochs}]"

        loss = total_loss / len(test_loader)
        acc = 100 * total_correct / total_data
        validate_loss.append(loss)
        validate_acc.append(acc)

        print(f'验证集准确率: {acc:.2f}%\n')
        return loss, acc


if __name__ == "__main__":
    args = parse_opt()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 确定设备
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用 {device_type} 设备: {device}")

    # 创建输出目录
    model_dir = os.path.join(args.output_dir, args.model, args.dataset)
    os.makedirs(model_dir, exist_ok=True)

    # 设置数据转换
    if args.dataset == 'mnist':
        # MNIST数据集转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 如果模型不是CNN，需要将单通道转为三通道
        if args.model.lower() != 'cnn':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道转为三通道
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
            ])

        # 加载MNIST数据集
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif args.dataset == 'cifar10':
        # CIFAR10数据集转换
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        # 如果模型是CNN，需要将三通道转为单通道
        if args.model.lower() == 'cnn':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=1),  # 将三通道转为单通道
                transforms.ToTensor(),
                transforms.Normalize((0.4914,), (0.2470,))
            ])

            test_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # 将三通道转为单通道
                transforms.ToTensor(),
                transforms.Normalize((0.4914,), (0.2470,))
            ])

        # 加载CIFAR10数据集
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # 创建模型
    model = get_model(args.model)
    model.to(device)
    print(f"创建模型: {args.model}")

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs)

    # 初始化跟踪变量
    train_loss = []
    train_acc = []
    validate_loss = []
    validate_acc = []
    iteration = 1
    best_acc = 0

    # 训练循环
    for i in range(args.epochs):
        train_loss_epoch, train_acc_epoch = train(i, model, train_loader, optimizer, criterion, device, args)
        val_loss_epoch, val_acc_epoch = validate(i, model, test_loader, criterion, device, args)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # 保存最佳模型
        is_best = val_acc_epoch > best_acc
        best_acc = max(val_acc_epoch, best_acc)

        if is_best:
            torch.save(model.state_dict(), os.path.join(model_dir, f"{args.model}_best.pth"))
            print(f"保存最佳模型，准确率: {best_acc:.2f}%")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, f"{args.model}_final.pth"))

    # 保存训练历史
    epoch_range = np.arange(1, args.epochs + 1)
    dataframe = pd.DataFrame({
        'epoch': epoch_range,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'validate_loss': validate_loss,
        'validate_accuracy': validate_acc
    })
    dataframe.to_csv(os.path.join(model_dir, "training_history.csv"), index=False)

    print(f"训练完成。最终模型保存到 {model_dir}")
    print(f"最佳验证准确率: {best_acc:.2f}%")
