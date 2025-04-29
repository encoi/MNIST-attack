import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from models.resnet import resnet18

def parse_opt():
    parser = argparse.ArgumentParser(description='ResNet-CIFAR10')
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--model_dir', type=str, default='./results/resnet_cifar10', help='模型保存目录')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集保存目录')
    
    args = parser.parse_args()
    return args

def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    train_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        train_bar.desc = f"训练轮次 [{epoch+1}/{args.epochs}] 损失: {train_loss/(batch_idx+1):.3f} 准确率: {100.*correct/total:.2f}%"
    
    return train_loss/len(train_loader), 100.*correct/total

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(test_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            test_bar.desc = f"测试轮次 [{epoch+1}/{args.epochs}] 损失: {test_loss/(batch_idx+1):.3f} 准确率: {100.*correct/total:.2f}%"
    
    # 保存最佳模型
    acc = 100.*correct/total
    if acc > best_acc[0]:
        print(f'保存最佳模型: {acc:.2f}%')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.model_dir, 'resnet18_cifar10_best.pth'))
        best_acc[0] = acc
    
    return test_loss/len(test_loader), 100.*correct/total

if __name__ == "__main__":
    args = parse_opt()
    
    # 创建保存模型的目录
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据转换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 显示数据集信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数量: {len(train_dataset.classes)}")
    print(f"类别名称: {train_dataset.classes}")
    
    # 可视化一些样本
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:4]))
    print(' '.join(f'{train_dataset.classes[labels[j]]}' for j in range(4)))
    
    # 创建模型
    model = resnet18()
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练和测试
    best_acc = [0]  # 使用列表以便在函数内部修改
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        scheduler.step()
        
        # 记录损失和准确率
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # 打印当前学习率
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'resnet18_cifar10_final.pth'))
    
    # 保存训练记录
    epochs = np.arange(1, args.epochs + 1)
    dataframe = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_accuracy': train_accs,
        'test_accuracy': test_accs
    })
    dataframe.to_csv(os.path.join(args.model_dir, "training_history.csv"), index=False)
    
    # 绘制损失和准确率曲线
    plt.figure(figsize=(16, 6))  # 进一步增加宽度
    
    # 设置matplotlib使用英文字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r-', label='Validation Accuracy') 
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'training_curves.png'), 
               dpi=300, 
               bbox_inches='tight',
               facecolor='white')  # 添加白色背景
    plt.close()  # 明确关闭图形
    
    print(f"训练完成! 最佳测试准确率: {best_acc[0]:.2f}%")