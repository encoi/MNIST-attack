import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from models.resnet import resnet18

# 设置模型保存路径
model_dir = './results/resnet_cifar10'

# 配置日志记录
logging.basicConfig(
    filename=os.path.join(model_dir, "fgsm_attack.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def fgsm_attack(model, loss_fn, data, target, epsilon):
    """
    对输入的真实样本施加 FGSM 扰动，生成对抗样本
    """
    # 开启对输入数据梯度计算
    data.requires_grad = True
    
    # 前向传播
    output = model(data)
    loss = loss_fn(output, target)
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 获取数据梯度
    data_grad = data.grad.data
    
    # FGSM 公式：生成扰动并加到原始数据上
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    
    # 裁剪，确保数据在有效范围内
    perturbed_data = torch.clamp(perturbed_data, -1, 1)
    
    return perturbed_data

def test_with_attack(model, device, test_loader, epsilon):
    """
    对整个测试集进行测试，统计干净样本和对抗样本下的识别准确率
    """
    loss_fn = nn.CrossEntropyLoss()
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total += target.size(0)
        
        # 真实样本（干净样本）预测
        output = model(data)
        pred_clean = output.max(1, keepdim=True)[1]
        correct_clean += pred_clean.eq(target.view_as(pred_clean)).sum().item()
        
        # 对抗攻击：生成扰动后的对抗样本，并预测
        adv_data = fgsm_attack(model, loss_fn, data, target, epsilon)
        output_adv = model(adv_data)
        pred_adv = output_adv.max(1, keepdim=True)[1]
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
    
    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    
    # 记录到日志文件
    logging.info("-----------------------------------------------------")
    logging.info(f"扰动强度 epsilon = {epsilon:.3f}")
    logging.info(f"真实样本识别准确率: {correct_clean}/{total} ({clean_acc:.2f}%)")
    logging.info(f"对抗样本识别准确率: {correct_adv}/{total} ({adv_acc:.2f}%)")
    logging.info("-----------------------------------------------------")
    
    print("-----------------------------------------------------")
    print("对抗攻击测试结果:")
    print(f"扰动强度 epsilon = {epsilon:.3f}")
    print(f"真实样本识别准确率: {correct_clean}/{total} ({clean_acc:.2f}%)")
    print(f"对抗样本识别准确率: {correct_adv}/{total} ({adv_acc:.2f}%)")
    print("-----------------------------------------------------")
    
    return clean_acc, adv_acc

def visualize_adversarial_examples(model, device, test_loader, epsilon, num_examples=5):
    """
    可视化展示部分样本的原始图像与对抗样本图像
    """
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    examples = []
    classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    
    # 取若干个样本进行展示
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 干净样本预测
        output = model(data)
        pred_clean = output.max(1, keepdim=True)[1]
        
        # 生成对抗样本并预测
        adv_data = fgsm_attack(model, loss_fn, data, target, epsilon)
        output_adv = model(adv_data)
        pred_adv = output_adv.max(1, keepdim=True)[1]
        
        examples.append((data, adv_data, target, pred_clean, pred_adv))
        if len(examples) >= num_examples:
            break
    
    # 反归一化函数
    def denormalize(img):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
        return img * std + mean
    
    for i, (orig, adv, target, pred_clean, pred_adv) in enumerate(examples):
        # 反归一化图像
        orig_img = denormalize(orig[0]).cpu().detach()
        adv_img = denormalize(adv[0]).cpu().detach()
        
        # 转换为numpy数组用于显示
        orig_img = orig_img.numpy().transpose(1, 2, 0)
        adv_img = adv_img.numpy().transpose(1, 2, 0)
        
        # 裁剪到[0,1]范围
        orig_img = np.clip(orig_img, 0, 1)
        adv_img = np.clip(adv_img, 0, 1)
        
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.title(f"真实样本\n真实标签: {classes[target.item()]} 预测: {classes[pred_clean.item()]} (ε={epsilon:.2f})")
        plt.imshow(orig_img)
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.title(f"对抗样本\n真实标签: {classes[target.item()]} 预测: {classes[pred_adv.item()]} (ε={epsilon:.2f})")
        plt.imshow(adv_img)
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"adv_example_{i}_eps_{epsilon:.2f}.png"))
        plt.show()

def plot_accuracy_vs_epsilon(model, device, test_loader, epsilons):
    """绘制准确率随epsilon变化的曲线"""
    clean_accs = []
    adv_accs = []
    
    for eps in epsilons:
        clean_acc, adv_acc = test_with_attack(model, device, test_loader, eps)
        clean_accs.append(clean_acc)
        adv_accs.append(adv_acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, clean_accs, 'b-', label='干净样本准确率', marker='o')
    plt.plot(epsilons, adv_accs, 'r--', label='对抗样本准确率', marker='s')
    plt.title("识别准确率 vs 扰动强度")
    plt.xlabel("扰动强度 (epsilon)")
    plt.ylabel("准确率 (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "accuracy_vs_epsilon.png"))
    plt.show()
    
    # 记录结果到日志
    logging.info("epsilon 变化 vs 识别准确率:")
    for eps, clean, adv in zip(epsilons, clean_accs, adv_accs):
        logging.info(f"Epsilon={eps:.2f}, 干净样本准确率={clean:.2f}%, 对抗样本准确率={adv:.2f}%")

def main():
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载测试数据集
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )
    
    # 载入训练好的模型
    model = resnet18()
    model = model.to(device)
    
    # 加载最佳模型权重
    checkpoint = torch.load(os.path.join(model_dir, 'resnet18_cifar10_best.pth'), map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"加载模型成功，测试准确率: {checkpoint['acc']:.2f}%, 训练轮次: {checkpoint['epoch']+1}")
    
    # 设置测试的epsilon范围
    epsilons = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    # 1. 绘制准确率随epsilon变化的曲线
    plot_accuracy_vs_epsilon(model, device, test_loader, epsilons)
    
    # 2. 可视化对抗样本
    visualize_adversarial_examples(model, device, test_loader, epsilon=0.06)

if __name__ == '__main__':
    main()