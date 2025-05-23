import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import CNN  # 您的CNN模型定义在model.py中
import matplotlib
import logging

# 配置日志记录
logging.basicConfig(
    filename="fgsm_attack.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体来显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def fgsm_attack(model, loss_fn, data, target, epsilon):
    """
    对输入的真实样本施加 FGSM 扰动，生成对抗样本
    参数：
      model: 神经网络模型
      loss_fn: 损失函数
      data: 原始输入数据 (Tensor)
      target: 真实标签 (Tensor)
      epsilon: 扰动强度
    返回：
      扰动后的对抗样本 (Tensor)
    """
    # 反归一化数据到原始像素范围 [0, 255]
    data_denorm = unnormalize(data)

    # 开启对输入数据梯度计算
    data_denorm.requires_grad = True

    # 前向传播：得到模型输出及损失
    output = model(data_denorm)
    loss = loss_fn(output, target)

    # 清零梯度并反向传播得到梯度信息
    model.zero_grad()
    loss.backward()
    data_grad = data_denorm.grad.data

    # FGSM 公式：生成扰动并加到原始数据上
    perturbed_data = data_denorm + epsilon * data_grad.sign()
    # 裁剪步骤：限制像素值在 [0, 255] 范围内
    perturbed_data = torch.clamp(perturbed_data, 0, 255)

    # 重新归一化到 [0, 1] 范围内
    perturbed_data = normalize(perturbed_data)

    return perturbed_data


def unnormalize(img):
    return img * 0.3081 + 0.1307  # 反归一化


def normalize(img):
    return (img - 0.1307) / 0.3081  # 归一化


def test_with_attack(model, device, test_loader, epsilon):
    """
    对整个测试集进行测试，统计干净样本和对抗样本下的识别准确率
    """
    loss_fn = nn.CrossEntropyLoss()
    correct_clean = 0
    correct_adv = 0

    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # 真实样本（干净样本）预测
        output = model(data)
        pred_clean = output.max(1, keepdim=True)[1]
        correct_clean += pred_clean.eq(target.view_as(pred_clean)).sum().item()

        # 对抗攻击：生成扰动后的对抗样本，并预测
        adv_data = fgsm_attack(model, loss_fn, data, target, epsilon)
        output_adv = model(adv_data)
        pred_adv = output_adv.max(1, keepdim=True)[1]
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    total = len(test_loader.dataset)
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
    print("扰动强度 epsilon = {:.3f}".format(epsilon))
    print("真实样本识别准确率: {}/{} ({:.2f}%)".format(correct_clean, total, clean_acc))
    print("对抗样本识别准确率: {}/{} ({:.2f}%)".format(correct_adv, total, adv_acc))
    print("-----------------------------------------------------")
    return clean_acc, adv_acc  # 新增返回值


def visualize_adversarial_examples(model, device, test_loader, epsilon, num_examples=5):
    """
    可视化展示部分样本的原始图像与对抗样本图像，
    同时显示真实标签、干净样本预测和对抗样本预测
    """
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    examples = []
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

    # 由于训练时采用 Normalize((0.1307,), (0.3081,))，因此显示前需要反标准化
    def unnormalize(img):
        return img * 0.3081 + 0.1307

    for i, (orig, adv, target, pred_clean, pred_adv) in enumerate(examples):
        # MNIST为灰度图，形状 [1,28,28]
        orig_img = unnormalize(orig[0].cpu().detach()).squeeze().numpy()
        adv_img = unnormalize(adv[0].cpu().detach()).squeeze().numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        # 修改标题：添加扰动强度ε
        plt.title(f"真实样本\n真实标签: {target.item()} 预测: {pred_clean.item()} (ε={epsilon:.2f})")
        # plt.title("真实样本\n真实标签: {} 预测: {}".format(target.item(), pred_clean.item()))
        plt.imshow(orig_img, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        # 修改标题：添加扰动强度ε
        plt.title(f"对抗样本\n真实标签: {target.item()} 预测: {pred_adv.item()} (ε={epsilon:.2f})")
        plt.imshow(adv_img, cmap="gray")
        plt.axis("off")
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
    plt.show()
    # 记录结果到日志
    logging.info("epsilon 变化 vs 识别准确率:")
    for eps, clean, adv in zip(epsilons, clean_accs, adv_accs):
        logging.info(f"Epsilon={eps:.2f}, 干净样本准确率={clean:.2f}%, 对抗样本准确率={adv:.2f}%")


def main():
    # 开始：判断设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理：与训练时一致
    transform = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 移除该行
    ])

    # 加载测试数据集，这里可以选择 MNIST 测试集（也可替换为您的手写数字数据集）
    test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 载入训练好的模型
    model = CNN().to(device)
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    # 请根据实际情况修改模型权重保存路径
    model_path = "./{}/CNN.pth".format(device_type)
    logging.info(f"正在载入 {device_type} 训练的模型...")
    print("正在载入 {} 训练的模型...".format(device_type))
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 设置 FGSM 攻击扰动强度 epsilon（可调）
    # epsilon = 0.6

    # 设置测试的epsilon范围
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # 1. 测试单个epsilon的准确率（可选）
    # test_with_attack(model, device, test_loader, epsilon=0.3)

    # 2. 绘制准确率随epsilon变化的曲线（主）
    plot_accuracy_vs_epsilon(model, device, test_loader, epsilons)

    # 3. 可视化对抗样本（可选）
    visualize_adversarial_examples(model, device, test_loader, epsilon=0.35)

    # 1. 对抗攻击：测试干净样本与对抗样本的识别准确率
    # test_with_attack(model, device, test_loader, epsilon)
    # 2. 可视化：展示部分样本的原始图像与扰动后图像
    # visualize_adversarial_examples(model, device, test_loader, epsilon, num_examples=5)


if __name__ == '__main__':
    main()