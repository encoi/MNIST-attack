import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models.resnet import resnet18
from torchattack import (
    FGSM, PGD, PGDL2, MIFGSM, NIFGSM, SINIFGSM, 
    VMIFGSM, VNIFGSM, DIFGSM, TIFGSM, DeepFool
)

# Set font for display
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Set model save path
model_dir = './results/resnet_cifar10'
os.makedirs('attack_results', exist_ok=True)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load test dataset
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

# 修改 DataLoader 的 num_workers 参数
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0  # 将 num_workers 改为 0 以避免多进程问题
)

# Class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Normalization function
def normalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    return (x - mean) / std

# Denormalization function
def denormalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    return x * std + mean

# Load trained model
model = resnet18()
model = model.to(device)

# Load best model weights
checkpoint = torch.load(os.path.join(model_dir, 'resnet18_cifar10_best.pth'), map_location=device)
model.load_state_dict(checkpoint['model'])
print(f"Model loaded successfully, test accuracy: {checkpoint['acc']:.2f}%, training epoch: {checkpoint['epoch']+1}")

# Define attack parameters
attacks_config = [
    (FGSM, {"eps": 8/255}, "FGSM"),
    (PGD, {"eps": 8/255, "steps": 10}, "PGD"),
    (MIFGSM, {"eps": 8/255, "steps": 10}, "MI-FGSM"),
    (NIFGSM, {"eps": 8/255, "steps": 10}, "NI-FGSM"),
    (DeepFool, {"steps": 50}, "DeepFool")
]

# Evaluate attack effectiveness
def evaluate_attack(model, attack, test_loader, num_samples=1000):
    correct = 0
    total = 0
    adv_examples = []
    
    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        adv_data = attack(data, target)
        
        # Prediction
        with torch.no_grad():
            output = model(adv_data)
            _, predicted = torch.max(output.data, 1)
            
        # Statistics
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Save some adversarial examples for visualization
        if len(adv_examples) < 5:
            for i in range(min(5 - len(adv_examples), len(data))):
                adv_examples.append((
                    data[i:i+1].detach(),
                    adv_data[i:i+1].detach(),
                    target[i:i+1],
                    predicted[i:i+1]
                ))
        
        # Stop after reaching sample limit
        if total >= num_samples:
            break
    
    # Calculate accuracy (lower means more successful attack)
    accuracy = 100.0 * correct / total
    
    return accuracy, adv_examples

# Visualize adversarial examples
def visualize_examples(examples, attack_name):
    plt.figure(figsize=(12, 10))
    
    for i, (orig, adv, target, pred) in enumerate(examples):
        # Denormalize
        orig_img = denormalize(orig[0]).cpu().detach()
        adv_img = denormalize(adv[0]).cpu().detach()
        
        # Convert to numpy array
        orig_img = orig_img.numpy().transpose(1, 2, 0)
        adv_img = adv_img.numpy().transpose(1, 2, 0)
        
        # Clip to [0,1] range
        orig_img = np.clip(orig_img, 0, 1)
        adv_img = np.clip(adv_img, 0, 1)
        
        # Calculate perturbation
        perturbation = adv_img - orig_img
        perturbation = np.abs(perturbation)
        perturbation = perturbation / perturbation.max()
        
        plt.subplot(len(examples), 3, 3*i+1)
        plt.title(f"Original Image\nLabel: {classes[target.item()]}", fontsize=12)
        plt.imshow(orig_img)
        plt.axis('off')
        
        plt.subplot(len(examples), 3, 3*i+2)
        plt.title(f"Adversarial Example\nPrediction: {classes[pred.item()]}", fontsize=12)
        plt.imshow(adv_img)
        plt.axis('off')
        
        plt.subplot(len(examples), 3, 3*i+3)
        plt.title("Perturbation Magnified", fontsize=12)
        plt.imshow(perturbation)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"attack_results/{attack_name}_examples.png")
    plt.close()

# Run all attacks and collect results
results = {}

for attack_class, attack_args, attack_name in attacks_config:
    print(f"\nTesting {attack_name} attack...")
    
    # Create attack instance
    attack = attack_class(
        model=model,
        normalize=normalize,
        device=device,
        **attack_args
    )
    
    # Evaluate attack effectiveness
    accuracy, examples = evaluate_attack(model, attack, test_loader)
    results[attack_name] = accuracy
    
    print(f"{attack_name}: Accuracy under attack = {accuracy:.2f}% (Attack success rate = {100-accuracy:.2f}%)")
    
    # Visualize adversarial examples
    visualize_examples(examples, attack_name)

# Plot attack effectiveness comparison
plt.figure(figsize=(10, 6))
attacks_names = list(results.keys())
accuracies = list(results.values())
success_rates = [100-acc for acc in accuracies]

plt.bar(attacks_names, success_rates, color='coral')
plt.xlabel('Attack Method', fontsize=12)
plt.ylabel('Attack Success Rate (%)', fontsize=12)
plt.title('Comparison of Different Attack Methods', fontsize=14)
plt.ylim(0, 100)

# Add value labels
for i, v in enumerate(success_rates):
    plt.text(i, v+1, f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("attack_results/attack_comparison.png")
plt.show()

print("Testing completed. Results saved in 'attack_results' directory.")


if __name__ == '__main__':
    # 添加 Windows 多进程支持
    import multiprocessing
    multiprocessing.freeze_support()
    
    print(f"Using device: {device}")
    
    # 加载训练好的模型
    model = resnet18()
    model = model.to(device)
    
    # 加载最佳模型权重
    checkpoint = torch.load(os.path.join(model_dir, 'resnet18_cifar10_best.pth'), map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded successfully, test accuracy: {checkpoint['acc']:.2f}%, training epoch: {checkpoint['epoch']+1}")
    
    # 运行所有攻击并收集结果
    results = {}
    
    for attack_class, attack_args, attack_name in attacks_config:
        print(f"\nTesting {attack_name} attack...")
        
        # 创建攻击实例
        attack = attack_class(
            model=model,
            normalize=normalize,
            device=device,
            **attack_args
        )
        
        # 评估攻击效果
        accuracy, examples = evaluate_attack(model, attack, test_loader)
        results[attack_name] = accuracy
        
        print(f"{attack_name}: Accuracy under attack = {accuracy:.2f}% (Attack success rate = {100-accuracy:.2f}%)")
        
        # 可视化对抗样本
        visualize_examples(examples, attack_name)
    
    # 绘制攻击效果比较图
    plt.figure(figsize=(10, 6))
    attacks_names = list(results.keys())
    accuracies = list(results.values())
    success_rates = [100-acc for acc in accuracies]
    
    plt.bar(attacks_names, success_rates, color='coral')
    plt.xlabel('Attack Method', fontsize=12)
    plt.ylabel('Attack Success Rate (%)', fontsize=12)
    plt.title('Comparison of Different Attack Methods', fontsize=14)
    plt.ylim(0, 100)
    
    # 添加数值标签
    for i, v in enumerate(success_rates):
        plt.text(i, v+1, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("attack_results/attack_comparison.png")
    plt.show()
    
    print("Testing completed. Results saved in 'attack_results' directory.")