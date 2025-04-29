import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from matplotlib.font_manager import FontProperties

def recreate_training_curves():
    """重新创建训练曲线图，确保中文字符正确显示"""
    # 设置路径
    model_dir = './results/resnet_cifar10'
    csv_path = os.path.join(model_dir, 'training_history.csv')
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"训练历史文件不存在: {csv_path}")
        return
    
    # 读取训练历史数据
    try:
        data = pd.read_csv(csv_path)
        epochs = data['epoch'].values
        train_losses = data['train_loss'].values
        test_losses = data['test_loss'].values
        train_accs = data['train_accuracy'].values
        test_accs = data['test_accuracy'].values
        
        # 设置matplotlib使用英文字体
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        plt.figure(figsize=(16, 6))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs, test_accs, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局并保存
        plt.tight_layout()
        output_path = os.path.join(model_dir, 'training_curves_fixed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已成功创建修复后的训练曲线图: {output_path}")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")

if __name__ == "__main__":
    recreate_training_curves()