import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.resnet import ResNet18
from my_dataset import MyMnistDataset
import os

# 设置模型保存路径
model_dir = './results/resnet'

transform = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 载入自己的数据集
dataset = MyMnistDataset(root='../my_mnist_dateset', transform=transform)
test_loader = DataLoader(dataset=dataset, shuffle=False)

# 生成ResNet神经网络并载入训练好的模型
model = ResNet18()
device_type = "GPU" if torch.cuda.is_available() else "CPU"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("loading {} trained model...".format(device_type))
model_path = os.path.join(model_dir, "ResNet18.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

def test():
    model.eval()
    correct = 0
    total = 0
    print("label       predicted")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print("  {}            {}".format(int(labels.item()), predicted.data.item()))

        print('ResNet18 trained model： accuracy on my_mnist_dataset set:%d %%' % (100 * correct / total))

if __name__ == '__main__':
    test()