# src/models/model.py

import torch
from torch import nn
import torch.nn.functional as F

def get_model(config: dict):
    """
    根据配置返回一个模型实例。

    参数:
    config (dict): 包含模型配置的字典。
                   'model': {
                       'name': 'SimpleCNN' or 'ComplexCNN',
                       'input_channels': 1,
                       'output_dim': 10
                   }

    返回:
    torch.nn.Module: 一个 PyTorch 模型实例。
    
    异常:
    ValueError: 如果模型名称不被支持。
    """
    model_name = config['model']['name']
    input_channels = config['model']['input_channels']
    output_dim = config['model']['output_dim']
    
    if model_name == "SimpleCNN":
        return SimpleCNN(input_channels=input_channels, output_dim=output_dim)
    elif model_name == "ComplexCNN":
        return ComplexCNN(input_channels=input_channels, output_dim=output_dim)
    else:
        raise ValueError(f"不支持的模型: {model_name}")


class SimpleCNN(nn.Module):
    """
    一个简单的卷积神经网络模型。
    适用于 MNIST 等简单图像分类任务。
    """
    def __init__(self, input_channels: int = 1, output_dim: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50) # 假设输入为 MNIST (28x28)
        self.fc2 = nn.Linear(50, output_dim)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        
        if self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 50).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class ComplexCNN(nn.Module):
    """
    一个更复杂的卷积神经网络模型。
    
    包含两个卷积块和 Dropout 层，更适用于 CIFAR-10 等复杂数据集。
    架构: ConvBlock1 -> ConvBlock2 -> Classifier
    """
    def __init__(self, input_channels: int = 3, output_dim: int = 10):
        """
        初始化模型。

        参数:
        input_channels (int): 输入图像的通道数 (例如, MNIST 为 1, CIFAR-10 为 3)。
        output_dim (int): 输出的维度，通常是类别的数量。
        """
        super(ComplexCNN, self).__init__()
        
        # 第一个卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        )
        
        # 第二个卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        )
        
        # 分类器 (全连接层)
        # 对于 32x32 的输入 (CIFAR-10), 经过两次 pooling 后是 8x8
        # 对于 28x28 的输入 (MNIST), 经过两次 pooling 后是 7x7
        # 我们将在 forward 中动态计算
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), # 暂时以 CIFAR-10 为例
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        """
        定义前向传播。
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # 动态调整分类器的输入维度
        # 这是一个小技巧，确保模型可以适应不同尺寸的输入图片
        current_features = x.shape[1] * x.shape[2] * x.shape[3]
        if self.classifier[1].in_features != current_features:
            self.classifier[1] = nn.Linear(current_features, 512).to(x.device)
            
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)
