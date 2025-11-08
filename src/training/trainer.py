# src/training/trainer.py
"""
训练器模块
实现联邦学习中客户端的本地训练逻辑。
支持FedAvg和FedProx算法的本地优化。
"""

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

class Trainer:
    """
    封装了单个客户端的本地训练过程。
    支持 FedAvg 和 FedProx 两种算法的本地训练。
    """
    def __init__(self, model, local_dataset, config: dict):
        """
        初始化训练器。

        参数:
        model (torch.nn.Module): 需要训练的模型。这是从服务器传来的全局模型的深拷贝。
        local_dataset (torch.utils.data.Dataset): 分配给该客户端的本地数据集。
        config (dict): 包含训练配置的字典，需要以下键:
                       'training': {
                           'optimizer': 'SGD',
                           'batch_size': 10,
                           'local_epochs': 5,
                           'learning_rate': 0.01
                       },
                       'device': 'cuda' or 'cpu'
        """
        self.model = model
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        
        # 为本地数据创建 DataLoader
        self.train_loader = DataLoader(
            local_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        
        # 根据配置选择优化器
        optimizer_name = config['training']['optimizer']
        lr = config['training']['learning_rate']
        if optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 检查是否使用 FedProx 算法
        self.algorithm = self.config['federation'].get('algorithm', 'FedAvg')
        if self.algorithm == 'FedProx':
            # 如果是 FedProx，则保存一份初始的全局模型权重，用于计算近端项
            self.global_model_weights = {
                name: param.clone().detach() for name, param in model.named_parameters()
            }

    def _calculate_proximal_term(self):
        """
        计算 FedProx 的近端项。
        近端项 = (mu / 2) * ||本地模型权重 - 全局模型权重||^2
        """
        proximal_term = 0.0
        for name, local_param in self.model.named_parameters():
            if local_param.requires_grad:
                global_param = self.global_model_weights[name].to(self.device)
                proximal_term += ((local_param - global_param) ** 2).sum()
        
        mu = self.config['federation'].get('mu', 0.01)
        return (mu / 2) * proximal_term

    def train(self):
        """
        执行本地训练过程。

        该方法会在本地数据上训练模型指定的轮次 (local_epochs)。
        """
        # 将模型切换到训练模式，并移动到指定设备
        self.model.train()
        self.model.to(self.device)
        
        local_epochs = self.config['training']['local_epochs']
        
        # 执行多个本地轮次的训练
        for epoch in range(local_epochs):
            for data, target in self.train_loader:
                # 将数据和标签移动到指定设备
                data, target = data.to(self.device), target.to(self.device)
                
                # 1. 清零梯度
                self.optimizer.zero_grad()
                
                # 2. 前向传播
                output = self.model(data)
                
                # 3. 计算损失 (使用负对数似然损失)
                loss = F.nll_loss(output, target)
                
                # 4. 反向传播
                loss.backward()
                
                # 5. 更新模型参数
                self.optimizer.step()
        
        # 训练结束后，将模型移回 CPU，这是一种好习惯，
        # 因为模型参数的聚合通常在 CPU 上完成。
        self.model.to("cpu")
