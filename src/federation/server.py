# src/federation/server.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging

class Server:
    """
    联邦学习的中心服务器。
    负责协调整个联邦学习过程，包括模型分发、聚合和评估。
    """
    def __init__(self, global_model, all_clients: list, test_dataset, strategy, config: dict, logger: logging.Logger):
        """
        初始化服务器。
        """
        self.global_model = global_model
        self.all_clients = all_clients
        self.strategy = strategy
        self.config = config
        self.logger = logger
        self.device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )

    def run(self):
        """
        启动并运行整个联邦学习流程。

        返回:
        float: 整个训练过程中达到的最高测试准确率。
        """
        self.logger.info("[服务器] 开始联邦学习流程...")
        
        best_accuracy = 0.0
        global_rounds = self.config['federation']['global_rounds']
        
        for current_round in range(global_rounds):
            self.logger.info(f"\n===== 全局轮次 {current_round + 1}/{global_rounds} =====")
            
            aggregated_weights = self.strategy.perform_round(
                global_model=self.global_model,
                all_clients=self.all_clients
            )
            
            if aggregated_weights:
                self.global_model.load_state_dict(aggregated_weights)
                self.logger.info("[服务器] 全局模型已更新。")
            
            accuracy = self.evaluate()
            if accuracy > best_accuracy:
                best_accuracy = accuracy

        self.logger.info(f"\n[服务器] 联邦学习流程结束。最佳准确率: {best_accuracy:.2f}%")
        return best_accuracy

    def evaluate(self):
        """
        在测试数据集上评估当前全局模型的性能。

        返回:
        float: 当前模型在测试集上的准确率。
        """
        self.global_model.eval()
        self.global_model.to(self.device)
        
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        self.logger.info(f"[服务器评估] 测试集平均损失: {test_loss:.4f}, 准确率: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)")
        
        self.global_model.to("cpu")
        return accuracy