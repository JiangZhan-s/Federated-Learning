# src/federation/server.py
"""
联邦学习服务器模块
实现联邦学习的中心协调器，负责模型分发、客户端协调、模型聚合和评估。
管理整个联邦学习训练流程。
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import os

class Server:
    """
    联邦学习的中心服务器。
    负责协调整个联邦学习过程，包括模型分发、聚合和评估。
    """
    def __init__(self, global_model, all_clients: list, test_dataset, strategy, config: dict, logger: logging.Logger):
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
        
        # 创建用于保存模型的目录
        self.save_dir = "saved_models"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def run(self, initial_evaluation: bool = False):
        """
        启动并运行整个联邦学习流程。

        参数:
        initial_evaluation (bool): 是否在开始训练前进行一次初始评估。
                                   这在加载预训练模型时很有用。

        返回:
        dict: 包含 'best_accuracy', 'accuracies', 'losses' 的结果字典。
        """
        self.logger.info("[服务器] 开始联邦学习流程...")
        
        best_accuracy = 0.0
        if initial_evaluation:
            self.logger.info("[服务器] 进行初始评估...")
            best_accuracy, initial_loss = self.evaluate()
        
        accuracies = []
        losses = []
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
            
            accuracy, loss = self.evaluate()
            accuracies.append(accuracy)
            losses.append(loss)

            # 如果当前准确率超过历史最佳，则保存模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model(current_round, best_accuracy)

        self.logger.info(f"\n[服务器] 联邦学习流程结束。最佳准确率: {best_accuracy:.2f}%")
        
        return {
            "best_accuracy": best_accuracy,
            "accuracies": accuracies,
            "losses": losses
        }

    def evaluate(self):
        """
        在测试数据集上评估当前全局模型的性能。

        返回:
        tuple: (accuracy, test_loss)
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
        return accuracy, test_loss

    def save_model(self, round_num, accuracy):
        """保存当前全局模型的权重。"""
        model_name = self.config['model']['name']
        dataset_name = self.config['dataset']['name']
        
        filename = f"{model_name}_{dataset_name}_best.pth"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            torch.save(self.global_model.state_dict(), filepath)
            self.logger.info(f"[服务器] 新的最佳模型已保存! 轮次: {round_num + 1}, 准确率: {accuracy:.2f}%, 文件: {filepath}")
        except IOError as e:
            self.logger.error(f"[服务器] 保存模型失败。原因: {e}")
