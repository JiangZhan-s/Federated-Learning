# src/federation/strategies.py

import random
import logging
from src.federation.aggregator import aggregate_weights

class FedAvgStrategy:
    """
    实现了 FedAvg 策略，负责协调一轮联邦学习的完整流程。
    
    这个策略足够通用，同样适用于 FedProx，因为 FedProx 的服务器端聚合
    逻辑与 FedAvg 完全相同。算法的差异体现在客户端的本地训练过程中，
    这部分逻辑由 Trainer 类处理。
    """
    def __init__(self, config: dict, logger: logging.Logger):
        """
        初始化策略。

        参数:
        config (dict): 全局配置字典，需要以下键:
                       'federation': {
                           'clients_per_round': 10
                       }
        logger (logging.Logger): 用于记录日志的记录器实例。
        """
        self.config = config
        self.logger = logger

    def select_clients(self, all_clients: list):
        """
        从所有客户端中随机选择一部分参与本轮训练。

        参数:
        all_clients (list): 包含所有客户端实例的列表。

        返回:
        list: 被选中的客户端实例列表。
        """
        clients_per_round = self.config['federation']['clients_per_round']
        num_clients = len(all_clients)
        
        # 确保选择的客户端数量不超过总数
        clients_per_round = min(clients_per_round, num_clients)
        
        # 随机选择客户端
        selected_clients = random.sample(all_clients, clients_per_round)
        
        return selected_clients

    def perform_round(self, global_model, all_clients: list):
        """
        执行一轮完整的联邦学习。

        包括：选择客户端、分发模型、触发本地训练、收集更新、聚合权重。

        参数:
        global_model (torch.nn.Module): 当前的全局模型。
        all_clients (list): 包含所有客户端实例的列表。

        返回:
        collections.OrderedDict: 聚合后得到的新全局模型的状态字典。
        """
        # 1. 选择本轮参与的客户端
        selected_clients = self.select_clients(all_clients)
        self.logger.info(f"  [策略] 本轮选择 {len(selected_clients)} 个客户端参与训练: {[c.cid for c in selected_clients]}")

        # 2. 触发所有被选中的客户端进行本地更新
        #    客户端的 update_model 方法会根据配置（FedAvg 或 FedProx）
        #    自动执行正确的本地训练算法。
        updates = []
        for client in selected_clients:
            # 客户端执行本地训练并返回更新
            local_weights, num_samples = client.update_model(global_model)
            updates.append((local_weights, num_samples))
            self.logger.info(f"  [策略] 收到来自客户端 {client.cid} 的更新，数据量: {num_samples}")

        # 3. 聚合所有客户端的更新
        self.logger.info("  [策略] 开始聚合所有客户端的权重...")
        aggregated_weights = aggregate_weights(updates)
        self.logger.info("  [策略] 权重聚合完成。")

        return aggregated_weights