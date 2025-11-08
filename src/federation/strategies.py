# src/federation/strategies.py

import random
import logging
from src.federation.aggregator import aggregate_weights
from src.config_schema import AppConfig

class FedAvgStrategy:
    """
    实现了 FedAvg 策略，负责协调一轮联邦学习的完整流程。
    """
    def __init__(self, config: AppConfig, logger: logging.Logger):
        """
        初始化策略。
        """
        self.config = config
        self.logger = logger

    def select_clients(self, all_clients: list):
        """
        从所有客户端中随机选择一部分参与本轮训练。
        """
        clients_per_round = self.config.federation.clients_per_round
        num_clients = len(all_clients)
        
        clients_per_round = min(clients_per_round, num_clients)
        
        selected_clients = random.sample(all_clients, clients_per_round)
        
        return selected_clients

    def perform_round(self, global_model, all_clients: list):
        """
        执行一轮完整的联邦学习。
        """
        # 1. 选择本轮参与的客户端
        selected_clients = self.select_clients(all_clients)
        self.logger.info(f"  [策略] 本轮选择 {len(selected_clients)} 个客户端参与训练: {[c.cid for c in selected_clients]}")

        # 2. 触发所有被选中的客户端进行本地更新
        updates = []
        for client in selected_clients:
            local_weights, num_samples = client.update_model(global_model)
            updates.append((local_weights, num_samples))
            self.logger.info(f"  [策略] 收到来自客户端 {client.cid} 的更新，数据量: {num_samples}")

        # 3. 聚合所有客户端的更新
        self.logger.info("  [策略] 开始聚合所有客户端的权重...")
        aggregated_weights = aggregate_weights(updates)
        self.logger.info("  [策略] 权重聚合完成。")

        return aggregated_weights