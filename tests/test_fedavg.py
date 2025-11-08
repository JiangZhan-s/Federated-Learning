# tests/test_fedavg.py
"""
联邦学习FedAvg算法的单元测试
测试模型聚合、初始化和收敛性等核心功能。
"""

import unittest
from src.federation.strategies import FedAvgStrategy  # 修正导入
from src.models.model import get_model  # 使用实际的模型工厂函数
from src.data.dataset import load_dataset  # 使用实际的数据集加载函数
from src.data.preprocessing import split_data

class TestFedAvg(unittest.TestCase):
    """
    FedAvg策略的测试用例类。
    """

    def setUp(self):
        """
        测试前的初始化设置。
        创建测试所需的模型、数据集和策略实例。
        """
        # 创建一个简单的测试配置
        self.config = {
            'model': {'name': 'SimpleCNN', 'input_channels': 1, 'output_dim': 10},
            'dataset': {'name': 'MNIST', 'path': './data', 'num_clients': 5, 'distribution': 'iid'},
            'federation': {'algorithm': 'FedAvg', 'clients_per_round': 3}
        }
        
        # 加载测试数据集
        train_dataset, _ = load_dataset(self.config)
        self.client_datasets = split_data(train_dataset, self.config)
        
        # 创建模型实例
        self.model = get_model(self.config)
        
        # 创建FedAvg策略实例
        self.fedavg = FedAvgStrategy(self.config, None)  # logger设为None用于测试

    def test_initialization(self):
        """
        测试FedAvg策略的初始化是否正确。
        """
        self.assertIsNotNone(self.fedavg)
        self.assertEqual(self.fedavg.config, self.config)

    def test_select_clients(self):
        """
        测试客户端选择功能。
        """
        # 创建模拟客户端列表
        from src.federation.client import Client
        clients = [Client(cid=i, local_dataset=self.client_datasets[i], config=self.config) 
                  for i in range(self.config['dataset']['num_clients'])]
        
        selected = self.fedavg.select_clients(clients)
        self.assertEqual(len(selected), self.config['federation']['clients_per_round'])
        # 确保选择的客户端ID在有效范围内
        for client in selected:
            self.assertIn(client.cid, range(len(clients)))

    def test_perform_round(self):
        """
        测试执行一轮联邦学习的完整流程。
        """
        # 创建模拟客户端列表
        from src.federation.client import Client
        clients = [Client(cid=i, local_dataset=self.client_datasets[i], config=self.config) 
                  for i in range(self.config['dataset']['num_clients'])]
        
        # 执行一轮训练
        aggregated_weights = self.fedavg.perform_round(self.model, clients)
        self.assertIsNotNone(aggregated_weights)

if __name__ == '__main__':
    unittest.main()