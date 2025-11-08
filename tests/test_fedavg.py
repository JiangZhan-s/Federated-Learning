import unittest
from src.federation.strategies import FedAvg
from src.models.model import MyModel  # 假设有一个模型类
from src.data.dataset import MyDataset  # 假设有一个数据集类

class TestFedAvg(unittest.TestCase):

    def setUp(self):
        self.model = MyModel()
        self.dataset = MyDataset()
        self.fedavg = FedAvg(self.model, self.dataset)

    def test_initialization(self):
        self.assertIsNotNone(self.fedavg)
        self.assertEqual(self.fedavg.model, self.model)
        self.assertEqual(self.fedavg.dataset, self.dataset)

    def test_aggregate_updates(self):
        updates = [self.model.get_weights() for _ in range(5)]  # 模拟5个客户端的模型更新
        aggregated_weights = self.fedavg.aggregate(updates)
        self.assertIsNotNone(aggregated_weights)

    def test_fedavg_convergence(self):
        initial_weights = self.model.get_weights()
        for _ in range(10):  # 模拟10轮训练
            updates = [self.model.get_weights() for _ in range(5)]
            self.fedavg.aggregate(updates)
        final_weights = self.model.get_weights()
        self.assertNotEqual(initial_weights, final_weights)

if __name__ == '__main__':
    unittest.main()