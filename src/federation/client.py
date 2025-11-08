# src/federation/client.py

import copy
from src.training.trainer import Trainer

class Client:
    """
    模拟一个联邦学习的参与方（客户端）。
    """
    def __init__(self, cid: int, local_dataset, config: dict):
        """
        初始化客户端。

        参数:
        cid (int): 客户端的唯一标识符。
        local_dataset (torch.utils.data.Dataset): 分配给该客户端的本地数据集。
        config (dict): 全局配置字典。
        """
        self.cid = cid
        self.local_dataset = local_dataset
        self.config = config

    def update_model(self, global_model):
        """
        使用本地数据更新模型。

        这个方法是客户端的核心功能。它接收全局模型，在本地进行训练，
        然后返回更新后的模型权重和本地数据集的大小。

        参数:
        global_model (torch.nn.Module): 从服务器接收的当前全局模型。

        返回:
        tuple: 包含两个元素的元组:
               - (dict): 更新后的本地模型的状态字典 (state_dict)。
               - (int): 本地数据集中的样本数量。
        """
        # 1. 创建一个全局模型的深拷贝，作为本地模型
        #    这是非常重要的一步，确保每个客户端的训练不会相互影响，
        #    也不会直接修改传入的全局模型对象。
        local_model = copy.deepcopy(global_model)

        # 2. 使用本地模型、本地数据和配置来初始化训练器
        trainer = Trainer(
            model=local_model,
            local_dataset=self.local_dataset,
            config=self.config
        )

        # 3. 执行本地训练
        trainer.train()

        # 4. 返回更新后的本地模型权重和数据量
        #    服务器将使用这些信息进行加权平均。
        return local_model.state_dict(), len(self.local_dataset)