# src/data/preprocessing.py
"""
数据预处理模块
负责将训练数据集按照IID或Non-IID方式划分给多个客户端。
支持联邦学习中的数据分布模拟。
"""

import numpy as np
from torch.utils.data import Subset

def split_data(train_dataset, config: dict):
    """
    根据配置将训练数据集划分为多个客户端的子集。

    支持 IID (独立同分布) 和 Non-IID (非独立同分布) 两种划分策略。

    参数:
    train_dataset: 完整的 PyTorch 训练数据集。
    config (dict): 包含数据划分配置的字典，需要以下键:
                   'dataset': {
                       'num_clients': 100,
                       'distribution': 'iid' or 'non-iid'
                   }

    返回:
    dict: 一个字典，键是客户端 ID (从 0 到 num_clients-1)，
          值是分配给该客户端的数据子集 (torch.utils.data.Subset)。
    
    异常:
    ValueError: 如果配置文件中的数据分布名称不被支持。
    """
    num_clients = config['dataset']['num_clients']
    distribution = config['dataset']['distribution']
    
    num_items = len(train_dataset)
    client_data_map = {}

    if distribution == "iid":
        # IID: 随机打乱，然后平均分配
        all_indices = np.arange(num_items)
        np.random.shuffle(all_indices)
        items_per_client = num_items // num_clients
        
        for i in range(num_clients):
            start_idx = i * items_per_client
            end_idx = (i + 1) * items_per_client
            client_indices = all_indices[start_idx:end_idx]
            client_data_map[i] = Subset(train_dataset, client_indices)

    elif distribution == "non-iid":
        # Non-IID: 按标签排序，确保每个客户端只拥有有限类别的样本
        # 这是一个常见的 Non-IID 模拟方法：每个客户端只分配 2 个类别的样本
        
        # 1. 获取所有样本的标签
        labels = np.array(train_dataset.targets)
        
        # 2. 按标签对样本索引进行排序
        sorted_indices = np.argsort(labels)
        
        # 3. 将排序后的索引划分为 200 个分片 (shards)
        num_shards = 200
        shards_per_client = 2
        shard_size = num_items // num_shards
        shards = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size
            shards.append(sorted_indices[start_idx:end_idx])
        
        # 4. 为每个客户端分配 2 个分片
        shard_indices = np.arange(num_shards)
        np.random.shuffle(shard_indices)
        
        for i in range(num_clients):
            # 为客户端 i 分配两个分片的索引
            client_shard_indices = shard_indices[i * shards_per_client : (i + 1) * shards_per_client]
            
            # 将这两个分片中的所有样本索引合并
            client_indices = np.concatenate([shards[s_idx] for s_idx in client_shard_indices], axis=0)
            
            # 创建数据子集
            client_data_map[i] = Subset(train_dataset, client_indices)

    else:
        raise ValueError(f"不支持的数据分布: {distribution}。请在 'iid' 或 'non-iid' 中选择。")

    return client_data_map