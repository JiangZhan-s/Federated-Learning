# src/federation/aggregator.py
"""
模型聚合器模块
实现联邦学习中的模型权重聚合功能。
支持FedAvg算法的加权平均聚合。
"""

from collections import OrderedDict
import torch

def aggregate_weights(updates):
    """
    根据 FedAvg 算法聚合客户端的模型更新。

    参数:
    updates (list): 一个包含客户端更新的列表。
                    列表中的每个元素都是一个元组 (state_dict, num_samples)，其中:
                    - state_dict (dict): 客户端更新后的模型权重。
                    - num_samples (int): 该客户端用于训练的样本数量。

    返回:
    collections.OrderedDict: 聚合后得到的新全局模型的状态字典。
    """
    if not updates:
        return None

    total_samples = sum(num_samples for _, num_samples in updates)
    
    if total_samples == 0:
        return updates[0][0]

    # 以第一个客户端的权重作为模板，复制其结构和内容
    aggregated_weights = OrderedDict(updates[0][0])

    # 将所有浮点型参数清零，准备累加
    for key in aggregated_weights.keys():
        if aggregated_weights[key].dtype == torch.float32:
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

    # 执行加权平均
    for client_weights, num_samples in updates:
        weight = num_samples / total_samples
        
        for key in client_weights.keys():
            # --- 关键修改 ---
            # 只对浮点类型的参数进行聚合
            if aggregated_weights[key].dtype == torch.float32:
                aggregated_weights[key] += client_weights[key] * weight
            # 对于非浮点类型的参数（如 BatchNorm 的 num_batches_tracked），
            # 我们直接采用第一个客户端的值（或者保持初始值），不进行聚合。
            # 这里的逻辑是，我们已经用第一个客户端的权重初始化了 aggregated_weights，
            # 所以非浮点参数已经有了值，我们只需在循环中跳过它们即可。

    return aggregated_weights