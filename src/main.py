# src/main.py
"""
联邦学习主程序入口
负责初始化配置、加载数据、创建客户端和服务器，并启动联邦学习模拟。
"""

import logging
import torch
import argparse
import os
from datetime import datetime
from src.config import config
from src.data.dataset import load_dataset
from src.data.preprocessing import split_data
from src.models.model import get_model
from src.federation.client import Client
from src.federation.server import Server
from src.federation.strategies import FedAvgStrategy
from src.utils.logger import setup_logger
from src.utils.history import save_history
from src.utils.plotter import plot_results

def parse_args():
    """
    解析命令行参数，允许在运行时覆盖 YAML 文件中的配置。
    返回解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="运行联邦学习模拟")
    
    # --- 数据和模型相关参数 ---
    parser.add_argument('--dataset.distribution', type=str, help="数据分布方式 (iid or non-iid)")
    parser.add_argument('--model.name', type=str, help="要使用的模型名称 (SimpleCNN or ComplexCNN)")
    
    # --- 联邦学习核心参数 ---
    parser.add_argument('--federation.algorithm', type=str, help="联邦学习算法 (FedAvg or FedProx)")
    parser.add_argument('--federation.global_rounds', type=int, help="全局通信总轮次")
    parser.add_argument('--federation.mu', type=float, help="FedProx 的近端项系数")
    
    # --- 本地训练参数 ---
    parser.add_argument('--training.learning_rate', type=float, help="学习率")
    parser.add_argument('--training.local_epochs', type=int, help="客户端本地训练轮次")
    
    # --- 系统与功能性参数 ---
    parser.add_argument('--device', type=str, help="训练设备 (cuda or cpu)")
    parser.add_argument('--load_model', type=str, default=None, help="要加载的预训练模型文件的路径")
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """
    使用命令行传入的参数更新从 YAML 文件加载的配置字典。
    支持嵌套键的更新，如 'dataset.distribution'。
    
    参数:
    - config: 配置字典
    - args: 解析后的命令行参数对象
    """
    args_dict = vars(args)
    for key, value in args_dict.items():
        # 只处理用户实际传入的参数
        if value is not None:
            # 将 'dataset.distribution' 这样的键转换为嵌套字典的更新
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
            print(f"[配置更新] 通过命令行将 '{key}' 设置为: {value}")

def main():
    """
    主函数，负责组装和启动整个联邦学习模拟流程。
    包括数据加载、模型初始化、客户端创建、服务器运行等步骤。
    """
    # 0. 解析命令行参数并更新配置
    args = parse_args()
    update_config_from_args(config, args)

    # 1. 生成一个唯一的实验时间戳，用于命名日志和图表
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 2. 设置日志记录器
    logger = setup_logger(timestamp)
    logger.info("[主程序] 开始初始化...")
    logger.info(f"[主程序] 当前最终配置: {config}")

    # 3. 加载数据集
    logger.info(f"[主程序] 正在加载数据集: {config['dataset']['name']}...")
    train_dataset, test_dataset = load_dataset(config)
    logger.info("[主程序] 数据集加载完成。")

    # 4. 划分数据给客户端
    logger.info(f"[主程序] 正在以 {config['dataset']['distribution']} 方式将数据划分给 {config['dataset']['num_clients']} 个客户端...")
    client_data_map = split_data(train_dataset, config)
    logger.info("[主程序] 数据划分完成。")

    # 5. 初始化全局模型
    logger.info(f"[主程序] 正在初始化全局模型: {config['model']['name']}...")
    global_model = get_model(config)
    logger.info("[主程序] 全局模型初始化完成。")

    # 模型"干跑" (Dry Run) - 通过一次前向传播确定模型结构
    logger.info("[主程序] 正在对全局模型进行干跑以确定最终结构...")
    dummy_input_shape = (1, config['model']['input_channels'], 28, 28)
    if config['dataset']['name'] == 'CIFAR10':
        dummy_input_shape = (1, config['model']['input_channels'], 32, 32)
    dummy_input = torch.randn(dummy_input_shape)
    global_model(dummy_input)
    logger.info("[主程序] 模型结构已根据输入尺寸最终确定。")
    
    # 如果指定了加载模型，则加载权重
    if args.load_model:
        if os.path.exists(args.load_model):
            try:
                global_model.load_state_dict(torch.load(args.load_model))
                logger.info(f"[主程序] 成功从 '{args.load_model}' 加载预训练模型。")
            except Exception as e:
                logger.error(f"[主程序] 加载模型失败: {e}")
                exit(1)
        else:
            logger.error(f"[主程序] 找不到指定的模型文件: {args.load_model}")
            exit(1)

    # 6. 创建所有客户端实例
    logger.info("[主程序] 正在创建客户端实例...")    
    all_clients = [Client(cid=i, local_dataset=client_data_map[i], config=config) for i in range(config['dataset']['num_clients'])]
    logger.info(f"[主程序] {len(all_clients)} 个客户端创建完成。")

    # 7. 初始化联邦学习策略
    #    注意：FedAvgStrategy 被复用于 FedAvg 和 FedProx
    strategy = FedAvgStrategy(config, logger)
    logger.info(f"[主程序] 已选择联邦策略: {type(strategy).__name__} (用于算法: {config['federation']['algorithm']})")

    # 8. 初始化并创建服务器
    logger.info("[主程序] 正在初始化服务器...")
    server = Server(
        global_model=global_model,
        all_clients=all_clients,
        test_dataset=test_dataset,
        strategy=strategy,
        config=config,
        logger=logger
    )
    logger.info("[主程序] 服务器初始化完成。")

    # 9. 启动服务器，开始联邦学习，并获取结果
    # 如果指定了加载模型，则在训练开始前进行一次评估
    results = server.run(initial_evaluation=bool(args.load_model))

    # 10. 保存本次实验的记录
    save_history(config, results['best_accuracy'])

    # 11. 绘制并保存结果图表
    plot_results(results, config, timestamp)

if __name__ == '__main__':
    main()