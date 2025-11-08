# src/main.py

import logging
import torch
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

def main():
    """
    主函数，负责组装和启动整个联邦学习模拟流程。
    """
    # 0. 生成一个唯一的实验时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. 设置日志记录器
    logger = setup_logger(timestamp)
    logger.info("[主程序] 开始初始化...")

    # 2. 加载数据集
    logger.info(f"[主程序] 正在加载数据集: {config['dataset']['name']}...")
    train_dataset, test_dataset = load_dataset(config)
    logger.info("[主程序] 数据集加载完成。")

    # 3. 划分数据给客户端
    logger.info(f"[主程序] 正在以 {config['dataset']['distribution']} 方式将数据划分给 {config['dataset']['num_clients']} 个客户端...")
    client_data_map = split_data(train_dataset, config)
    logger.info("[主程序] 数据划分完成。")

    # 4. 初始化全局模型
    logger.info(f"[主程序] 正在初始化全局模型: {config['model']['name']}...")
    global_model = get_model(config)
    logger.info("[主程序] 全局模型初始化完成。")

    # 模型“干跑” (Dry Run)
    logger.info("[主程序] 正在对全局模型进行干跑以确定最终结构...")
    dummy_input_shape = (1, config['model']['input_channels'], 28, 28)
    if config['dataset']['name'] == 'CIFAR10':
        dummy_input_shape = (1, config['model']['input_channels'], 32, 32)
    dummy_input = torch.randn(dummy_input_shape)
    global_model(dummy_input)
    logger.info("[主程序] 模型结构已根据输入尺寸最终确定。")

    # 5. 创建所有客户端实例
    logger.info("[主程序] 正在创建客户端实例...")
    all_clients = []
    for i in range(config['dataset']['num_clients']):
        client = Client(cid=i, local_dataset=client_data_map[i], config=config)
        all_clients.append(client)
    logger.info(f"[主程序] {len(all_clients)} 个客户端创建完成。")

    # 6. 初始化联邦学习策略
    strategy = FedAvgStrategy(config, logger)
    logger.info(f"[主程序] 已选择联邦策略: {type(strategy).__name__}")

    # 7. 初始化并创建服务器
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

    # 8. 启动服务器，开始联邦学习，并获取结果
    results = server.run()

    # 9. 保存本次实验的记录
    save_history(config, results['best_accuracy'])

    # 10. 绘制并保存结果图表
    plot_results(results, config, timestamp)

if __name__ == '__main__':
    main()