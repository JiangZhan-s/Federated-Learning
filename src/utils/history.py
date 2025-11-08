# src/utils/history.py

import os
import csv
from datetime import datetime

def save_history(config: dict, best_accuracy: float):
    """
    将本次实验的配置和最终结果追加到 history.csv 文件中。

    参数:
    config (dict): 本次实验的全局配置字典。
    best_accuracy (float): 本次实验达到的最高测试准确率。
    """
    history_file = "logs/history.csv"
    file_exists = os.path.isfile(history_file)

    # 提取关键配置
    dataset_config = config.get('dataset', {})
    federation_config = config.get('federation', {})
    training_config = config.get('training', {})

    # 定义表头和当前行数据
    header = [
        "timestamp", "dataset", "distribution", "num_clients",
        "global_rounds", "clients_per_round", "local_epochs",
        "optimizer", "learning_rate", "batch_size", "best_accuracy"
    ]
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_config.get('name'),
        "distribution": dataset_config.get('distribution'),
        "num_clients": dataset_config.get('num_clients'),
        "global_rounds": federation_config.get('global_rounds'),
        "clients_per_round": federation_config.get('clients_per_round'),
        "local_epochs": training_config.get('local_epochs'),
        "optimizer": training_config.get('optimizer'),
        "learning_rate": training_config.get('learning_rate'),
        "batch_size": training_config.get('batch_size'),
        "best_accuracy": f"{best_accuracy:.2f}%"
    }

    try:
        with open(history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            # 如果文件是新创建的，则写入表头
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"实验记录已保存到 {history_file}")
    except IOError as e:
        print(f"错误: 无法写入历史记录文件 {history_file}。原因: {e}")

