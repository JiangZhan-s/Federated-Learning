# src/utils/logger.py

import logging
import os
from datetime import datetime

def setup_logger():
    """
    设置日志记录器，使其同时输出到控制台和文件。

    返回:
    logging.Logger: 配置好的日志记录器实例。
    """
    # 创建 logs 目录（如果不存在）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成带时间戳的日志文件名
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"experiment_{current_time}.log")

    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除之前可能存在的任何处理器，以避免重复记录
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器 (FileHandler)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 创建控制台处理器 (StreamHandler)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s') # 控制台只输出消息
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
