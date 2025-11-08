# src/config.py

import yaml
from typing import Any, Dict
import os

# 获取 config.py 文件所在的目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 构建到 config/fedavg.yaml 的绝对路径
_DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, '..', 'config', 'fedavg.yaml'))


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    加载并解析 YAML 配置文件。

    参数:
    path (str): YAML 配置文件的路径。

    返回:
    dict: 包含配置参数的字典。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件未找到，请确保 '{path}' 存在。")
        exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 解析 YAML 文件时出错: {e}")
        exit(1)

# 加载全局配置，使其在项目中可以被方便地导入和使用
config = load_config()