# scripts/run_federated_simulation.py

import sys
import os

# 将项目根目录添加到 Python 路径中
# 这确保了无论从哪里运行此脚本，`src` 模块都可以被正确导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import main

if __name__ == '__main__':
    """
    程序的最终入口点。
    """
    main()