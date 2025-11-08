# 联邦学习项目（FedAvg）

## 项目简介
本项目旨在从零开始实现基于 FedAvg 算法的联邦学习原型，包含数据处理、客户端训练、聚合策略等模块，支持在模拟环境中快速验证联邦学习流程。

## 目录结构
- `config/`：配置文件示例（如 `fedavg.yaml`）
- `docs/`：项目架构与设计说明
- `scripts/`：运行联邦仿真脚本
- `src/`：
  - `data/`：数据集加载与预处理
  - `federation/`：客户端、聚合器与策略实现
  - `models/`：模型定义
  - `training/`：本地训练逻辑
  - `main.py`：程序入口
- `tests/`：单元测试
- `requirements.txt`：依赖说明

## 快速开始
```bash
pip install -r requirements.txt
python scripts/run_federated_simulation.py --config config/fedavg.yaml
```

## 开发指南
1. 在 `models/` 中定义需要联邦训练的模型。
2. 在 `data/` 中编写数据加载与预处理流程。
3. 在 `federation/strategies.py` 中扩展聚合策略。
4. 使用 `tests/` 目录补充模块的单元测试。

## 参考资料
- [Federated Averaging (FedAvg) 论文](https://arxiv.org/abs/1602.05629)