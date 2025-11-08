# 可扩展的联邦学习模拟框架

## 1. 项目简介

本项目是一个用 Python 和 PyTorch 实现的、模块化且可扩展的联邦学习模拟框架。它旨在提供一个清晰、易于理解和修改的平台，用于快速实现、测试和比较不同的联邦学习算法。

目前，该框架已经从一个基础的 FedAvg 实现，演变成一个支持多种算法、模型和高级功能的迷你研究工具。

## 2. 核心功能

- **多种联邦算法**:
  - ✅ **FedAvg**: 经典的联邦平均算法。
  - ✅ **FedProx**: 增加了近端项以缓解 Non-IID 数据下的客户端漂移问题。
- **灵活的数据分布**:
  - ✅ **IID**: 独立同分布，数据被随机均匀打乱并分配。
  - ✅ **Non-IID**: 非独立同分布，通过标签倾斜模拟真实世界的数据异构性。
- **可更换的模型架构**:
  - ✅ **SimpleCNN**: 一个适用于 MNIST 的简单卷积网络。
  - ✅ **ComplexCNN**: 包含批量归一化和 Dropout 的更深层网络，适用于更复杂的数据集。
- **强大的实验管理**:
  - ✅ **命令行参数**: 无需修改配置文件，通过命令行即可动态调整实验参数。
  - ✅ **自动模型保存**: 自动将在验证集上表现最佳的模型保存到 `saved_models/` 目录。
  - ✅ **模型加载**: 支持从已保存的模型文件继续训练或进行评估。
- **完善的结果追踪**:
  - ✅ **详细日志**: 为每次实验生成带时间戳的独立日志文件，存放在 `logs/` 目录。
  - ✅ **历史记录**: 自动将每次实验的核心配置和最终结果追加到 `logs/history.csv`，方便对比。
  - ✅ **结果可视化**: 自动为每次实验绘制“准确率/损失 vs. 全局轮次”的图表，并保存到 `plots/` 目录。

## 3. 目录结构
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

## 4. 安装

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd <project-folder>

# 2. (推荐) 创建并激活一个虚拟环境
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. 安装依赖
pip install -r requirements.txt
```

## 5. 开发指南

- **定义新模型**: 在 `src/models/model.py` 中添加新的模型类，并更新 `get_model` 工厂函数。
- **实现新算法**:
  - 如果算法需要修改本地训练过程 (例如 `FedProx` 的近端项), 请在 `src/training/trainer.py` 中更新 `Trainer` 类。
  - 如果算法需要新的聚合逻辑，请在 `src/federation/aggregator.py` 中添加新的聚合函数。
  - 在 `src/federation/strategies.py` 中创建新的策略类来编排新的联邦学习轮次流程。
- **添加配置文件**: 在 `config/` 目录下为新算法或实验添加新的 `.yaml` 配置文件。
- **编写单元测试**: 在 `tests/` 目录下为新功能补充单元测试。

## 6. 参考资料

- [Federated Averaging (FedAvg) 论文](https://arxiv.org/abs/1602.05629)
- [Federated Optimization in Heterogeneous Networks (FedProx) 论文](https://arxiv.org/abs/1812.06127)
