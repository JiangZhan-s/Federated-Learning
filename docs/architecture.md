# FedAvg 项目架构文档

本文档详细阐述了该联邦学习项目的架构设计、核心概念和执行流程。项目旨在通过 FedAvg (Federated Averaging) 算法，在一个模拟环境中实现分布式机器学习模型的协同训练。

## 1. 核心概念

项目围绕以下几个核心概念构建：

*   **全局模型 (Global Model)**: 存放在中心服务器上，是所有客户端协同训练的目标。
*   **客户端 (Client)**: 模拟独立的设备或机构，拥有本地私有数据，但从不上传数据本身。
*   **本地训练 (Local Training)**: 客户端下载当前的全局模型，并使用自己的本地数据对其进行训练。
*   **模型更新 (Model Update)**: 本地训练完成后，客户端将更新后的模型权重（而非数据）发送回服务器。
*   **聚合 (Aggregation)**: 服务器收集来自多个客户端的模型更新，并根据 FedAvg 算法（加权平均）将它们合并，生成一个更优的新版全局模型。
*   **通信轮次 (Communication Round)**: 一次完整的“分发 -> 训练 -> 聚合”流程被称为一个全局通信轮次。

## 2. 项目结构

```
fedavg-federated-learning-project/
├── CODING_PLAN.md
├── README.md
├── requirements.txt
├── config/
│   └── fedavg.yaml
├── data/
│   └── MNIST/
├── docs/
│   └── architecture.md
├── scripts/
│   └── run_federated_simulation.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── federation/
│   │   ├── __init__.py
│   │   ├── aggregator.py
│   │   ├── client.py
│   │   ├── server.py
│   │   └── strategies.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       └── __init__.py
└── tests/
    ├── __init__.py
    └── test_fedavg.py
```

## 3. 模块详解

### 3.1 配置 (`config/` 和 `src/config.py`)
*   **`config/fedavg.yaml`**: 项目的“控制面板”，以 YAML 格式定义了所有超参数，如学习率、客户端数量、数据分布方式 (IID/Non-IID) 等。
*   **`src/config.py`**: 负责安全地加载和解析 `fedavg.yaml`，并将其作为全局可访问的配置对象。

### 3.2 数据模块 (`src/data/`)
*   **`dataset.py`**: 负责从 `torchvision` 下载并加载指定的标准数据集（如 MNIST）。
*   **`preprocessing.py`**: **联邦学习模拟的核心**。它将完整的数据集根据配置（IID 或 Non-IID）模拟划分给多个独立的客户端。

### 3.3 模型模块 (`src/models/`)
*   **`model.py`**: 定义了所有参与方（服务器和客户端）共享的神经网络架构（`SimpleCNN`）。

### 3.4 训练模块 (`src/training/`)
*   **`trainer.py`**: 封装了标准的 PyTorch 训练循环。它是一个专用的“训练引擎”，负责在给定的本地数据集上训练模型。

### 3.5 联邦学习模块 (`src/federation/`)
这是实现联邦学习逻辑的核心。
*   **`client.py`**: 定义 `Client` 类。它接收全局模型，调用 `Trainer` 在其本地数据上进行训练，并返回更新后的模型权重。
*   **`aggregator.py`**: 实现 FedAvg 的核心数学逻辑——**加权平均**。它接收来自多个客户端的权重和数据量，并输出聚合后的新权重。
*   **`strategies.py`**: 定义 `FedAvgStrategy` 策略类。它负责**协调一轮**完整的联邦学习流程：选择客户端、触发它们的本地训练、收集更新，并调用聚合器。这种设计将“一轮做什么”与“做多少轮”解耦。
*   **`server.py`**: 定义 `Server` 类，是整个流程的**总控制器**。它持有全局模型，循环执行指定的全局轮次，并在每轮调用策略对象来完成训练和聚合。同时，它还负责在每轮结束后评估全局模型的性能。

### 3.6 入口与执行 (`src/main.py` 和 `scripts/`)
*   **`src/main.py`**: 程序的“总装车间”。它导入所有模块，按顺序创建配置、数据、模型、客户端、策略和服务器实例，并将它们组装起来。
*   **`scripts/run_federated_simulation.py`**: 项目的最终入口点，它只做一件事：调用 `src/main.py` 中的 `main` 函数来启动整个模拟。

## 4. 执行流程

当运行 `python scripts/run_federated_simulation.py` 时，程序按以下顺序执行：

1.  **初始化**: `main()` 函数被调用。
    *   加载 `config.yaml` 配置。
    *   加载并预处理数据集，将其划分为多个客户端的本地数据子集。
    *   创建全局模型、所有客户端实例、联邦策略 (`FedAvgStrategy`)。
    *   所有对象被组装成一个 `Server` 实例。
2.  **开始训练**: `server.run()` 被调用。
    *   服务器开始进行 `global_rounds` 中定义的全局轮次循环。
3.  **单轮循环**: 在每个全局轮次中：
    *   服务器调用 `strategy.perform_round()`。
    *   **客户端选择**: 策略从所有客户端中随机选择一部分。
    *   **模型分发与本地训练**: 策略将当前的全局模型分发给被选中的每个客户端，并调用它们的 `update_model()` 方法。
    *   **返回更新**: 每个客户端在本地训练后，返回更新后的模型权重和本地数据量。
    *   **聚合**: 策略收集所有更新，并调用 `aggregator.aggregate_weights()` 来计算新的全局模型权重。
    *   **模型更新**: 服务器接收到新的权重，并用 `load_state_dict()` 更新自己的全局模型。
4.  **评估**: 每轮结束后，服务器在独立的全局测试集上评估新模型的准确率和损失，并打印结果。
5.  **结束**: 所有全局轮次完成后，流程结束。