# src/data/dataset.py

from torchvision import datasets, transforms

def load_dataset(config: dict):
    """
    加载并准备指定的数据集 (训练集和测试集)。

    根据配置字典中的信息，下载、转换并返回相应的数据集。
    目前支持 MNIST 和 CIFAR-10。

    参数:
    config (dict): 包含数据集配置的字典，至少需要包含以下键:
                   'dataset': {
                       'name': 'MNIST' or 'CIFAR10',
                       'path': './data'
                   }

    返回:
    tuple: 包含两个元素的元组 (train_dataset, test_dataset)
           - train_dataset: PyTorch 训练数据集对象。
           - test_dataset: PyTorch 测试数据集对象。
    
    异常:
    ValueError: 如果配置文件中的数据集名称不被支持。
    """
    dataset_name = config['dataset']['name']
    data_path = config['dataset']['path']

    # 根据不同的数据集，定义不同的数据转换流程
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST 数据集的均值和标准差
        ])
        
        # 加载 MNIST 训练集，如果本地没有则下载
        train_dataset = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        # 加载 MNIST 测试集，如果本地没有则下载
        test_dataset = datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR-10 数据集的均值和标准差
        ])

        # 加载 CIFAR-10 训练集
        train_dataset = datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        # 加载 CIFAR-10 测试集
        test_dataset = datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    else:
        # 如果是不支持的数据集，则抛出异常
        raise ValueError(f"不支持的数据集: {dataset_name}。请在 'MNIST' 或 'CIFAR10' 中选择。")

    return train_dataset, test_dataset