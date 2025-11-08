# src/utils/plotter.py

import matplotlib.pyplot as plt
import os

def plot_results(results: dict, config: dict, timestamp: str):
    """
    绘制并保存实验结果图表（准确率和损失）。

    参数:
    results (dict): 包含 'accuracies' 和 'losses' 列表的字典。
    config (dict): 全局配置字典，用于在图表标题中显示关键信息。
    timestamp (str): 用于命名图表文件的时间戳。
    """
    # 创建 plots 目录（如果不存在）
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 提取关键配置用于标题
    dataset_name = config['dataset']['name']
    distribution = config['dataset']['distribution']
    model_name = config['model']['name']
    
    # 创建一个包含两个子图的图窗
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(f'FedAvg Performance\n(Dataset: {dataset_name}, Distribution: {distribution}, Model: {model_name})', fontsize=16)

    # 绘制准确率曲线
    rounds = range(1, len(results['accuracies']) + 1)
    ax1.plot(rounds, results['accuracies'], marker='o', linestyle='-', color='b')
    ax1.set_title('Accuracy vs. Global Rounds')
    ax1.set_xlabel('Global Rounds')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.grid(True)

    # 绘制损失曲线
    ax2.plot(rounds, results['losses'], marker='x', linestyle='--', color='r')
    ax2.set_title('Loss vs. Global Rounds')
    ax2.set_xlabel('Global Rounds')
    ax2.set_ylabel('Average Test Loss')
    ax2.grid(True)

    # 调整子图间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图表
    plot_filename = os.path.join(plots_dir, f"results_{timestamp}.png")
    try:
        plt.savefig(plot_filename)
        print(f"结果图表已保存到: {plot_filename}")
    except IOError as e:
        print(f"错误: 无法保存图表文件。原因: {e}")
    
    plt.close() # 关闭图表以释放内存