import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import numpy as np
from datetime import datetime
import os

# Define the function to read data
def read_data(filename):
    with open(filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        y = []
        for row in datareader:
            # Assuming each row has only one value as y
            y.append(float(row[0]))
        return y

# Define moving average function
def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# Define function to compute standard deviation, keeping the length consistent with moving average
def compute_std(data, window_size):
    std_devs = []
    for i in range(len(data) - window_size + 1):
        std_dev = np.std(data[i:i + window_size])
        std_devs.append(std_dev)
    return std_devs

# Define function to plot agent rewards
def plot_agent_rewards(agent_number, window_size=50):
    # Define filenames for four categories of files
    filenames = [
        f'KD_Black_Box_WoFed_0827_Agent{agent_number}.csv',
        f'KD_Black_Box_SL_10_0827_Agent{agent_number}.csv',
        f'KD_Black_Box_SL_10_KDper10_Agent{agent_number}.csv',
        f'KD_Black_Box_SL_10_KDper20_Agent{agent_number}.csv'  # New fourth category
    ]

    # Create lists for smoothed data and standard deviation
    y_smooth_list = []
    y_std_list = []
    for filename in filenames:
        y = read_data(filename)
        y_smooth = moving_average(y, window_size)  # Smooth the data
        y_std = compute_std(y, window_size)  # Compute standard deviation

        # Ensure standard deviation length matches smoothed data length
        y_std = y_std[:len(y_smooth)]

        y_smooth_list.append((y_smooth, y_std))

    return y_smooth_list

def plot_selected_agents(agent_numbers, window_size=50, save_to_file=False,
                         save_path = 'E:\Paper by Jiang\FHPG_revise\Fig'):
    # 确保只选择5个智能体
    if len(agent_numbers) != 5:
        raise ValueError("Please select exactly 5 agent numbers.")

    # 设置图形大小
    fig_width, fig_height = 20, 3
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 使用 GridSpec 控制子图大小
    gs = gridspec.GridSpec(1, 5, wspace=0.3, hspace=0.2)

    labels = ['NoFed', 'FedHPK(d=5)', 'FedHPK(d=10)', 'FedHPK(d=20)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    handles = []
    for i, agent_num in enumerate(agent_numbers):
        y_smooth_list = plot_agent_rewards(agent_num, window_size)
        ax = plt.subplot(gs[i])
        ax.set_title(f'Agent {agent_num}', fontsize=10)
        for j, (y_smooth, y_std) in enumerate(y_smooth_list):
            x = range(1, len(y_smooth) + 1)
            line, = ax.plot(x, y_smooth, label=labels[j], color=colors[j], linewidth=1.5)
            std_fill_lower = np.array(y_smooth) - np.array(y_std)
            std_fill_upper = np.array(y_smooth) + np.array(y_std)
            ax.fill_between(x, std_fill_lower, std_fill_upper, color=colors[j], alpha=0.2)
            if i == 0:
                handles.append(line)
        ax.set_xlabel('Rounds', fontsize=10)
        ax.set_ylabel('Rewards', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout(pad=2.0)

    # 添加全局图例
    fig.subplots_adjust(bottom=0.27)
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10)

    if save_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'Rewards_Cartpole_Selected_{timestamp}.png'

        # 如果提供了保存路径，就使用它；否则使用当前目录
        if save_path:
            # 确保保存路径存在
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, filename)
        else:
            full_path = filename

        plt.savefig(full_path, dpi=600, bbox_inches='tight')
        print(f"Image saved to: {full_path}")
    else:
        plt.show()

# 使用示例：选择性绘制5个智能体
selected_agents = [1, 2, 4, 6, 9]  # 您可以更改这些数字来选择不同的智能体
plot_selected_agents(selected_agents, save_to_file=True)
