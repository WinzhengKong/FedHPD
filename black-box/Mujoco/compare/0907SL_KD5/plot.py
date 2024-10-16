import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取两个 CSV 文件
data1 = pd.read_csv('rewards_WoFed_0831.csv')
data2 = pd.read_csv('rewards_Fed_0901.csv')

# 获取所有智能体的奖励值列名
agent_columns = [f'Agent {i}' for i in range(1, 11)]

# 确保所有序列长度一致
min_length = min(len(data1), len(data2))
data1 = data1[agent_columns].iloc[:min_length]
data2 = data2[agent_columns].iloc[:min_length]

# 定义移动平均函数和标准差函数
def moving_average(data, window_size):
    return data.rolling(window_size).mean()

def compute_std(data, window_size):
    return data.rolling(window_size).std()

# 计算每个智能体的移动平均和标准差
window_size = 50
ma_1 = data1.apply(moving_average, window_size=window_size)
std_1 = data1.apply(compute_std, window_size=window_size)
ma_2 = data2.apply(moving_average, window_size=window_size)
std_2 = data2.apply(compute_std, window_size=window_size)

# 绘制图形
fig, axs = plt.subplots(2, 5, figsize=(30, 12))
axs = axs.flatten()

for i, agent in enumerate(agent_columns):
    axs[i].plot(ma_1[agent], label=f'{agent} WoFed')
    axs[i].plot(ma_2[agent], label=f'{agent} FedBPG')
    axs[i].fill_between(range(min_length), ma_1[agent] - std_1[agent], ma_1[agent] + std_1[agent], alpha=0.2)
    axs[i].fill_between(range(min_length), ma_2[agent] - std_2[agent], ma_2[agent] + std_2[agent], alpha=0.2)
    axs[i].set_title(agent)
    axs[i].set_xlabel('Episode')
    axs[i].set_ylabel('Reward')
    axs[i].legend()

plt.tight_layout()
plt.show()
