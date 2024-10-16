import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import seaborn as sns
import numpy as np

# 使用 Seaborn 设置样式
sns.set(style="whitegrid")

# 定义读取数据的函数
def read_data(filename):
    with open(filename, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        y = []
        for row in datareader:
            # 假设每一行只有一个值作为y
            y.append(float(row[0]))
        return y

# 定义移动平均函数
def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

# 计算标准差的函数，保持与移动平均值的长度一致
def compute_std(data, window_size):
    std_devs = []
    for i in range(len(data) - window_size + 1):
        std_dev = np.std(data[i:i + window_size])
        std_devs.append(std_dev)
    return std_devs

# 定义绘制图形的函数
def plot_agent_rewards(agent_number, window_size=50):
    # 定义文件名
    filenames = [
        f'KD_NoFed_Agent{agent_number}.csv',
        f'KD_V3_Agent{agent_number}.csv',
        f'KD_V3_MB_Agent{agent_number}.csv'  # 添加第五类文件
    ]

    # 创建图形和轴
    y_smooth_list = []
    y_std_list = []
    for filename in filenames:
        y = read_data(filename)
        y_smooth = moving_average(y, window_size)  # 对数据进行平滑处理
        y_std = compute_std(y, window_size)  # 计算标准差

        # 使标准差的长度与平滑后的数据长度一致
        y_std = y_std[:len(y_smooth)]

        y_smooth_list.append((y_smooth, y_std))

    return y_smooth_list

# 定义绘制多个子图的函数
def plot_all_agents(num_agents, window_size=50):
    # 设置图形尺寸
    fig_width, fig_height = 20, 8
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 使用 GridSpec 控制子图大小
    gs = gridspec.GridSpec(2, 5, wspace=0.3, hspace=0.3, width_ratios=[1] * 5, height_ratios=[1] * 2)

    labels = ['WoFed', 'FedV3', 'FedV3_MB']  # 更新标签
    colors = ['#1f77b4', '#d62728', '#9467bd']  # 更新颜色

    for i in range(1, num_agents + 1):
        y_smooth_list = plot_agent_rewards(i, window_size)
        row, col = (i - 1) // 5, (i - 1) % 5
        ax = plt.subplot(gs[row, col])
        ax.set_title(f'Agent {i}')
        for j, (y_smooth, y_std) in enumerate(y_smooth_list):
            x = range(1, len(y_smooth) + 1)
            ax.plot(x, y_smooth, label=labels[j], color=colors[j])
            # 确保标准差与平滑数据对齐
            std_fill_lower = np.array(y_smooth) - np.array(y_std)
            std_fill_upper = np.array(y_smooth) + np.array(y_std)
            ax.fill_between(x, std_fill_lower, std_fill_upper, color=colors[j], alpha=0.2)
        ax.legend()
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Trajectories')
        ax.set_ylim(0, 500)  # 设置 y 轴范围

    plt.tight_layout()
    # 保存为高清PDF文件
    # plt.savefig('Cartpole_NoFed_Fed_NewMethod.pdf', format='pdf', dpi=300)
    plt.show()

# 调用绘图函数, 可以修改num_agents来绘制不同数量的智能体图形
plot_all_agents(10)
