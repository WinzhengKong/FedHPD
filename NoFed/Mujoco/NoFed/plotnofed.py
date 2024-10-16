import pandas as pd
import matplotlib.pyplot as plt
import glob

# 读取以 `rewards_Agent_10` 开头的所有文件
files = glob.glob("rewards_Agent_10_*.csv")

# 创建一个 DataFrame 来存储所有文件的平均奖励
average_rewards = pd.DataFrame()

for file in files:
    # 读取每个 CSV 文件
    data = pd.read_csv(file)
    if average_rewards.empty:
        # 如果是第一个文件，初始化 DataFrame
        average_rewards = data
    else:
        # 如果不是第一个文件，将每个 Episode 的奖励值累加
        average_rewards[f"Agent 10"] += data[f"Agent 10"]

# 计算相同 Episode 的奖励值平均值
average_rewards[f"Agent 10"] /= len(files)

# 保存平均奖励值为新的 CSV 文件
output_filename = "average_rewards_Agent_10.csv"
average_rewards.to_csv(output_filename, index=False)
print(f"Average rewards saved to {output_filename}")

# 绘制平均奖励曲线图
plt.plot(average_rewards["Episode"], average_rewards[f"Agent 10"], label="Agent 10 Average Reward")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Average Reward Curve of Agent 10 Across Different Seeds")
plt.legend()
plt.show()
