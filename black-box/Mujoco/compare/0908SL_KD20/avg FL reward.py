import pandas as pd
import glob

# 读取所有以 'rewards_Fed_SL_seed' 开头的 CSV 文件
file_paths = glob.glob('rewards_Fed_SL_seed*.csv')

# 初始化一个空的数据帧用于存储累加值
cumulative_df = None
file_count = 0

# 遍历所有文件
for file_path in file_paths:
    # 读取当前CSV文件并只保留Agent列
    df = pd.read_csv(file_path, usecols=['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5', 'Agent 6', 'Agent 7', 'Agent 8', 'Agent 9', 'Agent 10'])

    # 如果是第一个文件,初始化cumulative_df
    if cumulative_df is None:
        cumulative_df = df.copy()
    else:
        # 累加每个文件的值
        cumulative_df += df

    file_count += 1

# 计算平均值
average_df = cumulative_df.copy()
average_df = average_df.divide(file_count)

# 保存结果到新的CSV文件
average_df.to_csv('rewards_Fed_KD20_0909.csv', index=False)

print("计算完成,结果已保存至 'rewards_Fed_KD20_0909.csv'")
