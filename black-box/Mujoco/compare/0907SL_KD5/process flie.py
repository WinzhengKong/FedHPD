import pandas as pd

# 创建一个空的 DataFrame，用于存放整合后的数据
merged_data = pd.DataFrame()

# 循环读取每个文件并合并到 merged_data 中
for i in range(1, 11):
    # 读取 CSV 文件
    file_name = f'average_rewards_Agent_{i}.csv'
    df = pd.read_csv(file_name)

    # 如果是第一个文件，初始化 merged_data
    if merged_data.empty:
        merged_data = df
    else:
        # 只合并 'Agent i' 列，保持 'Episode' 列不变
        merged_data = pd.merge(merged_data, df, on='Episode')

# 保存整合后的数据到一个新的 CSV 文件
merged_data.to_csv('rewards_WoFed_0831.csv', index=False)

print("整合完成")
