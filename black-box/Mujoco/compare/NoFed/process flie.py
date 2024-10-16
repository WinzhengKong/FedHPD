import pandas as pd

# 创建一个空的 DataFrame，用于存放整合后的数据
merged_data = pd.DataFrame()

# 循环读取每个文件并合并到 merged_data 中
for i in range(1, 11):
    # 读取 CSV 文件
    file_name = f'average_rewards_Agent_{i}.csv'
    df = pd.read_csv(file_name)

    # 删除 'Episode' 列
    df = df.drop('Episode', axis=1)

    # 如果是第一个文件，初始化 merged_data
    if merged_data.empty:
        merged_data = df
    else:
        # 合并 'Agent i' 列
        merged_data = pd.concat([merged_data, df], axis=1)

# 保存整合后的数据到一个新的 CSV 文件
merged_data.to_csv('rewards_WoFed_0907.csv', index=False)

print("整合完成")
