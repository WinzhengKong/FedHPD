import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 定义读取和解析数据的函数
def read_csv_data(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
    parsed_data = []
    for index, row in df.iterrows():
        for item in row:
            if pd.notna(item):
                values = list(map(float, item.replace('"', '').split(',')))
                parsed_data.append(values)
    return parsed_data

# 文件名列表，可以根据需要修改
file_paths = [
    'KD_state_5-from-each.csv',
    'KD_state_5-from-each_20240919_102207.csv',
    'KD_state_1-from-1_20240919_100815.csv',
    # 添加更多文件名
]

# 定义新的颜色
colors = ['#ff7f0e', '#2ca02c', '#d62728']  # Replace with desired colors
markers = ['o', 's', '^']

# 设置 Seaborn 主题
sns.set_theme(style="whitegrid")

# 创建一个空的 DataFrame 来存储所有数据
all_reduced_df = pd.DataFrame()

# 遍历文件列表，读取数据并存储到一个 DataFrame 中
for i, file_path in enumerate(file_paths):
    data = read_csv_data(file_path)
    df = pd.DataFrame(data, columns=['Dim1', 'Dim2', 'Dim3', 'Dim4'])

    # 使用 PCA 进行降维
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)

    # 转换为 DataFrame 以便绘图
    reduced_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
    reduced_df['Dataset'] = f'Dataset {i + 1}'  # 添加数据集标签

    # 将当前的数据合并到总的 DataFrame
    all_reduced_df = pd.concat([all_reduced_df, reduced_df], ignore_index=True)

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=all_reduced_df, x='PCA1', y='PCA2', hue='Dataset', style='Dataset',
                palette=colors[:len(file_paths)], markers=markers[:len(file_paths)],
                s=25, alpha=0.4)

# 自定义 x 轴和 y 轴标签以及数字的大小
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 调整图例字体大小
plt.legend(fontsize=12)

plt.grid(False)  # Disable grid

# Save the figure with 600 dpi
plt.savefig('pca_plot.png', dpi=600, bbox_inches='tight')
plt.show()
