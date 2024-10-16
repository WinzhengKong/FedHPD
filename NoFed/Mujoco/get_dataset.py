import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 读取CSV文件
df = pd.read_csv("all_state_transitions_20240815_211652.csv")

# 将State列中的数据转化为numpy数组
states = np.array(df["State"].apply(eval).tolist())

# 使用PCA将状态降维到2维（便于可视化）
pca = PCA(n_components=2)
states_pca = pca.fit_transform(states)

# 使用KMeans聚类
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(states_pca)

# 可视化PCA降维后的数据和聚类结果
plt.figure(figsize=(10, 8))
for cluster_id in range(n_clusters):
    cluster_points = states_pca[df['Cluster'] == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')

plt.title("PCA Reduced States with KMeans Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# 为每个类别选取1000个状态
sampled_states = pd.DataFrame()

for cluster_id in range(n_clusters):
    cluster_df = df[df['Cluster'] == cluster_id]
    if len(cluster_df) >= 1000:
        sampled_df = cluster_df.sample(n=1000, random_state=42)
    else:
        sampled_df = cluster_df
    sampled_states = pd.concat([sampled_states, sampled_df])

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
'''
# 保存到新的CSV文件中
sampled_states.to_csv(f"sampled_state_transitions_{timestamp}.csv", index=False)
print("Selected states saved to 'sampled_state_transitions.csv'")
'''

# 只保留State一栏并保存到新的CSV文件中
sampled_states = sampled_states[['State']]
sampled_states.to_csv(f"sampled_state_transitions_{timestamp}.csv", index=False)
print("Selected states saved to 'sampled_state_transitions.csv'")
