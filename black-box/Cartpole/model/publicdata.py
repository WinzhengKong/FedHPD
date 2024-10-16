import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_states(filename):
    df = pd.read_csv(filename)
    states = df['State'].apply(lambda x: list(map(float, x.split(','))))
    return states.tolist(), df

def embed_states(states, n_components=2):
    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)

    pca = PCA(n_components=n_components)
    states_embedded = pca.fit_transform(states_scaled)

    return states_embedded

def cluster_states(states_embedded, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(states_embedded)

    return kmeans.labels_, kmeans.cluster_centers_

def plot_clusters(states_embedded, labels, centers):
    plt.scatter(states_embedded[:, 0], states_embedded[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('State Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

def select_and_save_states(filename, labels, df, selection_scheme):
    selected_indices = []
    unique_labels = np.unique(labels)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if selection_scheme == '1-from-1':
        # Randomly select one cluster
        cluster_label = np.random.choice(unique_labels)
        cluster_indices = np.where(labels == cluster_label)[0]
        selected_indices = np.random.choice(cluster_indices, 5000, replace=False)
    elif selection_scheme == '5-from-each':
        # Select 1000 states from each of the five clusters
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            selected_indices.extend(np.random.choice(cluster_indices, 200, replace=False))

    selected_states = df.loc[selected_indices]
    selected_states.to_csv(f'KD_state_{selection_scheme}_{timestamp}.csv', index=False)

def main():
    filename = 'states_20240919_094500.csv'  # Replace with your actual filename
    states, df = load_states(filename)

    states_embedded = embed_states(states, n_components=2)
    labels, centers = cluster_states(states_embedded, n_clusters=5)

    plot_clusters(states_embedded, labels, centers)

    # Choose the selection scheme here: '1-from-1' or '5-from-each'
    selection_scheme = '5-from-each'
    select_and_save_states(filename, labels, df, selection_scheme)

if __name__ == "__main__":
    main()
