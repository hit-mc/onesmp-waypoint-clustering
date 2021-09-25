import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# cluster algorithms
from sklearn.cluster import KMeans

# load data
# columns: name, x, z
waypoints = pd.read_csv('waypoints.csv')
data = waypoints[['x', 'z']]
colors = ['red', 'darkorange', 'olive', 'chartreuse', 'green', 'mediumspringgreen',
          'mediumturquoise', 'aqua', 'dodgerblue', 'blue', 'darkviolet', 'magenta',
          'deeppink', 'slategray']


def cluster_by_k_means(data_, n_clusters: int):
    # Calculate seeds from kmeans++
    data_ = np.array(data_)
    km = KMeans(n_clusters=n_clusters).fit(data_)
    return km.labels_, km.cluster_centers_


def plot(data_: pd.DataFrame, labels_: list[int], centers_: pd.DataFrame, colors_: list[str], caption: str):
    # enforce datatype
    centers_ = pd.DataFrame(centers_)
    # invert Y axis
    ax = plt.gca()
    if not ax.yaxis_inverted:
        ax.invert_yaxis()
    # draw
    plt.title(caption)
    plt.scatter(data_.iloc[:, 0], data_.iloc[:, 1], c=[colors_[x] for x in labels_], alpha=0.5)
    plt.scatter(centers_.iloc[:, 0],
                centers_.iloc[:, 1],
                c=colors_[:len(centers_)],
                marker='x',
                alpha=0.7)


subplots = [331, 332, 333, 334, 335, 336, 337, 338, 339]
plt.figure(figsize=(20.48, 15.36), dpi=100)
for clusters, plt_id in zip(range(1, 10), subplots):
    labels, centers = cluster_by_k_means(data, clusters)
    plt.subplot(plt_id)
    plot(data, labels, centers, colors, f'k-means, n_clusters={clusters}')
    print('labels:', labels)
    print('centers', centers)
plt.show()
