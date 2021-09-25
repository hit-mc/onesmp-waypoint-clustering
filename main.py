import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# cluster algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# configurations
ROWS, COLS = 3, 4
MAX_CLUSTERS = ROWS * COLS
div8 = True

# load data
# columns: name, x, z
waypoints = pd.read_csv('waypoints.csv')
data = waypoints[['x', 'z']]
if div8:
    data /= 8
colors = ['red', 'darkorange', 'olive', 'chartreuse', 'green', 'mediumspringgreen',
          'mediumturquoise', 'aqua', 'dodgerblue', 'blue', 'darkviolet', 'magenta',
          'deeppink', 'slategray']


def cluster_by_k_means(data_, n_clusters: int):
    data_ = np.array(data_)
    km = KMeans(n_clusters=n_clusters).fit(data_)
    return km.labels_, km.cluster_centers_


# noinspection PyUnresolvedReferences
def cluster_by_gmm(data_, n_clusters: int):
    data_ = np.array(data_)
    gmm = GaussianMixture(n_components=n_clusters).fit(data_)
    labels_ = gmm.predict(data_)
    return labels_, gmm.means_


def plot(data_: pd.DataFrame, labels_: list[int], centers_, colors_: list[str], caption: str):
    # enforce datatype
    centers_ = pd.DataFrame(centers_)
    # invert Y axis
    ax = plt.gca()
    if ax.yaxis_inverted:
        ax.invert_yaxis()
    # draw
    plt.title(caption)
    plt.scatter(data_.iloc[:, 0],
                data_.iloc[:, 1],
                c=[colors_[x] for x in labels_],
                alpha=0.5)
    plt.scatter(centers_.iloc[:, 0],
                centers_.iloc[:, 1],
                c=colors_[:len(centers_)],
                marker='x',
                alpha=0.7)


subplots = [(ROWS, COLS, i) for i in range(1, MAX_CLUSTERS + 1)]
cluster_methods = [cluster_by_k_means, cluster_by_gmm]
for method in cluster_methods:
    plt.figure(figsize=(20.48, 15.36), dpi=100)
    plt.suptitle(method.__name__, fontsize=24, fontweight='bold')
    for clusters, plt_id in zip(range(1, MAX_CLUSTERS + 1), subplots):
        print(f'Plotting with method {method.__name__}, n_cluster={clusters}')
        labels, centers = method(data, clusters)
        ax = plt.subplot(*plt_id)
        plot(data, labels, centers, colors, f'n_clusters={clusters}')
        # manually set ticks, more fine-grained
        ax.xaxis.set_label_text('x')
        ax.yaxis.set_label_text('z')
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, width=1, direction='inout')
        ax.tick_params(which='minor', length=4, width=1, direction='in')
        ax.grid(which='both', linewidth=0.5)
    plt.show()
