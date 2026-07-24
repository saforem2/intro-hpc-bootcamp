# Statistical Learning
Sam Foreman, Huihuo Zheng
2025-07-25

<link rel="preconnect" href="https://fonts.googleapis.com">

- [Learning to Cluster Data](#learning-to-cluster-data)
  - [Toy Dataset (2D Blobs)](#toy-dataset-2d-blobs)
- [K-means Clustering](#k-means-clustering)
  - [Step 1: Pick random starting centroids from
    data](#step-1-pick-random-starting-centroids-from-data)
  - [Step 2a: Calculate distance to each
    centroid](#step-2a-calculate-distance-to-each-centroid)
  - [Step 2b: Find nearest cluster for each
    point](#step-2b-find-nearest-cluster-for-each-point)
  - [Step 3: Calculate new centroids](#step-3-calculate-new-centroids)
  - [Step 4: Repeat until convergence](#step-4-repeat-until-convergence)
- [Run Example](#run-example)
- [K-Means on a Breast Cancer
  Dataset](#k-means-on-a-breast-cancer-dataset)
- [Self-organizing maps](#self-organizing-maps)
  - [Generating “Color” data and
    normalizing](#generating-color-data-and-normalizing)
  - [Defining SOM](#defining-som)
  - [Initial state of SOM color
    network](#initial-state-of-som-color-network)
  - [Training SOM](#training-som)
  - [Visualization of trained colormap
    SOM](#visualization-of-trained-colormap-som)
  - [SOM on Cancer data](#som-on-cancer-data)
  - [Training SOM on Cancer data](#training-som-on-cancer-data)
  - [Visualization of trained SOM](#visualization-of-trained-som)
- [Keep learning](#keep-learning)

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/intro-hpc-bootcamp/blob/main/docs/00-intro-AI-HPC/7-statistical-learning/index.ipynb)
[![](https://img.shields.io/badge/-View%20on%20GitHub-333333?style=flat&logo=github&labelColor=gray.png)](https://github.com/saforem2/intro-hpc-bootcamp/blob/main/docs/00-intro-AI-HPC/7-statistical-learning/README.md)

``` python
# This page uses the `bootcamp` helper package. On Colab (or any fresh
# environment) install it + its dependencies; locally this is a no-op.
try:
    import bootcamp  # noqa: F401
except ImportError:
    %pip install -q "git+https://github.com/saforem2/intro-hpc-bootcamp"
```

``` python
%load_ext autoreload
%autoreload 2
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina', 'svg', 'png')
import os
os.environ["TRUECOLOR"] = "1"
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 400
```

## Learning to Cluster Data

If we have data with distinct groupings, the objective is to devise a
method for labeling our data by its group in an automated way.

We will demonstrate this, first, on a toy dataset that we *design* to
have a lower inherent dimensionality, then we move to a higher
dimensional dataset.

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/intro-hpc-bootcamp/blob/main/docs/00-intro-AI-HPC/7-statistical-learning/index.ipynb)

### Toy Dataset (2D Blobs)

Define 3 blobs on a 2D plane:

``` python
import time
import os
import sklearn
from sklearn import datasets
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ambivalent
plt.style.use(ambivalent.STYLES["ambivalent"])

from matplotlib.colors import ListedColormap
from bootcamp.plots import scatter, COLORS

# set random seed for reproducibility
SEED = 42
DFIGSIZE = plt.rcParamsDefault['figure.figsize']
```

``` python
n_samples = 1000 # 300 2D data points
n_features = 2  # 2D data
n_clusters = 4  # 3 unique blobs

# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
# -- Returns --------------------------------------------------------
# x (n_samples, n_features): The generated samples.
# y (n_samples,): Int labels for cluster membership of each sample.
# -------------------------------------------------------------------
cmap = ListedColormap(list(COLORS.values()))
x, y = datasets.make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    random_state=42
)
scatter_kwargs = {
    'xlabel': 'x0',
    'ylabel': 'x1',
    'cmap': cmap,
    'plot_kwargs': {
        'alpha': 0.8,
        'edgecolor': '#222',
    }
}
_ = scatter(x, y, **scatter_kwargs)
```

![](index_files/figure-commonmark/cell-5-output-1.svg)

``` python
# Normalize features
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
x_sc = StandardScaler().fit_transform(x)
```

``` python
fig, (ax1, ax2) = plt.subplots(
    figsize=(1.5 * DFIGSIZE[0], 0.8 * DFIGSIZE[1]),
    ncols=2,
    subplot_kw={
        'aspect': 'equal',
    },
    
)
fig, ax1 = scatter(
    x,
    y,
    fig=fig,
    ax=ax1,
    title='Original',
    **scatter_kwargs
)
fig.subplots_adjust(wspace=0.2)
fig, ax2 = scatter(
    x_sc,
    y,
    fig=fig,
    ax=ax2,
    title='Normalized',
    **scatter_kwargs
)
```

![](index_files/figure-commonmark/cell-7-output-1.svg)

## K-means Clustering

K-means clustering aims to partition n observations into k clusters in
which each observation belongs to the cluster with the nearest mean,
serving as a prototype of the cluster. This results in a partitioning of
the data space into Voronoi cells.

1.  Pick random starting centroids from data
    - $k$ initial “means” (in this case, $k=3$) are randomly generated
      within the data domain (shown in color)
2.  Calculate distance to each centroid
    - $k$ clusters are created by associating every observation with the
      nearest mean. The partitions here represent the Voroni diagram
      generated by the means
    - Find nearest cluster for each point
3.  Calculate new centroids
    - The **centroid** of each of the $k$ clusters becomes the new mean
4.  Steps 2 and 3 are repeated until convergence has been reached

<div id="fig-kmeans">

<div class="flex-container">

![](./assets/k-means0.svg)

![](./assets/k-means1.svg)

![](./assets/k-means2.svg)

![](./assets/k-means3.svg)

</div>

Figure 1: Illustration of the K-means algorithm steps

</div>

### Step 1: Pick random starting centroids from data

- $k$ initial “means” (in this case, $k=3$) are randomly generated
  within the data domain (shown in color)

<div id="fig-kmeans-step1">

![](./assets/k-means0.svg)

Figure 2

</div>

``` python
def initialize_centroids(
        x: np.ndarray,
        n_clusters: int,
        seed: int = 123
) -> np.ndarray:
    """Initialize centroids.
    
    Inputs:
      - x (np.ndarray): Data of shape (num_points, num_features)
      - n_clusters (int): Number of clusters to use
      - seed (int): Random seed.
        
    Outputs:
      - centroids (np.ndarray): Randomly chosen from the data
        with shape: (num_clusters, num_features).
    """
    np.random.RandomState(seed)
    # 1. Randomly permute data points
    # 2. From this, pick the first `n_clusters` indices
    # 3. Return these as our initial centroids
    random_idx = np.random.permutation(x.shape[0])
    centroids = x[random_idx[:n_clusters]]
    return centroids
```

### Step 2a: Calculate distance to each centroid

- Calculate distance to each centroid
  - $k$ clusters are created by associating every observation with the
    nearest mean. The partitions here represent the Voroni diagram
    generated by the means.
  - Find nearest cluster for each point

<div id="fig-kmeans-step2a">

![](./assets/k-means1.svg)

Figure 3

</div>

``` python
def compute_distance(
        x: np.ndarray,
        centroids: np.ndarray,
        n_clusters: int
) -> np.ndarray:
    """Compute distance.

    Inputs:
      - x (np.ndarray): Input data of shape
        (num_points, num_features)
      - centroids (np.ndarray): Cluster centroids  with shape 
        (num_clusters, num_features)
      - n_clusters (int): Number of clusters being used.

    Outputs:
      - distance (np.ndarray): Distance of each point 
        to each centroid with shape 
        (num_points, num_clusters)
    """
    # distance vector
    distance = np.zeros((x.shape[0], n_clusters))
    # loop over each centroid
    for k in range(n_clusters):
        # calculate distance for each point from centroid
        kcentroid_distance = x - centroids[k, :]
        # apply normalization for stability
        row_norm = np.linalg.norm(kcentroid_distance, axis=1)
        # return distance squared
        distance[:, k] = np.square(row_norm) 

    return distance
```

### Step 2b: Find nearest cluster for each point

``` python
def find_closest_centroid(
        distance: np.ndarray
) -> np.ndarray:
    """Find closest centroid.

    Inputs:
      - distance (np.ndarray): Distance of each point to each centroid with shape
        (num_points, num_clusters)

    Outputs:
      - nearest_centroid_indices (np.ndarray): Index of nearest centroid with shape
        (num_points,)
    """
    nearest_centroid_indices = np.argmin(distance, axis=1)
    return nearest_centroid_indices
```

### Step 3: Calculate new centroids

3.  Calculate new centroids
    - The **centroid** of each of the $k$ clusters becomes the new mean

<div id="fig-kmeans-step3">

![](./assets/k-means2.svg)

Figure 4

</div>

``` python
def compute_centroids(
        x: np.ndarray,
        nearest_centroid_indices: np.ndarray,
        n_clusters: int
) -> np.ndarray:
    """Compute centroids.
    
    Inputs:
      - x (np.ndarray): Input data of shape
        (num_points, num_features)
      - nearest_centroid_indices: Index of nearest centroid of shape
        (num_points,)
      - n_clusters (int): Number of clusters being used
      
    Outputs:
      - centroids (np.ndarray): Cluster centroids with shape
        (num_clusters, num_features)
    """
    # new centroids vector
    centroids = np.zeros((n_clusters, x.shape[1]))
    # loop over each centroids
    for k in range(n_clusters):
        # calculate the mean of all points assigned to this centroid
        centroids[k, :] = np.mean(
            x[nearest_centroid_indices == k, :],
            axis=0
        )
    return centroids
```

### Step 4: Repeat until convergence

4.  Repeat steps 2 and 3 until convergence has been reached

<div id="fig-kmeans-step4">

![](./assets/k-means3.svg)

Figure 5

</div>

``` python
from __future__ import absolute_import, annotations, division, print_function

from typing import Optional
import seaborn as sns
import IPython.display as ipydis

from bootcamp.plots import plot_kmeans_points

def apply_kmeans(
    x: np.ndarray,
    n_clusters: int,
    iterations: int = 100,
    seed: int = 123,
    cmap: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (centroids, cluster_id)."""
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ initialize centroids:               ┃
    # ┃   - shape: (n_clusters, x.shape[1]) ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    centroids = initialize_centroids(
        x,  # (n_points, n_features)
        n_clusters=n_clusters,
        seed=seed,
    )
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ -- Iteratively improve centroid location ----------------- ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 1. Compute the distance from entries in x to each centroid ┃
    # ┃    - distance.shape: (n_points, n_clusters)                ┃
    # ┃ 2. Return the closest cluster (0, 1, ..., n_clusters-1)    ┃
    # ┃    - cluster_id.shape: (n_points)                          ┃
    # ┃ 3. Calculate the mean position of each labeled cluster     ┃
    # ┃    - centroids.shape: (n_clusters, n_features)             ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    for i in range(iterations):
        # save old centroids
        old_centroids = centroids
        distance = compute_distance(x, old_centroids, n_clusters)
        cluster_id = find_closest_centroid(distance)
        centroids = compute_centroids(x, cluster_id, n_clusters)

        # plotting for visual comprehension
        ipydis.clear_output("wait")
        print(f"Iteration: {i}")
        plot_kmeans_points(x, centroids, cluster_id, cmap=cmap)
        time.sleep(0.5)

        # if our points are the same as the old centroids, then we can stop
        if np.all(old_centroids == centroids):
            print(f"No change in centroids! Exiting!")
            break

    return centroids, cluster_id
```

## Run Example

``` python
from bootcamp.plots import COLORS

cmap = ListedColormap(list(COLORS.values()))

n_samples = 500  # 300 2D data points
n_features = 2  # 2D data
n_clusters_true = 5  # unique blobs
n_clusters_guess = 5
SEED = 456
iterations = 50

# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
x, y = datasets.make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters_true,
    random_state=SEED,
)

x_sc = StandardScaler().fit_transform(x)

centroids, cluster_id = apply_kmeans(
    x_sc,
    n_clusters_guess,
    iterations,
    SEED,
    cmap=cmap,
)
```

    Iteration: 12

![](index_files/figure-commonmark/cell-13-output-2.svg)

    No change in centroids! Exiting!

## K-Means on a Breast Cancer Dataset

Now we use more realistic data that cannot be easily plotted on a 2D
grid. This dataset has 30 features (columns) for 569 patients (rows). In
addition, there is a *target* feature that indicates if the cancer was
*malignant* (0) or *benign* (1). In the ideal case, our 30 features
would provide easy deliniation between these two classes.

Let’s extract our data into $x$ and our truth labels into $y$

``` python
import pandas as pd


def load_cancer_data() -> dict:
    """Return cancer dataset (unscaled)."""
    from sklearn import datasets

    data = datasets.load_breast_cancer()
    return data


def sort_cancer_data(data: dict) -> tuple[pd.DataFrame, pd.Series]:
    # Get features and target
    x = pd.DataFrame(
        data["data"],
        columns=data["feature_names"],
    )
    x = x[sorted(x.columns)]
    y = data["target"]
    return x, y
```

``` python
data = load_cancer_data()
x, y = sort_cancer_data(data)
```

``` python
print(data.keys())
print("data size:", len(data["data"]))
print("number of features:", len(data["feature_names"]))
print(data["feature_names"])
```

    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    data size: 569
    number of features: 30
    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness' 'mean compactness' 'mean concavity'
     'mean concave points' 'mean symmetry' 'mean fractal dimension'
     'radius error' 'texture error' 'perimeter error' 'area error'
     'smoothness error' 'compactness error' 'concavity error'
     'concave points error' 'symmetry error' 'fractal dimension error'
     'worst radius' 'worst texture' 'worst perimeter' 'worst area'
     'worst smoothness' 'worst compactness' 'worst concavity'
     'worst concave points' 'worst symmetry' 'worst fractal dimension']

``` python
# wrap data in pandas DataFrame
# y = data['target'] # value: 0 = 'Malignant' 1 = 'Benign'
x = data["data"]
y = data["target"]
print(f"(malignant, benign): {np.bincount(y)}")
print(f"x.shape: {x.shape}")
print(f"y.shape: {y.shape}")
```

    (malignant, benign): [212 357]
    x.shape: (569, 30)
    y.shape: (569,)

``` python
# normalize
x_sc = StandardScaler().fit_transform(x)
```

Plot histogram of number of true class labels (malignant/benign) and
number of k-means cluster labels.

This gives an indication of how well we did in clustering our data, but
is not a “correctness” or “accuracy” metric.

> [!NOTE]
>
> Keep in mind, k-means is not a classifier.
>
> Cluster label 0 given from k-means, does not correspond to cluster
> label 0 in the truth.

``` python
from bootcamp.plots import plot_hists

# for seed in range(10):
CSEED = 42
kmeans = KMeans(n_clusters=2, random_state=CSEED)
# fit the data
kfit = kmeans.fit(x_sc)
cluster_ids = kmeans.labels_
print(
    f"seed: {CSEED}\nNumber of samples in each cluster: {np.bincount(kmeans.labels_)}"
)
k_means_bins = np.bincount(cluster_ids)
y_bins = np.bincount(y)
plt.rcParams["figure.figsize"] = [1.5 * DFIGSIZE[0], 0.8 * DFIGSIZE[1]]
plot_hists(k_means_bins, y_bins, xlabels=["Malignant", "Benign"])
```

    seed: 42
    Number of samples in each cluster: [188 381]

![](index_files/figure-commonmark/cell-19-output-2.svg)

This plot shows the normalized inertia as we vary the number
`n_clusters` used in our k-means fit to the breast cancer data. This
value essentially indicates the mean distance of a point to the cluster
centroids. Obviously more clusters result in data points being nearer to
a centroid.

If our dataset easily split into 2 clusters, we would see a step
function behavior going from 1 to 2, then only very minor improvements
above 2. What we see here tells us our data does not easily cluster.

``` python
from bootcamp.plots import plot_kmeans_obj

plt.rcParams["figure.figsize"] = [DFIGSIZE[0], 0.6 * DFIGSIZE[1]]
_ = plot_kmeans_obj(x_sc, nclusters=10, plot_points=False)
```

![](index_files/figure-commonmark/cell-20-output-1.svg)

For example, if we return to our blob data with 2 clusters, it become
clear.

``` python
n_samples = 300  # 300 2D data points
n_features = 2  # 2D data
n_clusters = 2  # unique blobs
seed = 456
iterations = 25
x, y = datasets.make_blobs(
    n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=seed
)
plot_kmeans_obj(x, nclusters=10, plot_points=False)
```

    array([71.2677332 ,  2.06848282,  1.66344532,  1.29087519,  1.09143669,
            1.0136986 ,  0.80095706,  0.68552636,  0.65083526])

![](index_files/figure-commonmark/cell-21-output-2.svg)

## [Self-organizing maps](https://en.wikipedia.org/wiki/Self-organizing_map)

Self Organizing Maps (SOM) were proposed and became widespread in the
1980s, by a Finnish professor named Teuvo Kohonen and are also called
‘Kohonen maps’.

``` python
# finding best matching unit
def find_bmu(t, net, n):
    """
    Find the best matching unit for a given vector, t, in the SOM
    Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
             and bmu_idx is the index of this vector in the SOM
    """
    bmu_idx = np.array([0, 0])
    # set the initial minimum distance to a huge number
    # min_dist = #np.iinfo(np.int).max
    min_dist = np.inf
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(n, 1)
            # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
            sq_dist = np.sum((w - t) ** 2)
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = np.array([x, y])
    # get vector corresponding to bmu_idx
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(n, 1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)


# Decaying radius of influence
def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)


# Decaying learning rate
def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)


# Influence in 2D space
def calculate_influence(distance, radius):
    return np.exp(-distance / (2.0 * (radius**2)))


# Update weights
def update_weights(net, bmu_idx, r, l):
    wlen = net.shape[2]
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(wlen, 1)
            # get the 2-D distance (again, not the actual Euclidean distance)
            w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
            # if the distance is within the current neighbourhood radius
            if w_dist <= r**2:
                # calculate the degree of influence (based on the 2-D distance)
                influence = calculate_influence(w_dist, r)
                # now update the neuron's weight using the formula:
                # new w = old w + (learning rate * influence * delta)
                # where delta = input vector (t) - old w
                new_w = w + (l * influence * (t - w))
                # commit the new weight
                net[x, y, :] = new_w.reshape(1, wlen)
    return net
```

The idea behind a SOM is that you’re mapping high-dimensional vectors
onto a smaller dimensional (typically 2D) space. Vectors that are close
in the high-dimensional space also end up being mapped to nodes that are
close in 2D space thus preserving the “topology” of the original data.

### Generating “Color” data and normalizing

``` python
raw_data = np.random.randint(0, 255, (10000, 3))  # JPEG like data
# raw_data = np.random.uniform(size=(500, 3))
data = StandardScaler().fit_transform(raw_data)  # Normalized
```

### Defining SOM

Defining network size, number of iterations and learning rate

``` python
network_dimensions = np.array([50, 50])
n_iterations = 2500
init_learning_rate = 0.075
```

Establish size variables based on data

``` python
m = data.shape[0]
n = data.shape[1]
```

Weight matrix (i.e. the SOM) needs to be one n-dimensional vector for
each neuron in the SOM

``` python
net = np.random.random(
    (network_dimensions[0], network_dimensions[1], n)
)  # 25 neurons each with a 3D vector
```

Initial neighbourhood radius and decay parameter

``` python
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2.0
time_constant = n_iterations / np.log(init_radius)
```

### Initial state of SOM color network

``` python
plt.rcParams["image.cmap"] = "rainbow"
fig, ax = plt.subplots(figsize=(4, 4))
image = ax.matshow(net)
# _ = fig.colorbar(image)
_ = plt.show()
```

![](index_files/figure-commonmark/cell-28-output-1.svg)

### Training SOM

``` python
def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))


plt.rcParams["figure.figsize"] = [
    i / 2.0 for i in plt.rcParamsDefault["figure.figsize"]
]
init_learning_rate = 0.1
# net = np.random.uniform(size=net.shape)
# net = (255 * ((net + 1.) / 2.))
images = []
for iteration in range(n_iterations):
    # select a training example at random - shape of 1x3
    t = data[np.random.randint(0, m), :].reshape(np.array([n, 1]))
    # find its Best Matching Unit
    bmu, bmu_idx = find_bmu(t, net, n)  # Gives the row, column of the best neuron
    # decay the SOM parameters
    r = decay_radius(init_radius, iteration, time_constant)
    l = decay_learning_rate(init_learning_rate, iteration, n_iterations)
    # Update SOM weights
    net = update_weights(net, bmu_idx, r, l)
    if iteration % 50 == 0:
        print(f"Iteration: {iteration}")
        ipydis.clear_output("wait")
        fig, ax = plt.subplots(figsize=(4, 4))
        net_img = normalize(net)
        _ = fig.suptitle(f"Iteration: {iteration}", y=1.0)
        plt.tight_layout()
        image = ax.matshow(net_img, cmap="rainbow")
        if iteration % 100 == 0:
            images.append(net_img)
        _ = plt.show()
        time.sleep(0.5)
```

![](index_files/figure-commonmark/cell-29-output-1.svg)

### Visualization of trained colormap SOM

``` python
len(images)
```

    25

``` python
fig, axes = plt.subplots(
    figsize=(2.0 * DFIGSIZE[0], 1.5 * DFIGSIZE[1]),
    nrows=5,
    ncols=5,
    tight_layout=True,
)
fig.subplots_adjust(wspace=0.1)
axes = axes.flatten()
for idx, (img, ax) in enumerate(zip(images, axes)):
    _ = ax.matshow(img, cmap="rainbow")
    _ = ax.set_title(f"Iteration: {100 * idx}")
    _ = ax.set_xticks([])
    _ = ax.set_xticklabels([])
    _ = ax.set_yticks([])
    _ = ax.set_yticklabels([])

_ = plt.show()
```

![](index_files/figure-commonmark/cell-31-output-1.svg)

``` python
from matplotlib import cm

fig, ax = plt.subplots(figsize=(4, 4))
image = ax.imshow(normalize(net), cmap="rainbow")
_ = ax.set_title("Final SOM")
_ = plt.show()
```

![](index_files/figure-commonmark/cell-32-output-1.svg)

### SOM on Cancer data

``` python
# Load data
data = datasets.load_breast_cancer()
# Get features and target
X = data["data"]  # pd.DataFrame(data['data'], columns=data['feature_names'])
Y = data["target"]
print(X.shape, Y.shape)
```

    (569, 30) (569,)

``` python
network_dimensions = np.array([10, 10])
n_iterations = 2000
init_learning_rate = 0.01
# establish size variables based on data
n_points = X.shape[0]  # number of points
n_features = X.shape[1]  # 30 features per point

# weight matrix (i.e. the SOM) needs to be one n-dimensional vector for each neuron in the SOM
net = np.random.random((network_dimensions[0], network_dimensions[1], n_features))

# initial neighbourhood radius
init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
# radius decay parameter
time_constant = n_iterations / np.log(init_radius)
```

``` python
# convert the network to something we can visualize
net_vis = np.zeros(
    shape=(net.shape[0], net.shape[1], 3),
    dtype=np.float32,
)  # Array for SOM color map visualization

for sample in range(n_points):
    t = X[sample, :].reshape(np.array([n_features, 1]))
    # find its Best Matching Unit for this data point
    bmu, bmu_idx = find_bmu(t, net, n_features)
    # set that unit to the label of this data point
    net_vis[bmu_idx[0], bmu_idx[1], 0] = Y[sample]  # Red if benign


fig, ax = plt.subplots()
im = ax.matshow(normalize(net_vis))
# _ = plt.colorbar(im1, ax=ax1)
plt.show()
```

![](index_files/figure-commonmark/cell-35-output-1.svg)

### Training SOM on Cancer data

``` python
for iteration in range(n_iterations):
    # select a training example at random - shape of 1x3
    t = X[np.random.randint(0, n_points), :].reshape(np.array([n_features, 1]))
    # find its Best Matching Unit
    bmu, bmu_idx = find_bmu(t, net, n_features)
    # decay the SOM parameters
    r = decay_radius(init_radius, iteration, time_constant)
    l = decay_learning_rate(init_learning_rate, iteration, n_iterations)
    # Update SOM weights
    net = update_weights(net, bmu_idx, r, l)

    if iteration % 50 == 0:
        ipydis.clear_output("wait")
        net_vis = np.zeros(
            shape=(np.shape(net)[0], np.shape(net)[1], 3), dtype="double"
        )  # Array for SOM color map visualization
        for sample in range(n_points):
            t = X[sample, :].reshape(np.array([n_features, 1]))
            # find its Best Matching Unit
            bmu, bmu_idx = find_bmu(t, net, n_features)
            net_vis[bmu_idx[0], bmu_idx[1], 0] = Y[sample]  # Red if benign

        fig, ax = plt.subplots(figsize=(3, 3))
        _ = ax.set_title(f"Iteration: {iteration}")
        _ = ax.imshow(normalize(net_vis), cmap="rainbow")
        _ = plt.show()
        time.sleep(0.25)
```

![](index_files/figure-commonmark/cell-36-output-1.svg)

### Visualization of trained SOM

``` python
net_vis = np.zeros(
    shape=(net.shape[0], net.shape[1], 3), dtype="double"
)  # Array for SOM color map visualization

for sample in range(n_points):
    t = X[sample, :].reshape(np.array([n_features, 1]))
    # find its Best Matching Unit
    bmu, bmu_idx = find_bmu(t, net, n_features)
    net_vis[bmu_idx[0], bmu_idx[1], 0] = Y[sample]  # Red if benign

_ = plt.imshow(normalize(net_vis))
_ = plt.show()
```

![](index_files/figure-commonmark/cell-37-output-1.svg)

## Keep learning

1.  \<lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/course/\>
2.  \<www.coursera.org/learn/ml-clustering-and-retrieval/\>
3.  \<www.coursera.org/learn/machine-learning/home/week/8\>
