import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from kernels.KKmeans.KKmeans import KernelKMeans
from kernels.dataset_generators import Generator
import sklearn.cluster as skl_cluster

np.random.seed(0)

nsamples = 100
X, y = Generator.generate(dataset_name="manual_circles", n_samples=nsamples)
reds = y == 0
blues = y == 1

# My KKmeans
kkm_model_rbf = KernelKMeans(n_clusters=2, max_iter=200, kernel="linear")
kkm_clusters = kkm_model_rbf.fit(X)

# scikit_lear kmeans
Kmean = skl_cluster.KMeans(n_clusters=2)
Kmean.fit(X)
clusters = Kmean.predict(X)

plt.figure()
plt.subplot(3, 1, 1)
plt.title("Original Datasset")
plt.scatter(X[:, 0], X[:, 1], s=15, linewidth=0, c=y, cmap='flag')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplot(3, 1, 2)
plt.title("Kkmeans(linear, clusters =2)")
plt.scatter(X[:, 0], X[:, 1], s=15, linewidth=0, c=kkm_clusters, cmap='flag')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplot(3, 1, 3)
plt.title("sklearn Kmeans(clusters =2)")
plt.scatter(X[:, 0], X[:, 1], s=15, linewidth=0, c=clusters, cmap='flag')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


plt.tight_layout()
plt.show()
