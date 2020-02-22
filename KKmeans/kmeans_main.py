import numpy as np
import matplotlib.pyplot as plt
from kernels.KKmeans.KKmeans import KernelKMeans
from kernels.dataset_generators import Generator
from sklearn.cluster import SpectralClustering
import sklearn.cluster as skl_cluster
import sklearn.datasets as skl_data
import numpy as np

# X = np.array([[1, 1], [2, 1], [1, 0],[4, 7], [3, 5], [3, 6]])
# clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit(X)
# clustering.labels_array([1, 1, 1, 0, 0, 0])
#
# SpectralClustering(affinity='rbf', assign_labels='discretize', coef0=1,degree=3, eigen_solver=None, eigen_tol=0.0, gamma=1.0,kernel_params=None, n_clusters=2, n_init=10, n_jobs=None, n_neighbors=10, random_state=0)


nsamples = 100
# X, y = Generator.generate(dataset_name="circles", n_samples=nsamples, random_state=123)
X, y = skl_data.make_circles(n_samples=nsamples, noise=.01, random_state=0)

clusters = SpectralClustering(n_clusters=2,affinity='nearest_neighbors',  assign_labels='kmeans').fit_predict(X)
# clusters = clustering.labels_

reds = clusters == 0
blues = clusters == 1

plt.figure()
plt.title("Original space")
plt.scatter(X[reds, 0], X[reds, 1], color="red", alpha=0.5)
plt.scatter(X[blues, 0], X[blues, 1], color="blue", alpha=0.5)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()



import sklearn.datasets as skl_data

circles, circles_clusters = skl_data.make_circles(n_samples=400, noise=.01, random_state=0)