import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise
from kernels.KernelsHelper import KernelsHelper


class KernelKMeans:
    def __init__(self, n_clusters=3, kernel="linear", max_iter=50, random_state=None, gamma=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.gamma = gamma
        self.kernel = kernel


    def _initialize_cluster(self, X):
        self.N = np.shape(X)[0]
        self.y = np.random.randint(low=0, high=self.n_clusters, size=self.N)
        # self.K = KernelsHelper.gram_matrix(X, kernel=self.kernel, centered=True, bandwidth=self.gamma)
        # self.K = kernel(X)
        self.K = KernelsHelper.gram_matrix(X, kernel="rbf", centered=False, bandwidth=0.1)
        # print("k1 = ", self.K)
        # print("kk1 = ", KernelsHelper.gram_matrix(X, kernel="rbf", centered=False, bandwidth=0.1))

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        K = KernelsHelper.gram_matrix(X, kernel=self.kernel, centered=False, bandwidth=0.1)

        sw = np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist)
            self.labels_ = dist.argmin(axis=1)

        return dist.argmin(axis=1)

    def fit_predict(self, X):
        self._initialize_cluster(X)
        for _ in range(self.max_iter):
            obj = np.tile(np.diag(self.K).reshape((-1, 1)), self.n_clusters)
            N_c = np.bincount(self.y)
            for c in range(self.n_clusters):
                obj[:, c] -= 2 * \
                             np.sum((self.K)[:, self.y == c], axis=1) / N_c[c]
                obj[:, c] += np.sum((self.K)[self.y == c][:, self.y == c]) / \
                             (N_c[c] ** 2)
            self.y = np.argmin(obj, axis=1)
        return  self.y

    def _compute_dist(self, K, dist):
        sw = self.sample_weight_
        # print("K: ", K)
        for j in range(self.n_clusters):
            mask = self.labels_ == j

            denom = sw[mask].sum()
            denomsq = denom * denom
            KK = K[mask][:, mask]

            second_term = - 2 * np.sum(K[:, mask], axis=1) / denom
            third_term = np.sum(KK) / denomsq
            dist[:, j] += third_term
            dist[:, j] += second_term

# from sklearn.datasets import make_blobs
#
# X, y = make_blobs(n_samples=5, centers=5, random_state=0)
#
# km = KernelKMeans(n_clusters=2, max_iter=3, random_state=0)
#
# print("assigns: ", km.fit(X)[:10])
