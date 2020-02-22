from kernels.KernelsHelper import KernelsHelper
from scipy.linalg import eigh
import numpy as np
import numexpr as ne


class KPCA:
    def __init__(self, kernel="linear", bandwidth=0, n_components=None):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_components = n_components
        self.K = None
        print("KPCA Initialized with [", "kernel: ", kernel, ", n_components: ", n_components, "]")

    def fit(self, X):
        self.K = KernelsHelper.gram_matrix(X, kernel=self.kernel, centered=True, bandwidth=self.bandwidth)
        self.eigen_vals, self.eigen_vecs = eigh(self.K)
        self.eigen_vectors_to_return = None
        self.eigen_vectors_to_return = np.column_stack(
            (self.eigen_vecs[:, -i] for i in range(1, self.n_components + 1)))
        self.eigen_vals_to_return = [self.eigen_vals[-i] for i in range(1, self.n_components + 1)]
        # print('self.eigen_vals_to_return: ', self.eigen_vals_to_return )
        # print('self.eigen_vecs_to_return: ', self.eigen_vectors_to_return )
        # print('sqroot: ', (self.eigen_vectors_to_return / np.sqrt(self.eigen_vals_to_return) ))
        return self.eigen_vectors_to_return

    def transform(self):
        return self.K.dot(self.eigen_vectors_to_return / np.sqrt(self.eigen_vals_to_return))

    def fit_transform(self, X):
        self.fit(X)
        transformed = self.K.dot(self.eigen_vectors_to_return / np.sqrt(self.eigen_vals_to_return))
        return transformed
