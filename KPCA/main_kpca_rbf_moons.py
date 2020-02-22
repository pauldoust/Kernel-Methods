import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from kernels.KPCA.KPCA import KPCA
from kernels.dataset_generators import Generator

np.random.seed(0)

X, y = Generator.generate(dataset_name="moons", n_samples=100, random_state=123)
reds = y == 0
blues = y == 1

# My KPCA
kpcaModel = KPCA(kernel="rbf", bandwidth=15, n_components=2)
new_dataset = kpcaModel.fit_transform(X)

# scikit_rbf kpca
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.figure()
plt.subplot(3, 1, 1)
plt.title("Original space")
plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplot(3, 2, 3)
plt.title("KPCA(rbf,g =15,2 comps)")
plt.scatter(new_dataset[reds, 0], new_dataset[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(new_dataset[blues, 0], new_dataset[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("1st component")
plt.ylabel("2nd component")

plt.subplot(3, 2, 4)
plt.title("scikit(rbf,g =15,2 comps)")
plt.scatter(X_skernpca[reds, 0], X_skernpca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_skernpca[blues, 0], X_skernpca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("1st component")
plt.ylabel("2nd component")

plt.subplot(3, 2, 5)
plt.title("KPCA(rbf,g= 15,1 comp)")
plt.scatter(new_dataset[reds, 0], np.zeros(50), c="red",
            s=20, edgecolor='k')
plt.scatter(new_dataset[blues, 0], np.zeros(50), c="blue",
            s=20, edgecolor='k')
plt.xlabel("1st component")
plt.ylabel("2nd component")

plt.subplot(3, 2, 6)
plt.title("scikit(rbf,g=15,1 comp)")
plt.scatter(X_skernpca[reds, 0], np.zeros(50), c="red",
            s=20, edgecolor='k')
plt.scatter(X_skernpca[blues, 0], np.zeros(50), c="blue",
            s=20, edgecolor='k')
plt.xlabel("1st component")
plt.ylabel("2nd component")

plt.tight_layout()
plt.show()
