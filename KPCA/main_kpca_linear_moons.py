import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

from kernels.KPCA.KPCA import KPCA

from kernels.dataset_generators import Generator

np.random.seed(0)

nsamples = 50
half_samples = 25
X, y = Generator.generate(dataset_name="moons", n_samples=nsamples, random_state=123)

reds = y == 0
blues = y == 1

# My KPCA
kpcaModel = KPCA(kernel="linear", n_components=2)
new_dataset = kpcaModel.fit_transform(X)

# scikit_lear kpca
scikit_kpca = KernelPCA(n_components=2, kernel='linear')
X_skernpca = scikit_kpca.fit_transform(X)


plt.figure()
plt.subplot(3, 1, 1)
plt.title("Original space")
plt.scatter(X[reds, 0], X[reds, 1], color="red", alpha= 0.5)
plt.scatter(X[blues, 0], X[blues, 1], color="blue", alpha= 0.5)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


plt.subplot(3, 2, 3)
plt.title("KPCA(linear,2 comps)")
plt.scatter(new_dataset[reds, 0], new_dataset[reds, 1], color="red",alpha= 0.5)
plt.scatter(new_dataset[blues, 0], new_dataset[blues, 1], color="blue",alpha= 0.5)
plt.xlabel("1st component")
plt.ylabel("2nd component")



plt.subplot(3, 2, 4)
plt.title("scikit(linear,2 comps)")
plt.scatter(X_skernpca[reds, 0], X_skernpca[reds, 1], c="red",alpha= 0.5)
plt.scatter(X_skernpca[blues, 0], X_skernpca[blues, 1], c="blue",alpha= 0.5)
plt.xlabel("1st component")
plt.ylabel("2nd component")


plt.subplot(3, 2, 5)
plt.title("KPCA(linear,1 comp)")
plt.scatter(new_dataset[reds, 0], np.zeros(half_samples), color="red",alpha= 0.5)
plt.scatter(new_dataset[blues, 0], np.zeros(half_samples), color="blue",alpha= 0.5)
plt.xlabel("1st component")



plt.subplot(3, 2, 6)
plt.title("scikit(linear,,1 comp)")
plt.scatter(X_skernpca[reds, 0], np.zeros((half_samples,1)), color="red",alpha= 0.5)
plt.scatter(X_skernpca[blues, 0], np.zeros((half_samples,1)), color="blue",alpha= 0.5)
plt.xlabel("1st component")

plt.tight_layout()
plt.show()