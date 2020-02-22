from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import numpy as np


class Generator(object):
    @staticmethod
    def _make_dataset(N):
        X = np.zeros((N, 2))
        y = np.zeros((N))
        half = int(N / 2)
        # print('half: ', half)
        X[:half, 0] = 10 * np.cos(np.linspace(0.2 * np.pi, half, num=half))
        X[half:, 0] = np.random.randn(half)

        X[: half, 1] = 10 * np.sin(np.linspace(0.2 * np.pi, half, num=half))
        X[half:, 1] = np.random.randn(half)
        y[:half] = int(0)
        y[half:] = int(1)
        return X,y

    @staticmethod
    def generate(dataset_name, random_state=0, n_samples=100):
        X = None
        y = None
        if dataset_name == "moons":
            X, y = make_moons(n_samples=n_samples, random_state=random_state)
        elif dataset_name == "circles":
            X, y = make_circles(n_samples=n_samples, factor=.5, random_state=random_state)
        elif dataset_name == "manual_circles":
            X,y = Generator._make_dataset(n_samples)

        return X, y
