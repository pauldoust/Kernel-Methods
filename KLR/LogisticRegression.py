# https://github.com/martinpella/logistic-reg/blob/master/logistic_reg.ipynb
import numpy as np
from scipy.special import expit
from time import time

class LogisticRegression:
    def __init__(self, max_iter, learning_rate=0.001, debug = False):
        start = time()
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.debug = debug
        if self.debug:
            print(f'Init took: {time() - start} sec')

    # J(Theta)
    def _cost(self):
        term1 = -self.y.T * np.log(self.predict_prob())
        term2 = (1 - self.y).T * np.log(1 - self.predict_prob())
        result = (term1 - term2).mean()
        return result

    def predict_prob(self, X=None):
        if X is not None:
            self.bias = np.ones((X.shape[0], 1))
            self.X = np.concatenate((self.bias, X), axis=1)

        linear_prediction = np.dot(self.X, self.Theta)
        if self.debug:
            self.debug = True

        return self._sigmoid(linear_prediction)

    def predict(self, X):
        return self.predict_prob(X).round()

    def _sigmoid(self, z):
        return expit(z)
        # return 1 / (1 + np.exp(-z))

    def update_gradients(self):
        m = np.shape(self.X)[0]
        h = self.predict_prob()
        alpha = self.learning_rate
        grad = np.dot(self.X.T, (h - self.y))
        self.Theta = self.Theta - (alpha / m * grad)

    def fit(self, X, y):
        self.bias = np.ones((X.shape[0], 1))
        self.X = X
        self.y = y
        self.X = np.concatenate((self.bias, X), axis=1)
        self.Theta = np.zeros(self.X.shape[1])
        self.gradient_descent()

    # training
    def gradient_descent(self):
        for i in range(self.max_iter):
            self.update_gradients()
            # cost = self._cost()
            # if i % 10000 == 0:
            #     print ("iter: " + str(i) + " cost: " + str(cost))
            #     print("iter: " + str(i) + " theta: " + str(self.Theta))
        return self.Theta
