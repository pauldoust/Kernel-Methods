from kernels.KLR.LogisticRegression import LogisticRegression as myLr
from kernels.KLR.KernelLogisticRegression import KernelLogisticRegression
import numpy as np
from time import time
import matplotlib.pyplot as plt
from kernels.dataset_generators import Generator
from sklearn.linear_model import LogisticRegression

nsamples = 100
X, y = Generator.generate(dataset_name="manual_circles", n_samples=nsamples)

plt.figure()
plt.subplot(1, 1, 1)
plt.title("Original Datasset")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


reds = y == 0
blues = y == 1
two_third_of_data = int(nsamples * (2/3))
X_train = X[:two_third_of_data]
y_train = y[:two_third_of_data]
X_test = X[two_third_of_data:]
y_test = y[two_third_of_data:]

# print("X_train: ", X_train)
print("logistic regression")
start = time()
model = myLr(max_iter=100000, learning_rate=0.01)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('accuracy: ', (preds == y_test).mean() * 100)
print(f'execution time: {time() - start} sec')


print("Sklearn logistic regression")
start = time()
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
preds = clf.predict(X_test)
print('accuracy: ', (preds == y_test).mean() * 100)
print(f'execution time: {time() - start} sec')


print("kernelized logistic regression")
start = time()
model = KernelLogisticRegression(max_iter=100000, learning_rate=0.01, kernel="rbf")
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('accuracy: ', (preds == y_test).mean() * 100)
print(f'execution time: {time() - start} sec')



x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_prob(grid).reshape(xx.shape)

# print("probs: ", probs)
plt.contour(xx, yy, probs, [0.5], linewidths=1, colors='black');


plt.show();