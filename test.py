import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.tree import DecisionTreeClassifier

x = np.array([[260, 1, 417], [305, 1, 495], [312, 1, 703], [514, 1, 897], [588, 0, 529], [326, 1, 544] , [367, 1, 617], [92, 1, 100], [396, 1, 630], [241, 0, 191], [703, 1, 885], [136, 0, 176], [362, 1, 567], [118, 1, 195], [286, 1, 449], [459, 0, 408]])
x = x.reshape(-1, 3)
print(x.shape)
print(x)

y = np.array([40, 39, 65, 148, 53, 56, 49, 29.8, 48, 38, 126, 30, 46, 22, 33, 35])
y = y.reshape(-1, 1)
print(y.shape)
print(y)

regression_handler = DecisionTreeClassifier()
#regression_handler = linear_model.LinearRegression()
regression_handler.fit(x, y)

expect = np.array([260, 1, 468])
expect = expect.reshape(-1, 3)
predicted = regression_handler.predict(expect)

print(predicted[:20])
