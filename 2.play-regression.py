import numpy as np
from sklearn import datasets, linear_model

x = np.arange(100)
#print(x)
x = x.reshape(-1, 1)
#print(x)
y = x * 2

regression_handler = linear_model.LinearRegression()
regression_handler.fit(x[:x.size//2], y[:x.size//2])

predicted = regression_handler.predict(np.arange(20, 30).reshape(-1, 1))

print(predicted[:20])
