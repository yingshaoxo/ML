print('\n'*50)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=3, input_dim=3)) #`input_dim`
model.add(Dense(units=1)) #`last layer's units
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])



import numpy as np

x = [[1,2,3], [1,2,1]] #you got 3 numbers in every sub-list, so `input_dim=3`
y = [[5], [4]] #you got 1 numbers in every target sub-list, so you have to make sure the `last layer's units = 1`

x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
print()
print(x)
print(y)
print()

model.fit(x, y, epochs=1, batch_size=1)



print("\n\n\n" + "made by yingshaoxo" + "\n\n\n")



inputs = np.array([[1,1,1]])
print(model.predict(inputs))
