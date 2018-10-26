from sklearn.neural_network import MLPClassifier

x = [[-1, 6], [3, 34], [1, 0], [20, 9], [-5, 2],
     [0, -44], [80, 27], [120, 39], [0, 12], [-23, 12]]
y = [-1, 1, 0, 1, -1, 
        0, 1, 1, 0, -1]

classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                           hidden_layer_sizes=(12, 4), random_state=1)
classifier.fit(x, y)

test_x = [[5, 2], [-4, 1], [0, 4], [-2, 33], [39, 0], [0, 22]]
test_y = [1, -1, 0, -1, 0, 0]

predicted = classifier.predict(test_x)

print(test_x)
print(test_y)
print(predicted)
