import matplotlib.pyplot as plt
import nnfs
import numpy as np
from nnfs.datasets import vertical_data
from nnfs.datasets import spiral_data
from Libraries.ActivationFunctions import ActivationReLU, ActivationSoftmax
from Libraries.CostFunctions import CrossEntropy
from Libraries.Layer import LayerNetwork
# X,y = spiral_data(samples=100,classes=3)
X, y = vertical_data(samples=100, classes=3)
# plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap="brg")
# plt.show()
dense1 = LayerNetwork(2, 3)
activation1 = ActivationReLU()
dense2 = LayerNetwork(3, 3)
activation2 = ActivationSoftmax()

loss_function = CrossEntropy()
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.bias.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.bias.copy()


# Updated for each iteration with the best solution
for iteration in range(100000):
    dense1.weights += 0.05 * np.random.rand(2, 3)
    dense1.bias += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.bias += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print("New set of weights found,iteration: ", iteration,
              "loss: ", loss, " accuracy: ", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.bias.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.bias.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.bias = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.bias = best_dense2_biases.copy()
