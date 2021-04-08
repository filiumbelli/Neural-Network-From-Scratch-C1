# Neural networks become “deep”
# when they have 2 or more hidden layers
import matplotlib.pyplot as plt
import numpy as np
from nnfs.datasets import spiral_data
import nnfs

class LayerNetwork:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+ self.biases



nnfs.init()

inputs = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
]
#
# weights1 = [
#     [0.2, 0.8, -0.5, 1.0],
#     [0.5, -0.91, 0.26, -0.5],
#     [-0.26, -0.27, 0.17, 0.87]
# ]
# biases1 = [2, 3, 0.5]
#
# weights2 = [
#     [0.1, -0.14, 0.5],
#     [-0.5, 0.12, -0.33],
#     [-0.44, 0.73, -0.13]
# ]
# biases2 = [-1, 2, -0.5]
#
# layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
# print(layer2_outputs)

X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
# plt.show()
#
dense1 = LayerNetwork(2,3)
dense1.forward(X)
print(dense1.output[0:5])
