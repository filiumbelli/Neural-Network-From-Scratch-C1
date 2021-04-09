import numpy as np
import nnfs
from Cryptodome import Math
from nnfs.datasets import spiral_data

nnfs.init()


class LayerNetwork:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+ self.biases


class ActivationRelu:
    def forward(self,inputs):
        self.output = np.array(np.maximum(0,inputs))

X,y = spiral_data(samples=100,classes=3)

dense1 = LayerNetwork(2,3)
activation1 = ActivationRelu()
dense1.forward(X)
activation1.forward(dense1.output)

#--------------------- SOFTMAX ---------------------#

class ActivationSoftmax:
    def forward(self,inputs):
        self.expValues = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        self.output =  self.expValues/ np.sum(self.expValues,axis=1,keepdims=True)

# layer_outputs = [4.8,1.21,2.385]

dense2 = LayerNetwork(3,3)
activation2 = ActivationSoftmax()
# activation2.forward([layer_outputs])
# print(activation2.expValues)
# print(activation2.output)
dense2.forward(dense1.output)
activation2.forward(dense2.output)

print(activation2.output)