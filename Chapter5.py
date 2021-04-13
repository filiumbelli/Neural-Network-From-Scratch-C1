from Libraries.ActivationFunctions import *
from Libraries.CostFunctions import CrossEntropy
from Libraries.Layer import *
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
#
# softmax_output = [0.7,0.1,0.2]
# target_output = [1,0,0]
#
# output = -np.dot(np.log(softmax_output),np.array(target_output).T)
# print(output)

softmax_output = [[0.7, 0.1, 0.2],
                  [0.1, 0.5, 0.4],
                  [0.02, 0.9, 0.08]]
target_output = [0, 1, 1]
# for targ_idx, distribution in zip(target_output, softmax_output):
#     print(distribution[targ_idx])
# activationCrossEntropy = CrossEntropy()
# softmax_output = np.array(softmax_output)
# y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
# output = activationCrossEntropy.calculate(softmax_output,np.array(target_output))
X, y = spiral_data(samples=100, classes=3)
dense1 = LayerNetwork(2, 3)
activation1 = ActivationReLU()
dense2 = LayerNetwork(3, 3)
activation2 = ActivationSoftmax()
lossFunction = CrossEntropy()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = lossFunction.calculate(activation2.output, y)
print(activation2.output[0:5])
print("Loss: ", loss)
predictions = np.argmax(activation2.output, axis=1)
if (len(y.shape) == 2):
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print("acc: ", accuracy)
