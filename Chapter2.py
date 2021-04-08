import numpy as np

# First Part ---------------->
# inputs = [1, 2, 3, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2;
#
# output = (inputs[0] * weights[0] +
#           inputs[1] * weights[1] +
#           inputs[2] * weights[2] +
#           inputs[3] * weights[3]
#           ) + bias
# print(output)
# First Part ---------------->
import numpy as np

# First Part ---------------->
# inputs = [1, 2, 3, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2;
#
# output = (inputs[0] * weights[0] +
#           inputs[1] * weights[1] +
#           inputs[2] * weights[2] +
#           inputs[3] * weights[3]
#           ) + bias
# print(output)
# First Part ---------------->

# inputs = [1, 2, 3, 2.5]
# weights = [
#     [0.2, 0.8, -0.5, 1.0],
#     [0.5, -0.91, 0.26, -0.5],
#     [-0.26, -0.27, 0.17, 0.87]
# ]
# biases = [2,3,0.5]
# #
# # outputs = []
# #
# # for n_biases,n_weights in zip(biases,weights):
# #     result = 0
# #     for weight,input in zip(n_weights,inputs):
# #         result += weight * input
# #     result += n_biases
# #     outputs.append(result)
# #
# # print(outputs)
#
# outputs = np.dot(np.expand_dims(weights,axis=0),np.array(inputs)) + biases
#
# print(outputs)

#
# outputs = []
#
# for n_biases,n_weights in zip(biases,weights):
#     result = 0
#     for weight,input in zip(n_weights,inputs):
#         result += weight * input
#     result += n_biases
#     outputs.append(result)
#
# print(outputs)

inputs = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2, 3, 0.5]

# Note the use of double brackets here.
# To transform a list into a matrix containing a single row
# (perform an equivalent operation of turning a vector into row vector),
# we can put it into a list and create numpy array:
inputs = np.array(inputs)
weights = np.array(weights)
outputs = np.dot(inputs,weights.T) + biases

print(outputs)
