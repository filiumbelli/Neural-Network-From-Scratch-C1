import numpy as np

class ActivationSoftmax:
    def forward(self,inputs):
        self.exponential = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.output = self.exponential/np.sum(self.exponential,axis=1,keepdims=True)

class ActivationReLU:
    def forward(self,inputs):
        self.output = np.array(np.maximum(0,inputs))

