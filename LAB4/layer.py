import numpy as np

def sigmoid(x):
    return 1. / (1+np.exp(-x))

class rnn_layer():
    def __init__(self, neuron_num, activation=None, weights1=None, weights2=None, bias=None):
        """
        neuron_num: [#input layer, #neuronlayer]
        """
        self.U = weights1 if weights1 is not None else np.random.randn(neuron_num[1], neuron_num[0])
        self.W = weights2
        self.b = bias if bias is not None else np.random.randn(neuron_num[1], 1)
        #print('U', self.U)
        #print('W', self.W)
        #print('b', self.b)
        self.activation = activation

    def update_weight(self, weight1, bias, weight2=None):
        self.U = weight1
        self.W = weight2
        self.b = bias

    def forward(self, x, h=None):
        """
        x: input of the layer, shape: (|x|, 1)
        """
        #print(self.activation, self.b)
        if self.W is not None and h is not None:
            output = np.dot(self.U, x) + np.dot(self.W, h) + self.b
            #print(self.activation)
            #print(output)
            #print(sigmoid(output))
        else:
            output = np.dot(self.U, x) + self.b
        #print(self.activation, output)
        return self._apply_activation(output)

    def _apply_activation(self, out):
        if self.activation is None:
            return out
        elif self.activation is 'tanh':
            return np.tanh(out)
        elif self.activation is 'sigmoid':
            #print('sigmoid', out, sigmoid(out))
            return sigmoid(out)
        else:
            print('[activation] error')
            return out

