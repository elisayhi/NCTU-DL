import numpy as np
from layer import rnn_layer

def dsigmoid(x):
    return x * (1-x)

class RNN():
    def __init__(self, learning_rate=1e-3):
        self.lr = learning_rate
        self.time_step = 8
        x = 2
        y = 1
        h = 16
        self.U = np.random.randn(h, x)
        self.W = np.random.randn(h, h)
        self.V = np.random.randn(y, h)
        self.B = np.random.randn(h, 1)
        self.C = np.random.randn(y, 1)
        #self.U = np.ones((h, x))
        #self.W = np.ones((h, h))
        #self.V = np.ones((y, h))
        #self.B = np.ones((h, 1))
        #self.C = np.ones((y, 1))

        self.hidden = rnn_layer([2, 16], activation='sigmoid', weights1=self.U, weights2=self.W, bias=self.B)
        #self.hidden = rnn_layer([2, 16], activation='tanh', weights1=self.U, weights2=self.W, bias=self.B)
        self.outlayer = rnn_layer([16, 1], activation='sigmoid', weights1=self.V, bias=self.C)
        self.hidden_output = []
        self.pred = []

    def forward(self, X):
        """
        x: input of all time step (8, 2, 1)
        """
        h0 = np.zeros((16, 1))
        self.pred = []
        self.hidden_output = [h0]
        for x in X:
            h = self.hidden.forward(x, self.hidden_output[-1])
            #print('h', h)
            self.hidden_output.append(h)
            #print('out',self.outlayer.forward(self.hidden_output[-1]))
            self.pred.append(self.outlayer.forward(self.hidden_output[-1]))
        #print('h', self.hidden_output)
        #print('pred', self.pred)

    def backward(self, x, true):
        #print(true)
        #true = np.reshape(true, (8,1,1))
        #print('Y', true)
        H = []
        for t in range(1, self.time_step+1):
            #H_t = 1 - (self.hidden_output[t].T.squeeze())**2   # if activation is tanh
            H_t = dsigmoid(self.hidden_output[t].T.squeeze())  # if activation is sigmoid
            H_t = np.diag(H_t)
            H.append(H_t)
        delta1 = np.multiply(np.subtract(self.pred, true), dsigmoid(np.array(self.pred)))    # (time_step, |y|, 1)
        delta2 = []
        delta2.append(np.dot(self.V.T, delta1[-1]))
        for i in range(self.time_step-1, 0, -1):
            delta2.append(np.dot(np.dot(self.W.T, H[i]), delta2[-1]) + np.dot(self.V.T, delta1[i-1]))
        delta2.reverse()
        
        #-------------------------------------------------------------------------------------------
        self.C -= self.lr * np.sum(delta1, 0)
        deltaV, deltaB, deltaU, deltaW = [], [], [], []
        for t in range(self.time_step):
            deltaB.append(np.dot(H[t], delta2[t]))
            deltaW.append(np.dot(deltaB[-1], self.hidden_output[t].T))
            deltaU.append(np.dot(deltaB[-1], x[t].T))
            deltaV.append(np.dot(delta1[t], self.hidden_output[t+1].T))
        deltaV = np.sum(deltaV, 0)
        deltaB = np.sum(deltaB, 0)
        deltaW = np.sum(deltaW, 0)
        deltaU = np.sum(deltaU, 0)
        #print('W', deltaW)
        #print('U', deltaU)
        #print('V', deltaV)
        #print('B', deltaB)
        #print('C', np.sum(delta1, 0))

        # SGD
        self.V -= self.lr * deltaV
        self.B -= self.lr * deltaB
        self.W -= self.lr * deltaW
        self.U -= self.lr * deltaU
        #print(self.B)
        self.hidden.update_weight(self.U, self.B, self.W)
        self.outlayer.update_weight(self.V, self.C)
        #print(self.outlayer.U)
        #print(self.V)

    def train(self, x, y):
        """
        x: (time_step, |x|, 1)
        y: (time_step, |y|, 1)
        """
        self.forward(x)
        self.backward(x, y)
        return self.pred
