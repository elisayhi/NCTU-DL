"""
simple neuron network training as a XOR gate
input: 2 bits
output: 1 bit
# of hidden layer: 1
using SGD
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y, ofilename):
    plt.subplots(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.savefig(ofilename)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    #return np.multiply(x, 1.0-x)
    return np.multiply(sigmoid(x), 1.0-sigmoid(x))

#------------------------------------------------------------------
class NN:
    def __init__(self, neuron_num=[2, 4, 5, 1], learning_rate=0.01):
        """
        initial all weights and biases
        neuron_num: number of neuron in each layer, including input and output layer
        self.net: value of neuron before active function(sigmoid)
        self.out: value of neuron after active function, including input data at [0]
        """
        # set weights
        self.W = [np.array([[np.random.randn() for _ in range(neuron_num[l-1])] for __ in range(neuron_num[l])]) for l in range(1, len(neuron_num))]
        # set biases
        self.B = [np.array([np.random.randn() for _ in range(n)]) for n in neuron_num[1:]]
        #print(f'initial B: {self.B}')
        #print(f'initial W: {self.W}')
        # other settings
        self.neuron_num = neuron_num
        self.learning_rate = learning_rate
        self.net = []
        self.out = []
        self.error = 100

    def forward(self, input):
        #print(f'W: {self.W}')
        self.out, self.net = [], []
        self.out.append(input)
        for w, b in zip(self.W, self.B):
            self.net.append(np.add(np.dot(w, self.out[-1]), b[:, None]))    # wx+b
            self.out.append(sigmoid(self.net[-1]))

    def backward(self, y):
        """
        backward propagation in the neuron network
        """
        # calc all delta
        deltas = []     # from the output layer to first hidden layer
        # the last layer
        self.error = y-self.out[-1]
        delta = np.multiply(self.error, derivative_sigmoid(self.net[-1]))
        deltas.append(delta)
        for i in range(1, len(self.neuron_num)-1):  # input layer has no delta, delta of output layer is calculated(deltas[0])
            delta = np.multiply(np.dot(self.W[-i].T, deltas[i-1]), derivative_sigmoid(self.net[-i-1]))
            deltas.append(delta)
        deltas.reverse()
        # partial derivative of W and B
        delta_W = [np.dot(deltas[i], self.out[i].T) for i in range(len(self.W))]
        delta_B = [np.sum(np.squeeze(d.T), 0) for d in deltas]
        
        # gradient descent
        self.W = np.add(self.W, np.multiply(self.learning_rate, delta_W))
        self.B = np.add(self.B, np.multiply(self.learning_rate, delta_B))

    def predicted_y(self):
        return list(self.out[-1][0])
        #y_pred = []
        #for yp in self.out[-1][0]:
        #    if yp > 0.5:
        #        y_pred.append(1)
        #    else:
        #        y_pred.append(0)
        #return y_pred

def split_batch(X, Y, batch_size):
    split_cnt = np.ceil(len(X)/batch_size)
    X_batches = np.array_split(X, split_cnt)
    Y_batches = np.array_split(Y, split_cnt)
    return X_batches, Y_batches

def calc_acc(Y, y_pred):
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    acc = np.sum(np.squeeze(Y) == y_pred)/len(Y)*100
    return acc

def train(X, Y, batch_size, learning_rate, print_epoch=5000):
    nn = NN(learning_rate=learning_rate)
    X_batches, Y_batches = split_batch(X, Y, batch_size)
    epoch = 0
    acc = 0
    loss, pre_loss = 100, 100
    while True:
        epoch += 1
        pred_y = []
        for x, y in zip(X_batches, Y_batches):
            nn.forward(x.T)
            nn.backward(y.T)
            pred_y += nn.predicted_y()

        if epoch%print_epoch == 0:
            acc = calc_acc(Y, pred_y)
            pre_loss = loss
            loss = 2*np.sum(np.square(np.subtract(np.squeeze(Y), pred_y)))
            print(f'Epoch {epoch} accuracy: {acc}%', end='\t')
            print(f'loss: {loss}', )

        if acc == 100 and abs(loss-pre_loss) < 0.005:
            print(np.array(pred_y))
            return pred_y


if __name__ == '__main__':
    # params
    batch_size = 3

    print('XOR')
    X, Y = generate_XOR_easy()
    pred_y = train(X, Y, batch_size, 0.01)
    show_result(X, Y, pred_y, 'XOR_result.png')

    print('linear')
    X, Y = generate_linear()
    pred_y = train(X, Y, batch_size, 0.01, 1000)
    show_result(X, Y, pred_y, 'linear_result.png')
