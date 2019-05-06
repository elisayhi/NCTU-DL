import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from rnn import RNN
from gen_data import gen_data

def calc_error(true, pred):
    error = pred.size - np.sum(pred==true)
    return error

def calc_accuracy(true, pred):
    same = [np.array_equal(p, t) for p, t in zip(pred, true)]
    acc = np.sum(same) / len(pred)
    return acc

def plot(acc, error):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Result')
    plt.ylabel('Accuracy (%)')
    plt.plot(acc, label='accuracy')
    #plt.legend(loc=0)
    plt.subplot(2, 1, 2)
    plt.plot(error, label='error')
    plt.ylabel('Error')
    plt.xlabel('epoch')
    #plt.legend(loc=0)
    plt.savefig('result/result.png')

if __name__ == '__main__':
    EPOCH = 20
    rnn = RNN(0.1)
    acc, error = [], []
    for e in range(EPOCH):
        X, Y = gen_data(1000)
        pred = []
        for x, y in zip(X, Y):
            x, y = np.array(x), np.array(y)
            pred.append(rnn.train(x, y))
        pred = np.array(pred).squeeze()
        true = np.array(Y).squeeze()
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        pred = np.array(pred, dtype=np.uint8)
        error.append(calc_error(true, pred)/100)
        acc.append(calc_accuracy(true, pred)*100)
        print('error: ', error[-1])
        print('acc: ', acc[-1], '%')
    plot(acc, error)
            

