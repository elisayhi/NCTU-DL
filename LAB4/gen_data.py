import numpy as np
from bin_add import binAdd

def gen_point():
    x = [str(i) for i in np.random.binomial(1, .5, 8)]
    x = ''.join(x)
    return x

def gen_data_unit():
    x1 = gen_point()
    x2 = gen_point()
    y = binAdd(x1, x2)
    x = np.array([list(x1), list(x2)]).T.astype(int)
    y = np.reshape(list(y), (8, 1, 1)).astype(int)
    x = np.flip(x,0).reshape((8, 2, 1))
    return x, y[::-1]
    
def gen_data(n=1000):
    X, Y = [], []
    for _ in range(n):
        x, y = gen_data_unit()
        X.append(x)
        Y.append(y)
    return X, Y

