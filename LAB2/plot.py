import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def plot(datas, names):
    """
    datas: data to print, [data1, data2, ...]
    names: name according to data, [name1. name2, ...]
    """
    plt.figure()
    for data, name in zip(datas, names):
        plt.plot(data, label=name)
        plt.legend(loc=0)
    plt.savefig('result.png')
