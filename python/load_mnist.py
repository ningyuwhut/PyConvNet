# coding = utf-8

import os
import numpy as np
import cPickle
import gzip
import download_mnist

def load_mnist():
    if os.path.exists('../data/mnist/mnist.pkl.gz'):
        pass
    else:
        print 'download mnist dataset...'
        download_mnist.download_mnist()
    # Load the dataset
    f = gzip.open('../data/mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train_data = train_set[0]#train_data 的维度是(50000, 784)

    train_data = np.reshape(train_data, (-1, 28, 28, 1))#-1表示从其余的维度推断出该维度的值，维度为(50000，28, 28, 1)
    train_data = np.transpose(train_data, (1, 2, 3, 0)) * 255 #shape为(28,28,1,50000)
    train_label = train_set[1]#长度为50000的向量
    train_label = np.reshape(train_label, (1, -1)) #变成一个1*50000的矩阵

    valid_data = valid_set[0]
    valid_data = np.reshape(valid_data, (-1, 28, 28, 1))
    valid_data = np.transpose(valid_data, (1, 2, 3, 0)) * 255
    valid_label = valid_set[1]
    valid_label = np.reshape(valid_label, (1, -1))
    return train_data, train_label, valid_data, valid_label
