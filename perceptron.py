import numpy as np
import matplotlib.pyplot as plt


def optimizer(X, y, eta=1, max_iter=10000):
    '''
    :param X: input feature,is a ndarray type of matrix,m rows and n columns,m is the number of sample and n is the dimension of vector
    :param y: label,is a ndarray type of matrix,1 row and n columns,positive one is +1 and negative one is -1
    :param eta:learning rate
    :param max_iter:max number of iteration
    :return:weights w and the iteration number i in convergence
    '''
    m, n = X.shape
    w = np.zeros((n, 1))
    flag = 0
    for i in range(max_iter):
        x = X[i % m].reshape((n, 1))
        x = x * y[i % m]
        if np.dot(w.T, x) <= 0:
            w = w + eta * x
            flag = 0
        else:
            flag = flag + 1
        if flag == m:
            i = i - m
            break
    x1 = np.linspace(0, 1, 10)
    x2 = -(w[0] * x1 + w[2]) / w[1]
    plt.plot(x1, x2)
    return w, i


def gene_dataset(sample_num, hoped_w):
    '''
    :param sample_num: sample number
    :param hoped_w: the adarray shaped (n, 1),which are used to generate label y
    :return: X, y,the dataset,y is 1 or -1
    '''
    x1 = np.random.rand(sample_num)
    x2 = np.random.rand(sample_num)
    x3 = np.ones(sample_num, np.float32)
    X = np.vstack((x1, x2, x3)).T
    y = np.ones((sample_num, 1))
    x1_negative = []
    x2_negative = []
    x1_positive = []
    x2_positive = []
    for i in range(sample_num):
        if np.dot(hoped_w.T, X[i].reshape((3, 1))) < 0:
            y[i] = -1
            x1_negative.append(X[i, 0])
            x2_negative.append(X[i, 1])
        else:
            x1_positive.append(X[i, 0])
            x2_positive.append(X[i, 1])
    y = y.reshape((sample_num, 1))
    plt.scatter(x1_positive, x2_positive, c='r')
    plt.scatter(x1_negative, x2_negative, c='b')
    return X, y


def main():
    hoped_w = np.array([1, 1, -1]).reshape((3, 1))
    X, y = gene_dataset(100, hoped_w)
    w, i = optimizer(X, y, eta=1)
    print('the final optima result weights w:')
    print(w)
    print('iteraion number i:', i)
    plt.show()


if __name__=='__main__':
    main()
