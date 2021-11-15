import numpy as np


########## 激活函数 ##########

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def grad_sigmoid(X):
    return sigmoid(X) * (1.0 - sigmoid(X))


def relu(X):
    return np.maximum(X, np.zeros_like(X))


def grad_relu(X):
    return (X > 0).astype(np.float32)


def softmax(X):
    shift_X = X - np.max(X)
    exps = np.exp(shift_X)
    return exps / np.sum(exps)


def grad_softmax(X):
    return softmax(X) * (1.0 - softmax(X))



