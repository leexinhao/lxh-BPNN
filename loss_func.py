import numpy as np
from help_func import one_hot
from activation_func import softmax

########## 损失函数 ##########


def mse(y_pred, y_true, method='mean', sparse=False, num_classes=None):
    r'''
    均方根
    y_pred.shape = (n_samples, n_output)
    y_true.shape = (n_samples, n_output)分类的话需要是独热码
    sparse: 为True时将y_true转换成独热码
    '''
    if sparse:
        y_true = one_hot(y_true, num_classes=num_classes).reshape(y_pred.shape)
    assert y_pred.shape == y_true.shape, f"y_pred({y_pred.shape})和y_true({y_true.shape})的shape不相等!"
    if method is None:
        return (y_pred - y_true)**2
    elif method == 'mean':
        return np.mean((y_pred - y_true)**2, axis=0)
    elif method == 'sum':
        return np.sum((y_pred - y_true)**2, axis=0)
    else:
        assert False, f"不支持method={method}!"


def grad_mse(y_pred, y_true, sparse=False, num_classes=None):
    r'''
    返回均方根误差梯度向量
    '''
    if sparse:
        y_true = one_hot(y_true, num_classes=num_classes).reshape(y_pred.shape)
    assert y_pred.shape == y_true.shape, f"y_pred({y_pred.shape})和y_true({y_true.shape})的shape不相等!"
    return -(y_true - y_pred)



def binary_cross_entropy(y_pred, y_true, sparse=False, num_classes=None):
    r'''
    TODO 还可能有bug
    返回交叉熵误差
    '''
    return -(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))


def grad_binary_cross_entropy(y_pred, y_true, sparse=False, num_classes=None):
    r'''
    TODO 还可能有bug
    返回交叉熵误差梯度向量
    '''
    return -(y_true/y_pred - (1-y_true)/(1-y_pred))


def cross_entropy(y_pred, y_true, method='mean', sparse=False, num_classes=None):
    r'''
    TODO 还可能有bug
    '''
    r'''
    为了数值稳定性直接加了softmax的交叉熵函数
    y_pred.shape = (n_samples, n_output)
    y_true.shape = (n_samples, 1) 是数值类型不是独热码
    '''
    if not sparse:  # 将独热码转成数值类型
        y_true = y_true.argmax(axis=1)
    y_true = y_true.reshape(-1)
    assert len(y_pred) == len(
        y_true), f"y_pred:{len(y_pred)}和y_true:{len(y_true)}的长度不一致"
    assert len(y_pred.shape) == 2, f"y_pred应为二维概率输出！而不是{y_pred.shape}"
    p = softmax(y_pred)
    log_likelihood = -np.log(p[range(len(y_true)), y_true])
    if method is None:
        return log_likelihood
    elif method == 'mean':
        return np.mean(log_likelihood, axis=0)
    elif method == 'sum':
        return np.sum(log_likelihood, axis=0)
    else:
        assert False, f"不支持method={method}!"


def grad_cross_entropy(y_pred, y_true, sparse=False, num_classes=None):
    r'''
    TODO 还有bug
    返回交叉熵误差梯度向量
    '''
    if not sparse:  # 将独热码转成数值类型
        y_true = y_true.argmax(axis=1)
    m = y_true.shape[0]
    grad = softmax(y_pred)
    grad[range(m), y_true] -= 1
    grad = grad / m
    return grad


