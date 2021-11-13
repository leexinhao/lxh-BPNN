import numpy as np
from help_func import one_hot

########## 损失函数 ##########

def mse(y_pred, y_true, method='mean', sparse=True):
    r'''
    均方根
    y_pred.shape = (n_samples, n_output)
    y_true.shape = (n_samples, n_output)分类的话需要是独热码
    sparse: 为True时将y_true转换成独热码
    '''
    # if sparse:
    #     y_true = one_hot(y_true) TODO 这有大问题，one_hot不能这么用，应该指定标签数
    assert y_pred.shape == y_true.shape, f"y_pred({y_pred.shape})和y_true({y_true.shape})的shape不相等!"
    if method is None:
        return (y_pred - y_true)**2
    elif method == 'mean':
        return np.mean((y_pred - y_true)**2, axis=0)
    elif method == 'sum':
        return np.sum((y_pred - y_true)**2, axis=0)
    else:
        assert False, f"不支持method={method}!"


def grad_mse(y_pred, y_true):
    r'''
    返回均方根误差梯度向量
    '''
    assert y_pred.shape == y_true.shape, f"y_pred({y_pred.shape})和y_true({y_true.shape})的shape不相等!"
    return -(y_true - y_pred)


# def binary_cross_entropy(y_pred, y_true):
#     return -(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))


# def grad_binary_cross_entropy(y_pred, y_true):
#     return -(y_true/y_pred - (1-y_true)/(1-y_pred))


# def cross_entropy(y_pred, y_true):
#     r'''
#     为了数值稳定性直接加了softmax的交叉熵函数
#     '''
#     m = y_true.shape[0]
#     p = softmax(y_pred)
#     log_likelihood = -np.log(p[range(m), y_true])
#     loss = np.sum(log_likelihood) / m
#     return loss


# def grad_cross_entropy(y_pred, y_true):
#     m = y_true.shape[0]
#     grad = softmax(y_pred)
#     grad[range(m), y_true] -= 1
#     grad = grad / m
#     return grad


