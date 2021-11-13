import numpy as np

def accuracy(y_pred, y_true):
    r'''
    y_pred是预测标签，y_true是真实标签，都是数值类型
    '''
    assert y_pred.shape == y_true.shape, f"y_pred({y_pred.shape})和y_true({y_true.shape})的shape不相等!"
    y_pred = y_pred.reshape(-1).astype(np.int32)
    y_true = y_true.reshape(-1).astype(np.int32)
    return (y_pred == y_true).astype(np.int32).sum() / len(y_true)

