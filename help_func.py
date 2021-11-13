import numpy as np

def one_hot(X):
    r'''
    TODO 这个写法有问题，应该指定标签数
    '''
    assert len(X.shape) == 1, f"X.shape={X.shape}，输入必须是数值型向量！"
    X = X - X.min()  # 从0开始数
    return np.eye(X.max()+1)[X]
