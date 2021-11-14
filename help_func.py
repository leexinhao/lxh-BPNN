import numpy as np


def one_hot(X, num_classes=None):
    r'''
    TODO 这个写法有问题，应该指定标签数
    '''
    assert len(X.shape) == 1, f"X.shape={X.shape}，输入必须是数值型向量！"
    if num_classes is None:
        num_classes = X.max() + 1  # 从0开始数
    assert X.max() < num_classes, f"输入标签最大值：{X.max()}超过了指定的类别数：{num_classes}！"
    return np.eye(num_classes)[X]
