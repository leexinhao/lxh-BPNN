import numpy as np
import matplotlib.pyplot as plt
from typing import List
from activation_func import *
from loss_func import *
from metric_func import *
from help_func import one_hot

activation_func_dict = {'sigmoid': (sigmoid, grad_sigmoid), 'relu': (
    relu, grad_relu), 'softmax': (softmax, grad_softmax)}

loss_func_dict = {'mse': (mse, grad_mse)}
# loss_func_dict = {'mse': (mse, grad_mse), 'binary_cross_entropy': (
#     binary_cross_entropy, grad_binary_cross_entropy), 'cross_entropy': (
#     cross_entropy, grad_cross_entropy}

metric_func_dict = {'accuracy': accuracy}


class MyDenseLayer:
    r'''
    全连接神经网络层
    '''

    def __init__(self, in_features, out_features, bias=True, activation='relu'):
        self.in_features = in_features
        self.out_features = out_features
        assert activation in activation_func_dict.keys(
        ), f"不支持的激活函数：{activation}"
        self.activation = activation_func_dict[activation][0]
        self.grad_activation = activation_func_dict[activation][1]
        self.bias = bias


class MySequential:
    r'''
    全连接神经网络，使用反向传播算法更新
    '''

    def __init__(self, layers: List[MyDenseLayer]):
        # 检查网络层组合合理性
        assert len(layers) > 0, f"至少有一层神经元"
        for i in range(1, len(layers)):
            assert layers[i].in_features == layers[i -
                                                   1].out_features, f"第{i}层网络的输入维度应与第{i+1}层网络的输出维度一致"
        self.layers = layers
        self.input_dim = layers[0].in_features  # 输入层神经元数量
        self.output_dim = layers[-1].out_features  # 输出层神经元数量
        self.W = []  # 各层间权重
        for layer in layers:
            # 使用高斯分布初始化各层权重
            if layer.bias:  # 权重和bias共用一个W
                self.W.append(np.random.normal(
                    0.0, pow(layer.out_features, -0.5),
                    (layer.in_features+1, layer.out_features)))
            else:
                self.W.append(np.random.normal(
                    0.0, pow(layer.out_features, -0.5),
                    (layer.in_features, layer.out_features)))

    def forward(self, X, return_details=False):
        r'''
        前向传播计算输出
        '''
        if len(X.shape) == 1:  # 如果输入为向量，reshape成(1, n)的矩阵，即行向量
            X = X.reshape((1, len(X)))
        if return_details:
            inputs = []
            outputs = [X.T]
            for i, layer in enumerate(self.layers):
                if layer.bias:  # 添加偏置项
                    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
                X = np.matmul(X, self.W[i])
                inputs.append(X.T)  # 要求input为列向量
                X = layer.activation(X)
                outputs.append(X.T)  # 要求output为列向量
            return X, inputs, outputs
        else:
            for i, layer in enumerate(self.layers):
                if layer.bias:  # 添加偏置项
                    X = np.concatenate(
                        [X, np.ones((len(X), 1))], axis=1)
                X = np.matmul(X, self.W[i])
                X = layer.activation(X)
            return X

    def update_weight(self, learning_rate, batch_size):
        r'''
        更新权重，目前只支持用所有batch的平均值更新
        '''
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * self.grad_W[i] / batch_size

    def zero_grad(self):
        r'''
        清空累计梯度，也起到初始化累计梯度的作用
        '''
        self.grad_W = [np.zeros_like(w) for w in self.W]

    def backward(self, grad_loss, inputs, outputs):
        r"""
        delta: 初始值delta_0， 为列向量
        假设inputs, outpus存的都是列向量，即[n, 1]shape的矩阵， 且grad_loss, grad_S输出维度和输入维度一致
        TODO outputs[-2]有可能是输入，令outputs开头插入输入X的话会使得outputs的size+1，代码可读性变差
        但是为了减少条件判断提升效率还是采用了这种办法
        因而从0开始往后数的时候outputs[i]实际对应的就是outputs_{i-1}
        从-1往前数没有影响，因为是outputs开头插入了X
        计算梯度
        """
        delta = grad_loss * \
            self.layers[-1].grad_activation(inputs[-1])  # grad_loss * grad_S
        tmp_output = outputs[-2]  # outputs[-2]有可能是输入X，不过无需特判
        if self.layers[-1].bias:  # 偏置权重也要更新
            tmp_output = np.concatenate([tmp_output, np.ones((1, 1))], axis=0)
        # 马上算grad是为了不存delta数组
        self.grad_W[-1] += np.matmul(tmp_output, delta.T)
        for i in range(len(self.layers)-2, -1, -1):
            if self.layers[i+1].bias:  # 更新delta时不算偏置那项
                delta = np.matmul(self.W[i+1][:-1, :], delta)
            else:
                delta = np.matmul(self.W[i+1], delta)
            delta *= self.layers[i].grad_activation(inputs[i])
            # 计算梯度
            tmp_output = outputs[i]  # 实际是outputs_{i-1}
            if self.layers[i].bias:  # 偏置权重也要更新
                tmp_output = np.concatenate(
                    [tmp_output, np.ones((1, 1))], axis=0)
            # 马上算grad是为了不存delta数组
            self.grad_W[i] += np.matmul(tmp_output, delta.T)

    def fit(self, X_train, y_train, loss='mse', learning_rate=0.001,
            batch_size=32, epoch=5, shuffle=True, X_valid=None, y_valid=None, valid_spilt=None):
        r'''
        更新神经网络参数以拟合训练集，即训练神经网络
        X_train, y_train: 训练集
        loss: 损失函数
        learning_rate, batch_size, epoch: 超参数
        shuffle: 打乱训练集，先打乱再划分验证集（如果需要划分的话）
        X_valid, y_valid: 验证集
        valid_spilt: 训练集划分给验证集的比例，当指定验证集时此参数无效
        '''
        assert loss in loss_func_dict.keys(), f"不支持的损失函数：{loss}"
        if shuffle:
            random_seed = np.random.randint(0, 100)
            np.random.seed(random_seed)
            np.random.shuffle(X_train)
            np.random.seed(random_seed)
            np.random.shuffle(y_train)
        # 尝试加载验证集
        if valid_spilt is not None:
            if X_valid is None or y_valid is None:
                valid_num = int(len(X_train)*valid_spilt)
                X_valid = X_train[0:valid_num, :]
                y_valid = y_train[0:valid_num]
                X_train = X_train[valid_num:, :]
                y_train = y_train[valid_num:]
            has_valid = True
        else:
            has_valid = False
        epoch_losses = []  # 记录每个epoch的训练误差
        self.zero_grad()  # 初始化梯度变化矩阵
        for e in range(epoch):
            losses = []
            print(f'Epoch {e+1}, ', end="")
            batch_cnt = 0  # TODO 这个方法其实有点蠢了，batch可以用矩阵计算的
            for X, y in zip(X_train, y_train):
                # if loss == 'mse': # 保险起见先加个特判，TODO 事实上应该改成from_logit判断
                #     y = y.reshape((1, -1)) # 这里对于交叉熵可能有问题，因为会产生[[1]]这种 TODO 对于batch同时算也可能有问题
                batch_cnt += 1
                # 前向传播计算输出
                y_hat, inputs, outputs = self.forward(X, return_details=True)
                output_loss = loss_func_dict[loss][0](
                    y_hat, y.reshape((1, -1)))  # 计算损失函数值
                losses.append(output_loss)
                output_grad_loss = loss_func_dict[loss][1](
                    outputs[-1], y.reshape(-1, 1))  # TODO shape应为: (output_dim, 1)
                self.backward(output_grad_loss, inputs, outputs)
                if batch_cnt == batch_size:
                    batch_cnt = 0
                    self.update_weight(learning_rate, batch_size)
                    self.zero_grad()
            # 不一定刚好有那么多个batch
            if batch_cnt != 0:
                self.update_weight(learning_rate, batch_cnt)
                self.zero_grad()
            losses = np.array(losses)
            print(f'training loss = {losses.mean()}')
            epoch_losses.append(losses.mean())
            if has_valid:
                print('valid', end=" ")
                self.evaluate(X_valid, y_valid)
        # 绘制训练误差下降曲线
        plt.plot(epoch_losses)
        plt.xlabel('epoch')
        plt.ylabel('training loss')
        plt.show()

    def predict(self, X):
        r'''
        前向传播预测结果
        '''
        y_hat = self.forward(X, return_details=False)
        return y_hat.argmax(axis=1)

    def evaluate(self, X_test, y_test, metric='accuracy'):
        assert metric in metric_func_dict.keys(), f"不支持的评价指标：{metric}"
        y_pred = self.predict(X_test)
        print(metric+":"+str(metric_func_dict[metric](y_pred, y_test)))


if __name__ == "__main__":
    import tensorflow as tf
    # 加载训练集
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    np.random.seed(233)
    np.random.shuffle(X_train)
    np.random.seed(233)
    np.random.shuffle(y_train)

    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = X_train.reshape(
        len(X_train), -1), X_test.reshape(len(X_test), -1)
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # 创建单隐藏层神经网络
    model = MySequential(
        [MyDenseLayer(in_features=784, out_features=100, activation='sigmoid', bias=True),
         MyDenseLayer(in_features=100, out_features=100,
                      activation='sigmoid', bias=False),
         MyDenseLayer(in_features=100, out_features=10, activation='sigmoid', bias=True)])

    X_train, y_train = X_train[0:100, :], y_train[0:100]
    # learning rate 学习率有甜点，这个数据集这个方法大概在0.1-0.3之间效果好
    model.fit(X_train, y_train, epoch=10, batch_size=32,
              learning_rate=0.1, loss='mse')
