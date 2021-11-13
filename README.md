如图所示，为简化描述，我们用一个三元组$(in_i, out_i, bias_i)$来描述第$i$层layer，分表表示`in_feature`、`out_feature`、`bias`，使用$X_i$表示每层的输入`inputs[i]`，$Y_i$表示每层的输出`outputs[i]`。

## 前向传播

设样本数为$n$，则开始有$shape$为$(n, in_0)$的网络输入$X_{input}$，则有：
$$
Y_0 = \text{activation}(W_0 X_0)
$$

$$
\begin{cases}
X_0.shape = (n, in_0+1), W_0.shape=(in_0+1, out_0) & bias_0 = true \\
X_0.shape = (n,in_0), W_0.shape = (in_0, out_0)& bias_0 = false
\end{cases}
$$

$$
Y_0.shape = (n, out_0)
$$

显然后面的层也是一样的：
$$
X_i  = Y_{i-1}\\
Y_i = \text{activation}({W_i X_i})
$$

$$
\begin{cases}
X_i.shape = (n, in_i+1), W_i.shape=(in_i+1, out_i) & bias_i = true \\
X_i.shape = (n,in_i), W_i.shape = (in_i, out_i)& bias_i = false
\end{cases}
$$

$$
Y_i.shape = (n, out_i)
$$



