---
layout: post
title:  麦克船长 NLP 语言模型技术笔记 2：多层感知器（MLP）
date:   2023-01-19 03:44:09 +0800
categories: ai
tags: [NLP, 感知器, AI, 人工智能, 自然语言处理, 神经网络, 语言模型, 多层感知器, ANN, Perceptron, 激活后函数, 逻辑回归, 线性回归, Softmax, Sigmoid, ReLU, PyTorch]
description: 1957 年感知机（Perceptron）模型被提出，1959 年多层感知机（MLP）模型被提出。MLP 有时候也被称为 ANN，即 Artificial Neural Network，接下来我们来深入浅出地了解一下，并有一些动手的练习。
excerpt: 1957 年感知机（Perceptron）模型被提出，1959 年多层感知机（MLP）模型被提出。MLP 有时候也被称为 ANN，即 Artificial Neural Network，接下来我们来深入浅出地了解一下，并有一些动手的练习。
katex: True
location: 香港
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

本文关键词：感知器、线性回归、逻辑回归、激活函数、Sigmoid 函数/归一化/回归、Softmax 回归。

1957 年感知机（Perceptron）模型被提出，1959 年多层感知机（MLP）模型被提出。MLP 有时候也被称为 ANN，即 Artificial Neural Network，接下来我们来深入浅出地了解一下，并有一些动手的练习。

### 1、感知器（Perceptron）：解决二元分类任务的前馈神经网络

{% raw %}$$ x $${% endraw %} 是一个输入向量，{% raw %}$$ \omega $${% endraw %} 是一个权重向量（对输入向量里的而每个值分配一个权重值所组成的向量）。举一个具体任务例子，比如如果这两个响亮的内积超过某个值，则判断为 1，否则为 0，这其实就是一个分类任务。那么这个最终输出值可以如下表示：

{% raw %}
$$ y = \begin{cases} 1 & (\omega \cdot x \geq 0) \\ 0 & (\omega \cdot x \lt 0) \end{cases} $$
{% endraw %}

这就是一个典型的感知器（Perceptron，一般用来解决分类问题。还可以再增加一个偏差项（bias），如下：

{% raw %}
$$ y = \begin{cases} 1 & (\omega \cdot x + b \geq 0) \\ 0 & (\omega \cdot x + b \lt 0) \end{cases} $$
{% endraw %}

感知器其实就是一个前馈神经网络，由输入层、输出层组成，没有隐藏层。而且输出是一个二元函数，用于解决二元分类问题。

### 2、线性回归（Linear Regression）：从离散值的感知器（解决类问题），到连续值的线性回归（解决回归问题）

一般来说，我们认为感知器的输出结果，是离散值。一般来说，我们认为离散值作为输出解决的问题，是分类问题；相应地，连续值解决的问题是回归（Regression）。比如对于上面的感知器，如果我们直接将 {% raw %}$$ \omega \cdot x + b $${% endraw %} 作为输出值，则就变成了一个线性回归问题的模型了。

下面我们用 PyTorch 来实现一个线性回归的代码示例，首先我们要了解在 PyTorch 库里有一个非常常用的函数：

```python
nn.Linear(in_features, out_features)
```

这个函数在创建时会自动初始化权值和偏置，并且可以通过调用它的 `forward` 函数来计算输入数据的线性变换。具体来说，当输入为 `x` 时，`forward` 函数会计算 {% raw %}$$ y = \omega \cdot x + b $${% endraw %}，其中 {% raw %} $$ W $$ {% endraw %} 和 {% raw %} $$ b $$ {% endraw %} 分别是 `nn.Linear` 图层的权值和偏置。

我们来一个完整的代码示例：

```python
import torch
import torch.nn as nn

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# 初始化模型
model = LinearRegression(input_size=1, output_size=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 创建输入特征 X 和标签 y
X = torch.Tensor([[1], [2], [3], [4]])
y = torch.Tensor([[2], [4], [6], [8]])

# 训练模型
for epoch in range(100):
    # 前向传播
    predictions = model(X)
    loss = criterion(predictions, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 创建测试数据 X_test 和标签 y_test
X_test = torch.Tensor([[5], [6], [7], [8]])
y_test = torch.Tensor([[10], [12], [14], [16]])

# 测试模型
with torch.no_grad():
    predictions = model(X_test)
    loss = criterion(predictions, y_test)
    print(f'Test loss: {loss:.4f}')

```

上述代码，一开始先创建一个 LinearRegression 线性回归模型的类，其中有一个 `forward` 前向传播函数，调用时其实就是计算一下输出值 `y`。

主程序，一开始创建一个线性回归模型实例，然后定义一个用于评价模型效果的损失函数评价器，和用随机梯度下降（Stochastic Gradient Descent）作为优化器。

然后创建一个输入特征张量，和标签张量。用这组特征和标签进行训练，训练的过程就是根据 `X` 计算与测试 `predictions` 向量，再把它和 `y` 一起给评价器算出损失 `loss`，然后进行反向传播。注意反向传播的三行代码：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

如此训练 100 次（每一次都会黑盒化地更新模型的参数，一个 `epoch` 就是一次训练过程，有时也称为 `iteration` 或者 `step`，不断根据 `loss` 训练优化模型参数。

然后我们创建了一组测试特征值张量 `X_test`，和测试标签张量 `y_test`，然后用它们测试模型性能，把测试特征得到的 `predictions` 与 `y_test` 共同传给评价器，得到 `loss`。在这个例子中我们会得到如下结果：

```python
Test loss: 0.0034
```

### 3、逻辑回归（Logistic Regression）：没有值域约束的线性回归，到限定在一个范围内的逻辑回归（常用于分类问题）

可以看到线性回归问题，输出值是没有范围限定的。如果限定（limit）在特定的 {% raw %} $$ (0, L) $$ {% endraw %} 范围内，则就叫做逻辑回归了。那么如何将一个线性回归变成逻辑回归呢？一般通过如下公式变换：

{% raw %}
$$ y = \frac{L}{1 + e^{-k(z-z_0)}} $$
{% endraw %}

这样原来的 {% raw %} $$ z \in (-\infty, +\infty) $$ {% endraw %} 就被变换成了 {% raw %} $$ y \in (0, L) $$ {% endraw %} 了。

* **激活函数**：这种把输出值限定在一个目标范围内的函数，被叫做 **激活函数（Activation Function）**。
* **函数的陡峭程度** 由 {% raw %} $$ k $$ {% endraw %} 控制，越大越陡。
* 当 {% raw %} $$ z = z_0 $$ {% endraw %} 时，{% raw %} $$ y = \frac{L}{2} $$ {% endraw %}。

下面给出一个基于 Python 的 scikit-learn 库的示例代码：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 这是 scikit-learn 库里的一个简单的数据集
iris = load_iris()

# 把 iris 数据集拆分成训练集和测试集两部分
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)

# 用 scikit-learn 库创建一个逻辑回归模型的实例
lr = LogisticRegression()

# 用上边 split 出来的训练集数据，训练 lr 模型实例
lr.fit(X_train, y_train)

# 用训练过的模型，拿测试集的输入数据做测试
predictions = lr.predict(X_test)

# 用测试集的数据验证精确性
accuracy = lr.score(X_test, predictions)
print(accuracy)
```

### 4、Sigmoid 回归（Sigmoid Regression）：归一化的逻辑回归，一般用于二元分类任务

当 {% raw %} $$ L = 1, k = 1, z_0 = 0 $$ {% endraw %}，此时的激活函数就是 **Sigmoid** 函数，也常表示为 {% raw %} $$ \sigma $$ {% endraw %} 函数，如下：

{% raw %} $$ y = \frac{1}{1 + e^{-z}} $$ {% endraw %}

Sigmoid 回归的值域，恰好在 (0, 1) 之间，所以常备作为用来归一化的激活函数。而一个线性回归模型，再用 sigmoid 函数归一化，这种也常被称为「Sigmoid 回归」。Sigmoid 这个单词的意思也就是 S 形，我们可以看下它的函数图像如下：

![image](/img/src/2022-12-19-language-model-2.png)

因为归一化，所以也可以把输出值理解为一个概率。比如我们面对一个二元分类问题，那么输出结果就对应属于这个类别的概率。

这样一个 sigmoid 模型可以表示为：

{% raw %}
$$ \bold{y} = Sigmoid(\bold{W} \cdot \bold{x} + \bold{b}) $$
{% endraw %}

另外 sigmoid 函数的导数（即梯度）是很好算的：{% raw %} $$ y' = y \cdot (1-y) $$ {% endraw %}。这非常方便用于「梯度下降算法」根据 loss 对模型参数进行优化。Sigmoid 回归，一般用于二元分类任务。那么对于超过二元的情况怎么办呢？这就引出了下面的 Softmax 回归。

### 5、Softmax 回归（Softmax Regression）：从解决二元任务的 sigmoid，到解决多元分类任务的 Softmax

相对逻辑回归，Softmax 也称为多项逻辑回归。上面说 Sigmoid 一般用于解决二元分类问题，那么多元问题就要用 Softmax 回归了。我们来拿一个具体问题来解释，比如问题是对于任意输入的一个电商商品的图片，来判断这个图片所代表的的商品，属于哪个商品类目。假设我们一共有 100 个类目。那么一个图片比如说其所有像素值作为输入特征值，输出就是一个 100 维的向量 **{% raw %} $$ z $$ {% endraw %}**，输出向量中的每个值 {% raw %} $$ z_i $$ {% endraw %} 表示属于相对应类目的概率 {% raw %} $$ y_i $$ {% endraw %} ：

{% raw %}
$$ y_i = Softmax(\bold{z})_i = \frac{e^{z_i}}{e^{z_1} + e^{z_2} + ... + e^{z_100}} $$
{% endraw %}

那么最后得到的 {% raw %} $$ y $$ {% endraw %} 向量中的每一项就对应这个输入 {% raw %} $$ z $$ {% endraw %} 属于这 100 个类目的各自概率了。所以如果回归到一般问题，这个 Softmax 回归的模型就如下：

{% raw %}
$$ \bold{y} = Softmax(\bold{W} \cdot \bold{x} + \bold{b}) $$
{% endraw %}

对于上面电商商品图片的例子，假设每个图片的尺寸是 512x512，这个模型展开式如下：

{% raw %}
$$ \begin{bmatrix} y_1 \\ y_2 \\ ... \\ y_{100} \end{bmatrix} = Softmax(\begin{bmatrix} w_{1,1}, & w_{1,2}, & ... & w_{1, 512} \\ w_{2,1}, & w_{2,2}, & ... & w_{2, 512} \\ ... & ... & ... & ... \\ w_{100,1}, & w_{100,2}, & ... & w_{100, 512} \end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \\ ... \\ x_{512} \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ ... \\ b_{512} \end{bmatrix}) $$
{% endraw %}

这个对输入向量 {% raw %} $$ x $$ {% endraw %} 执行 {% raw %} $$ w \cdot x + b $$ {% endraw %} 运算，一般也常称为「线性映射/线性变化」。

### 6、多层感知器（Multi-Layer Perceptron）

上面我们遇到的所有任务，都是用线性模型（Linear Models）解决的。有时候问题复杂起来，我们就要引入非线性模型了。

这里我们要介绍一个新的激活函数 —— ReLU（Rectified Linear Unit）—— 一个非线性激活函数，其定义如下：

{% raw %}
$$ ReLU(\bold{z}) = max(0, \bold{z}) $$
{% endraw %}

比如对于 MNIST 数据集的手写数字分类问题，就是一个典型的非线性的分类任务，下面给出一个示例代码：

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 定义多层感知器模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 超参数
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='../../data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='../../data',
                              train=False,
                              transform=transforms.ToTensor())

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = MLP(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 输出训练损失
    print(f'Epoch {epoch + 1}, Training Loss: {loss.item():.4f}')
```

这段代码里，我们能看到 MLP 的模型定义是：

```python
nn.Linear(input_size, hidden_size)
nn.ReLU()
nn.Linear(hidden_size, num_classes)
```

与前面的模型示例代码类似，也都用到了反向传播、损失函数评价器、优化器。如果用公式表示的话，就是如下的模型定义：

{% raw %}
$$
\begin{aligned}
&\bold{z} = \bold{W}_1 \cdot \bold{x} + \bold{b}_1 \\
&\bold{h} = ReLU(\bold{z}) \\
&\bold{y} = \bold{W}_2 \cdot \bold{h} + \bold{b}_2
\end{aligned}
$$
{% endraw %}

我们知道 MLP 通常是一个输入和输出长度相同的模型，但少数情况下也可以构建输入和输出长度不同的 MLP 模型，比如输入一组序列后，输出是一个离散的分类结果。

### 7、MLP 的一个显著问题，帮我们引出 CNN 模型

我们可以看到，在 MLP 中，不论有多少层，某一层的输出向量 {% raw %} $$ h_n $$ {% endraw %} 中的每个值，都会在下一层计算输出向量 {% raw %} $$ h_{n+1} $$ {% endraw %} 的每个值时用到。具体来说，如果对于某一层的输出值如下：

{% raw %}
$$ \bold{h}_{n+1} = Softmax(\bold{W}_{n+1} \cdot \bold{h}_n + \bold{b}_{n+1}) $$
{% endraw %}

上一段话里所谓的「用到」，其实就是要针对 {% raw %} $$ h_n $$ {% endraw %} 生成相应的特征值 {% raw %} $$ W_{n+1} $$ {% endraw %} 权重矩阵中的每个行列里的数值和 {% raw %} $$ b_{n+1} $$ 偏差向量{% endraw %} 里的每个值。如果用图画出来，就是：

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=TB
	a[label="..."]
	b[label="..."]
	h_2_1[label="h_n+1_1"]
	h_2_2[label="h_n+1_2"]
	h_2_m[label="h_n+1_m"]

	{rank=same h_n_1 h_n_2 b h_n_m}
	{rank=same h_2_1 h_2_2 a h_2_m}

	h_n_1 -> h_2_1
	h_n_1 -> h_2_2
	h_n_1 -> a
	h_n_1 -> h_2_m

	h_n_1 -> h_2_1
	h_n_2 -> h_2_2
	h_n_2 -> a
	h_n_2 -> h_2_m

	b -> h_2_1
	b -> h_2_2
	b -> a
	b -> h_2_m

	h_n_m -> h_2_1
	h_n_m -> h_2_2
	h_n_m -> a
	h_n_m -> h_2_m
}
{% endgraphviz %}
</div>

可以看到，输入的所有元素都被连接，即被分配权重 w 和偏差项 b，所以这被称为一个「全连接层（**Fully Connected Layer**）」或者「**稠密层（Dense Layer）**」。但是对于一些任务这样做是很蠢的，会付出大量无效的计算。

因此我们需要 focus 在更少量计算成本的模型，于是有了卷积神经网络（CNN）。关于 CNN 请看本系列博客的第「3」篇。

### Reference

* 《自然语言处理：基于预训练模型的方法》车万翔 等
* 《自然语言处理实战：预训练模型应用及其产品化》安库·A·帕特尔 等