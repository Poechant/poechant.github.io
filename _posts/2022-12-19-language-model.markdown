---
layout: post
title:  自然语言处理的基本语言模型（Language Models）原理
date:   2022-12-19 23:58:11 +0800
categories: ai
tags: [AI, 人工智能, n-gram, 马尔科夫, 神经网络, GPT, BERT, 语言模型, LM, 深度学习, deep learning, n-gram, Attention, Transformer, CNN, RNN, 卷积神经网络, 循环神经网络, MLP, Perceptron, 感知器, 激活函数, softmax, linear regression, logistic regression, 线性回归, 逻辑回归, 分类问题, 回归问题, ANN, 及其学习, ML, Sigmoid]
description: 
excerpt: 
katex: True
location: 杭州
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

文章进度：

1. N 元文法
* 语言模型开头
* N 元文法语言模型
* 平滑技术
* 折扣法
* 加 1 平滑
* δ 平滑

2. MLP
* 【完成】神经网络开头
* 【完成】MLP 开头
* 【完成】感知器
* 【完成】线性回归
* 【完成】逻辑回归
* 【完成】softmax 回归
* 【完成】多层感知器

3. CNN

4. RNN
* 【完成】RNN 开头
* 【完成】RNN 经典结构 nvsn
* 【完成】RNN nvs1
* 【完成】RNN 1vsn
* 【完成】RNN nvsm
* 【完成】LSTM
* 【完成】BiRNN、BiLSTM、stacked RNN、stacked LSTM
* 【完成】RNN nm

5. Attention

6. Transformer

### 一、语言模型

### 二、N 元文法语言模型（N-gram Language Model）

下一次词出现的概率只依赖于它前面 n-1 个词，这种假设被称为「马尔科夫假设（Markov Assumption」。N 元文法，也称为 N-1 阶马尔科夫链。

* 一元文法（1-gram），unigram，零阶马尔科夫链，不依赖前面任何词；
* 二元文法（2-gram），bigram，一阶马尔科夫链，只依赖于前 1 个词；
* 三元文法（3-gram），trigram，二阶马尔科夫链，只依赖于前 2 个词；
* ……

通过前 t-1 个词预测时刻 t 出现某词的概率，用最大似然估计：

P(Wt | W1,W2...Wt-1) = C(W1,W2,...Wt) / C(W1,W2,...Wt-1)

进一步地，一组词（也就是一个句子）出现的概率就是：

P(W1,W2,...Wt) = P(Wt | W1,W2,...Wt-1)
			   * P(Wt-1 | W1,W2,...Wt-2)
			   * P(Wt-2 | W1,W2,...Wt-3)
			   * ...
			   * P(W1)

#### 平滑技术

##### 折扣法

##### 加 1 平滑

##### δ 平滑

### 三、神经网络语言模型（Neural Network Langauge Model）

N 元文法的缺陷是非常明显的：

* 模型容易受到数据稀疏的影响，一般需要对模型进行平滑处理；
* 无法对长度超过 N 的上下文依赖关系进行建模。

神经网络很好地解决了这两个问题：

* 引入了词向量，解决了数据稀疏的影响；
* 更先进的模型结构，可以对长距离上下文依赖进行有效的建模。

#### 1、MLP（Multi-Layer Perceptron）

关键词：感知器、线性回归、逻辑回归、激活函数、Sigmoid 函数/归一化/回归、Softmax 回归

1957 年感知机（Perceptron）模型被提出，1959 年多层感知机（MLP）模型被提出。MLP 有时候也被称为 ANN，即 Artificial Neural Network，接下来我们来深入浅出地了解一下，并有一些动手的练习。

##### 1.1、感知器（Perceptron）：解决二元分类任务的前馈神经网络

{% raw %}$$ x $${% endraw %} 是一个输入向量，{% raw %}$$ \omega $${% endraw %} 是一个权重向量（对输入向量里的而每个值分配一个权重值所组成的向量）。举一个具体任务例子，比如如果这两个响亮的内积超过某个值，则判断为 1，否则为 0，这其实就是一个分类任务。那么这个最终输出值可以如下表示：

{% raw %}
$$ y = \begin{cases} 1 & (\omega \cdot x \geq 0) \\ 0 & (\omega \cdot x \lt 0) \end{cases} $$
{% endraw %}

这就是一个典型的感知器（Perceptron，一般用来解决分类问题。还可以再增加一个偏差项（bias），如下：

{% raw %}
$$ y = \begin{cases} 1 & (\omega \cdot x + b \geq 0) \\ 0 & (\omega \cdot x + b \lt 0) \end{cases} $$
{% endraw %}

感知器其实就是一个前馈神经网络，由输入层、输出层组成，没有隐藏层。而且输出是一个二元函数，用于解决二元分类问题。

##### 1.2、线性回归（Linear Regression）：从离散值的感知器（分类问题），到连续值的线性回归（回归问题）

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

##### 1.3、逻辑回归（Logistic Regression）：没有值域约束的线性回归，到限定在一个范围内的逻辑回归（常用于分类问题）

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

// 这是 scikit-learn 库里的一个简单的数据集
iris = load_iris()

// 把 iris 数据集拆分成训练集和测试集两部分
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)

// 用 scikit-learn 库创建一个逻辑回归模型的实例
lr = LogisticRegression()

// 用上边 split 出来的训练集数据，训练 lr 模型实例
lr.fit(X_train, y_train)

// 用训练过的模型，拿测试集的输入数据做测试
predictions = lr.predict(X_test)

// 用测试集的数据验证精确性
accuracy = lr.score(X_test, predictions)
print(accuracy)
```

##### 1.4、Sigmoid 回归（Sigmoid Regression）：归一化的逻辑回归，一般用于二元分类任务

当 {% raw %} $$ L = 1, k = 1, z_0 = 0 $$ {% endraw %}，此时的激活函数就是 **Sigmoid** 函数，如下：

{% raw %} $$ y = \frac{1}{1 + e^{-z}} $$ {% endraw %}

Sigmoid 回归的值域，恰好在 (0, 1) 之间，所以常备作为用来归一化的激活函数。而一个线性回归模型，再用 sigmoid 函数归一化，这种也常被称为「Sigmoid 回归」。Sigmoid 这个单词的意思也就是 S 形，我们可以看下它的函数图像如下：

![image](/img/src/2022-12-19-language-model-2.png)

因为归一化，所以也可以把输出值理解为一个概率。比如我们面对一个二元分类问题，那么输出结果就对应属于这个类别的概率。

这样一个 sigmoid 模型可以表示为：

{% raw %}
$$ \bold{y} = Sigmoid(\bold{W} \cdot \bold{x} + \bold{b}) $$
{% endraw %}

另外 sigmoid 函数的导数（即梯度）是很好算的：{% raw %} $$ y' = y \cdot (1-y) $$ {% endraw %}。这非常方便用于「梯度下降算法」根据 loss 对模型参数进行优化。Sigmoid 回归，一般用于二元分类任务。那么对于超过二元的情况怎么办呢？这就引出了下面的 Softmax 回归。

##### 1.5、Softmax 回归（Softmax Regression）：从解决二元任务的 sigmoid，到解决多元分类任务的 softmax

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

##### 1.6、多层感知器（Multi-Layer Perceptron）

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

##### MLP 的一个显著问题，帮我们引出 CNN 模型

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

可以看到，输入的所有元素都被连接，即被分配权重 w 和偏差项 b，所以这被称为一个「全连接层（**Fully Connected Layer**）」或者「**稠密层（Dense Layer）**」。但是对于一些任务这样做是很蠢的，会付出大量无效的计算。这样的网络，也被叫做「全连接神经网络」，或者「前馈神经网络（Feed Forward neural Network，简称 FFN）」

因此我们需要 focus 在更少量计算成本的模型，于是有了卷积神经网络（CNN）。

#### 2、CNN（Convolutional Neural Network）

卷积神经网络（Convolutional Neural Network，简称 CNN）于 1989 年在论文《Backpropagation Applied to Handwritten Zip Code Recognition》中被提出。

#### 3、RNN（Recurrent Neural Network）

RNN 的 R 是 Recurrent 的意思，所以这是一个贷循环的神经网络。首先要明白一点，你并不需要搞懂 CNN 后才能学习 RNN 模型。你只要了解了 MLP 就可以学习 RNN 了。

##### 3.1、经典结构的 RNN

![image](/img/src/2022-12-19-language-model-1.png)

上图这是一个经典结构的 RNN 示意图，Unfold 箭头右侧是展开示意。输入序列（这里用 x 表示）传递给隐藏层（hidden layer，这里用 h 表示），处理完生成输出序列（这里用 o 表示）。序列的下一个词输入时的、上一步隐藏层会一起影响这一步的输出。U、V、W 都表示权重。在这个经典结构理，你可以看到非常重要的一点，就是输入序列长度与输出序列长度是相同的。

这种经典结构的应用场景，比如对一段普通话输入它的四川话版本，比如对视频的每一帧进行处理并输出，等等。

我们知道 RNN 是一个一个序列处理的，每个序列中的数据项都是有序的，所以对于计算一个序列内的所有数据项是无法并行的。但是计算不同序列时，不同序列各自的计算则是可以并行的。如果我们把上一个时刻 t 隐藏层输出的结果（{% raw %} $$ h_{t-1} $$ {% endraw %}）传给一个激活函数（比如说用正切函数 `tanh` 函数），然后和当下时刻 t 的这个输入（{% raw %} $$ x_{t} $$ {% endraw %}）一起，处理后产生一个时刻 t 的输出（{% raw %} $$ h_t $$ {% endraw %}）。然后把隐藏层的输出通过多项逻辑回归（Softmax）生成最终的输出值（{% raw %} $$ \bm{y} $$ {% endraw %}），我们可以如下表示这个模型：

{% raw %}
$$
\begin{aligned}
&\bm{h}_t = tanh(\bm{W}^{xh} \cdot \bm{x}_t + \bm{b}^{xh} + \bm{W}^{hh} \cdot \bm{h}_{t-1} + \bm{b}^{hh}) \\
&\bm{y}_t = Softmax(\bm{W}^{hy} \cdot \bm{h_t} + \bm{b}^{hy})
\end{aligned}
$$
{% endraw %}

对应的示意图如下：

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	{rank=same h1 h2 hddd hn}
	{rank=same x1 x2 xddd xn}
	{rank=same y1 y2 yddd yn}
	xddd[label="..."]
	yddd[label="..."]
	hddd[label="..."]

	y1[shape=plaintext]
	y2[shape=plaintext]
	yddd[shape=plaintext]
	yn[shape=plaintext]
	x1[shape=plaintext]
	x2[shape=plaintext]
	xddd[shape=plaintext]
	xn[shape=plaintext]

	h1 -> h2
	h2 -> hddd
	hddd -> hn

	x1 -> h1
	x2 -> h2
	xddd -> hddd
	xn -> hn

	h1 -> y1
	h2 -> y2
	hddd -> yddd
	hn -> yn
}
{% endgraphviz %}
</div>

这种输入和输出数据项数一致的 RNN，一般叫做 N vs. N 的 RNN。

##### 3.2、N vs.1 的 RNN

上面那个图里，如果只保留最后一个输出，那就是一个 N vs. 1 的 RNN 了。这种的应用场景，比如说判断一个文本序列是英语还是德语，比如根据一个输入序列来判断是一个正向情绪内容还是负向或者中性，或者比如根据一段语音输入序列来判断是哪一首曲子（听歌识曲）。

{% raw %}
$$
\begin{aligned}
&\bm{h}_t = tanh(\bm{W^{xh}} \cdot \bm{x}_t + \bm{b^{xh}} + \bm{W^{hh}} \cdot \bm{h}_{t-1} + \bm{b^{hh}}) \\
&\bm{y} = Softmax(\bm{W^{hy}} \cdot \bm{h}_n + \bm{b^{hy}})
\end{aligned}
$$
{% endraw %}

即这个模型里，每个序列只有隐藏层对最后一个数据项进行处理时才产生输出 {% raw %} $$ h_n $$ {% endraw %} 如果用示意图表示，则是如下结构：

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=LR
	{rank=same h1 h2 hddd hn}
	hddd[label="..."]
	xddd[label="..."]

	y[shape=plaintext]
	x1[shape=plaintext]
	x2[shape=plaintext]
	xddd[shape=plaintext]
	xn[shape=plaintext]

	h1 -> h2
	h2 -> hddd
	hddd -> hn

	x1 -> h1
	x2 -> h2
	xn -> hn
	xddd -> hddd

	hn -> y
}
{% endgraphviz %}
</div>

##### 3.3、1 vs. N 的 RNN

反过来，上面那个图里，如果只保留一个 x，那么就是一个 1 vs. N 的 RNN 了。这种场景的应用，比如 AI 创作音乐，还有通过一个 image 提炼或识别某些文本内容输出。

{% raw %}
$$
\begin{aligned}
&\bm{h}_t = \begin{cases} tanh(\bm{W^{xh}} \cdot \bm{x} + \bm{b^{xh}} + 0 + \bm{b^{hh}}) & (t=1) \\
tanh(0 + \bm{b^{xh}} + \bm{W^{hh}} \cdot \bm{h}_{t-1} + \bm{b^{hh}}) & (t>1) \end{cases} \\
&\bm{y} = Softmax(\bm{W^{hy}} \cdot \bm{h}_n + \bm{b^{hy}})
\end{aligned}
$$
{% endraw %}

示意图如下：

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	{rank=same h1 h2 hddd hn}
	{rank=same y1 y2 yddd yn}
	hddd[label="..."]
	yddd[label="..."]

	y1[shape=plaintext]
	y2[shape=plaintext]
	yddd[shape=plaintext]
	yn[shape=plaintext]
	x[shape=plaintext]

	h1 -> h2
	h2 -> hddd
	hddd -> hn

	x -> h1

	h1 -> y1
	h2 -> y2
	hddd -> yddd
	hn -> yn
}
{% endgraphviz %}
</div>

到这里我们可以看到，在 RNN 的隐藏层是能够存储一些有关于输入数据的一些相关内容的，所以也常把 RNN 的隐藏层叫做记忆单元。

##### 3.4、LSTM（Long Short-Term Memory）长短时记忆网络

###### 3.4.1、如何理解这个 Short-Term 呢？

1997 年论文《Long Short-Term Memory》中提出 LSTM 模型。我们先从模型的定义，精确地来理解一下：

{% raw %}
$$
\begin{aligned}
&\bm{h}_t = \bm{h}_{t-1} + tanh(\bm{W}^{xh} \cdot \bm{x}_t + \bm{b}^{xh} + \bm{W}^{hh} \cdot \bm{h}_{t-1} + \bm{b}^{hh}) \\
&\bm{y}_t = Softmax(\bm{W}^{hy} \cdot \bm{h_t} + \bm{b}^{hy})
\end{aligned}
$$
{% endraw %}

上式中与经典结构的 RNN（输入与输出是 N vs. N）相比，唯一的区别是第一个式子中多了一个「{% raw %} $$ \bm{h}_{t-1} $$ {% endraw %}」。如果我们把第一个式子的 {% raw %} $$ tanh $$ {% endraw %} 部分记作 {% raw %} $$ u_t $$ {% endraw %}：

{% raw %}
$$
\bm{u}_t = tanh(\bm{W}^{xh} \cdot \bm{x}_t + \bm{b}^{xh} + \bm{W}^{hh} \cdot \bm{h}_{t-1} + \bm{b}^{hh}) $$
{% endraw %}

所以：

{% raw %}
$$
\bm{h}_t = \bm{h}_{t-1} + \bm{u}_t
$$
{% endraw %}

那么可以展开出如下一组式子：

{% raw %}
$$
\begin{aligned}
\bm{h}_{k+1} &= \bm{h}_k + \bm{u}_{k+1} \\
\bm{h}_{k+2} &= \bm{h}_{k+1} + \bm{u}_{k+2} \\
&...... \\
\bm{h}_{t-1} &= \bm{h}_{t-2} + \bm{u}_{t-1} \\
\bm{h}_t &= \bm{h}_{t-1} + \bm{u}_t
\end{aligned}
$$
{% endraw %}

如果我们从 {% raw %} $$ h_{k+1} $$ {% endraw %} 到 {% raw %} $$ h_n $$ {% endraw %} 的所有式子左侧相加、右侧相加，我们就得到如下式子：

{% raw %}
$$
\begin{aligned}
&\bm{h}_{k+1} + ... + \bm{h}_{t-1} + \bm{h}_t \\
= &\bm{h}_k + \bm{h}_{k+1} + ... + \bm{h}_{t-2} + \bm{h}_{t-1} \\+ &\bm{u}_{k+1} + \bm{u}_{k+2} + ... + \bm{u}_{t-1} + \bm{u}_t
\end{aligned}
$$
{% endraw %}

进而推导出：

{% raw %}
$$
\bm{h}_t = \bm{h}_k + \bm{u}_{k+1} + \bm{u}_{k+2} + ... + \bm{u}_{t-1} + \bm{u}_t
$$
{% endraw %}

从这里我们就可以看到，第 t 时刻的隐藏层输出，直接关联到第 k 时刻的输出，t 到 k 时刻的相关性则用 {% raw %} $$ \bm{u}_{k+1} $$ {% endraw %} 到 {% raw %} $$ \bm{u}_t $$ {% endraw %} 相加表示。也就是有 t-k 的短期（Short Term）记忆。

###### 3.4.2、引入遗忘门 f、输入门 i、输出门 o、记忆细胞 c

如果我们为式子 {% raw %} $$ \bm{h}_t = \bm{h}_{t-1} + \bm{u}_t $$ {% endraw %} 右侧两项分配一个权重呢？就是隐藏层对上一个数据项本身被上一个数据项经过隐藏层计算的结果，这两者做一对权重考虑配比，如下：

{% raw %}
$$
\begin{aligned}
&\bm{f}_t = sigmoid(\bm{W}^{f,xh} \cdot \bm{x}_t + \bm{b}^{f,xh} + \bm{W}^{f,hh} \cdot \bm{x}_{t-1} + \bm{b}^{f,hh}) \\
&\bm{h}_t = \bm{f}_t \odot \bm{h}_{t-1} + (1 - \bm{f}_t) \odot \bm{u}_t
\end{aligned}
$$
{% endraw %}

其中：

* {% raw %} $$ \odot $$ {% endraw %} 是 Hardamard 乘积，即张量的对应元素相乘。
* {% raw %} $$ \bm{f}_t $$ {% endraw %} 是「遗忘门（Forget Gate）」，该值很小时 t-1 时刻的权重就很小，也就是「此刻遗忘上一刻」。该值应根据 t 时刻的输入数据、t-1 时刻数据在隐藏层的输出计算，而且其每个元素必须是 (0, 1) 之间的值，所以可以用 sigmoid 函数来得到该值：

但这种方式，对于过去 {% raw %} $$ \bm{h}_{t-1} $$ {% endraw %} 和当下 {% raw %} $$ \bm{u}_t $$ {% endraw %} 形成了互斥，只能此消彼长。但其实过去和当下可能都很重要，有可能都恨不重要，所以我们对过去继续采用 {% raw %} $$ \bm{f}_t $$ {% endraw %} 遗忘门，对当下采用 {% raw %} $$ \bm{i}_t $$ {% endraw %} 输入门（Input Gate）：

{% raw %}
$$
\begin{aligned}
&\bm{f}_t = sigmoid(\bm{W}^{f,xh} \cdot \bm{x}_t + \bm{b}^{f,xh} + \bm{W}^{f,hh} \cdot \bm{x}_{t-1} + \bm{b}^{f,hh}) \\
&\bm{i}_t = sigmoid(\bm{W}^{i,xh} \cdot \bm{x}_t + \bm{b}^{i,xh} + \bm{W}^{i,hh} \cdot \bm{h}_{t-1} + \bm{b}^{i,hh}) \\
&\bm{h}_t = \bm{f}_t \odot \bm{h}_{t-1} + \bm{i}_t \odot \bm{u}_t
\end{aligned}
$$
{% endraw %}

其中：
* 与 {% raw %} $$ \bm{f}_t $$ {% endraw %} 类似地，定义输入门 {% raw %} $$ \bm{i}_t $$ {% endraw %} ，但是注意 {% raw %} $$ \bm{f}_t $$ {% endraw %} 与 {% raw %} $$ \bm{h}_{t-1} $$ {% endraw %} 而非 {% raw %} $$ \bm{x}_{t-1} $$ {% endraw %} 有关。

再引入一个输出门：

{% raw %}
$$
\bm{o}_t = sigmoid(\bm{W}^{o,xh} \cdot \bm{x}_t + \bm{b}^{o,xh} + \bm{W}^{o,hh} \cdot \bm{x}_{t-1} + \bm{b}^{o,hh})
$$
{% endraw %}

再引入记忆细胞 {% raw %} $$ \bm{c}_t $$ {% endraw %}，它是原来 {% raw %} $$ \bm{h}_t $$ {% endraw %} 的变体，与 t-1 时刻的记忆细胞有遗忘关系（通过遗忘门），与当下时刻有输入门的关系：

{% raw %}
$$
\bm{c}_t = \bm{f}_t \odot \bm{c}_{t-1} + \bm{i}_t \odot \bm{u}_t
$$
{% endraw %}

那么此时 {% raw %} $$ \bm{h}_t $$ {% endraw %} ，我们可以把 {% raw %} $$ \bm{h}_t $$ {% endraw %} 变成：

{% raw %}
$$
\bm{h}_t = \bm{o}_t \odot tanh(\bm{c}_t)
$$
{% endraw %}

记忆细胞这个概念还有有一点点形象的，它存储了过去的一些信息。OK，到此我们整体的 LSTM 模型就变成了这个样子：

{% raw %}
$$
\begin{aligned}
&\bm{f}_t = sigmoid(\bm{W}^{f,xh} \cdot \bm{x}_t + \bm{b}^{f,xh} + \bm{W}^{f,hh} \cdot \bm{x}_{t-1} + \bm{b}^{f,hh}) \\
&\bm{i}_t = sigmoid(\bm{W}^{i,xh} \cdot \bm{x}_t + \bm{b}^{i,xh} + \bm{W}^{i,hh} \cdot \bm{h}_{t-1} + \bm{b}^{i,hh}) \\
&\bm{o}_t = sigmoid(\bm{W}^{o,xh} \cdot \bm{x}_t + \bm{b}^{o,xh} + \bm{W}^{o,hh} \cdot \bm{x}_{t-1} + \bm{b}^{o,hh}) \\
&\bm{u}_t = tanh(\bm{W}^{xh} \cdot \bm{x}_t + \bm{b}^{xh} + \bm{W}^{hh} \cdot \bm{h}_{t-1} + \bm{b}^{hh}) \\
&\bm{c}_t = \bm{f}_t \odot \bm{c}_{t-1} + \bm{i}_t \odot \bm{u}_t \\
&\bm{h}_t = \bm{o}_t \odot tanh(\bm{c}_t) \\
&\bm{y}_t = Softmax(\bm{W}^{hy} \cdot \bm{h_t} + \bm{b}^{hy})
\end{aligned}
$$
{% endraw %}

##### 3.5、双向循环神经网络、双向 LSTM

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	splines=ortho
	{rank=same h1 h2 hddd hn}

	hddd[label="..."]
	xddd[label="..."]
	yddd[label="..."]

	y1[shape=plaintext]
	y2[shape=plaintext]
	yddd[shape=plaintext]
	yn[shape=plaintext]
	x1[shape=plaintext]
	x2[shape=plaintext]
	xddd[shape=plaintext]
	xn[shape=plaintext]

	h1 -> y1
	h2 -> y2
	hddd -> yddd
	hn -> yn

	h1 -> h2
	h2 -> hddd
	hddd -> hn

	hn -> hddd
	hddd -> h2
	h2 -> h1

	x1 -> h1
	x2 -> h2
	xddd -> hddd
	xn -> hn
}
{% endgraphviz %}
</div>

##### 3.6、堆叠循环神经网络、堆叠 LSTM

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	{rank=same h11 h12 h1ddd h1n}
	{rank=same h21 h22 h2ddd h2n}

	h1ddd[label="..."]
	h2ddd[label="..."]
	xddd[label="..."]
	yddd[label="..."]

	y1[shape=plaintext]
	y2[shape=plaintext]
	yddd[shape=plaintext]
	yn[shape=plaintext]
	x1[shape=plaintext]
	x2[shape=plaintext]
	xddd[shape=plaintext]
	xn[shape=plaintext]

	h11 -> y1
	h12 -> y2
	h1ddd -> yddd
	h1n -> yn

	h11 -> h12
	h12 -> h1ddd
	h1ddd -> h1n

	h21 -> h22
	h22 -> h2ddd
	h2ddd -> h2n

	h21 -> h11
	h22 -> h12
	h2ddd -> h1ddd
	h2n -> h1n

	x1 -> h21
	x2 -> h22
	xddd -> h2ddd
	xn -> h2n
}
{% endgraphviz %}
</div>

##### 3.7、N vs. M 的 RNN

对于输入序列长度（长度 N）和输出序列长度（长度 M）不一样的 RNN 模型结构，也可以叫做 Encoder-Decoder 模型，也可以叫 Seq2Seq 模型。首先接收输入序列的 Encoder 先将输入序列转成一个隐藏态的上下文表示 C。C 可以只与最后一个隐藏层有关，甚至可以是最后一个隐藏层生成的隐藏态直接设置为 C，C 还可以与所有隐藏层有关。

有了这个 C 之后，再用 Decoder 进行解码，也就是从把 C 作为输入状态开始，生成输出序列。

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	{rank=same e1 e2 eddd en C d1 d2 dddd dm}

	eddd[label="..."]
	dddd[label="..."]
	xddd[label="..."]
	yddd[label="..."]
	C[shape=plaintext]
	x1[shape=plaintext]
	x2[shape=plaintext]
	xddd[shape=plaintext]
	xn[shape=plaintext]
	y1[shape=plaintext]
	y2[shape=plaintext]
	yddd[shape=plaintext]
	yn[shape=plaintext]

	x1 -> e1
	x2 -> e2
	xddd -> eddd
	xn -> en

	e1 -> e2
	e2 -> eddd
	eddd -> en

	en -> C
	C -> d1

	d1 -> y1
	d2 -> y2
	dddd -> yddd
	dm -> yn

	d1 -> d2
	d2 -> dddd
	dddd -> dm
}
{% endgraphviz %}
</div>

这种的应用就非常广了，因为大多数时候输入序列与输出序列的长度都是不同的，比如最常见的应用「翻译」，从一个语言翻译成另一个语言；再比如 AI 的一个领域「语音识别」，将语音序列输入后生成所识别的文本内容；还有比如 ChatGPT 这种问答应用等等。

但是 Seq2Seq 模型有一个很显著的问题，就是当输入序列很长时，Encoder 生成的 Context 可能就会出现所捕捉的信息不充分的情况，导致 Decoder 最终的输出是不尽如人意的。

#### 4、Attention 机制

Encoder-Decoder 的一个非常严重的问题，是依赖中间那个 context 向量，则无法处理特别长的输入序列 —— 记忆力不足，会忘事儿。而忘事儿的根本原因，是没有「注意力」。

##### 为什么说 RNN 模型没有体现「注意力」？

对于一般的 RNN 模型，Encoder-Decoder 结构并没有体现「注意力」—— 这句话怎么理解？当输入序列经过 Encoder 生成的中间结果（上下文 C），被喂给 Decoder 时，这些中间结果对所生成序列里的哪个词，都没有区别（没有特别关照谁）。这相当于在说：输入序列里的每个词，对于生成任何一个输出的词的影响，是一样的，而不是输出某个词时是聚焦特定的一些输入词。这就是模型没有注意力机制。

人脑的注意力模型，其实是资源分配模型。NLP 领域的注意力模型，是在 2014 年被提出的，后来逐渐成为 NLP 领域的一个广泛应用的机制。

所以 Attention 机制，就是在 Decoder 时，不是所有输出都依赖相同的「上下文 {% raw %} $$ \bm{C}_t $$ {% endraw %}」，而是时刻 t 的输出，使用 {% raw %} $$ C_t $$ {% endraw %}，而这个 {% raw %} $$ \bm{C}_t $$ {% endraw %} 来自对每个输入数据项根据「注意力」进行的加权。

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	splines=ortho
	{rank=same e1 e2 eddd en}
	{rank=same d1 d2 dddd dt0 dt dddd2}

	eddd[label="..."]
	dddd[label="..."]
	xddd[label="..."]
	yddd[label="..."]
	dt[label="d_t"]
	dt0[label="d_t-1"]
	yt[label="y_t"]
	yt0[label="y_t-1"]
	Ct[shape=plaintext]
	x1[shape=plaintext]
	x2[shape=plaintext]
	xddd[shape=plaintext]
	xn[shape=plaintext]
	y1[shape=plaintext]
	y2[shape=plaintext]
	yddd[shape=plaintext]
	dddd2[shape=plaintext, label=""]
	Ct[label="C_t", shape="square"]

	x1 -> e1
	x2 -> e2
	xddd -> eddd
	xn -> en

	e1 -> e2
	e2 -> eddd
	eddd -> en

	Ct -> dt

	d1 -> y1
	d2 -> y2
	dddd -> yddd
	dt0 -> yt0
	dt -> yt

	d1 -> d2
	d2 -> dddd
	dddd -> dt0
	dt0 -> dt

	e1 -> Ct
	e2 -> Ct
	eddd -> Ct
	en -> Ct

	dt -> dddd2
	dt0 -> Ct
}
{% endgraphviz %}
</div>

可以应用的场景，比如对于一个电商平台中很常见的白底图，其边缘的白色区域都是无用的，那么就不应该被关注（关注权重为 0）。比如机器翻译中，翻译词都是对局部输入重点关注的。

针对时刻 t 要产出的输出，隐藏层每一个隐藏细胞都与 {% raw %} $$ \bm{C}_t $$ {% endraw %} 有一个权重关系 {% raw %} $$ \bm{\alpha}_{t,i} $$ {% endraw %} 其中 {% raw %} $$ 1\le i\le n $$ {% endraw %}，这个权重值与「输入项经过编码器后隐藏层后的输出、解码器的前一时刻隐藏层输出」两者有关：

{% raw %}
$$
\begin{aligned}
&\bm{e} = tanh(\bm{W}^{xe} \cdot \bm{x} + \bm{b}^{xe}) \\
&s_{i,t} = score(\bm{e}_i,\bm{d}_{t-1}) \\
&\alpha_{i,t} = \frac{e^{s_{i,t}}}{\textstyle\sum_{j=1}^n e^{s_{j,t}}} \\
&\bm{C}_t = \displaystyle\sum_{i=1}^n \alpha_{i,t} \bm{e}_i \\
&\bm{d}_t = function(\bm{d}_{t-1}, \bm{y}_{t-1}, \bm{C}_t) \\
&\bm{y} = Softmax(\bm{W}^{dy} \cdot \bm{d} + \bm{b}^{dy})
\end{aligned}
$$
{% endraw %}

##### 参考：

* 《自然语言处理：基于预训练模型的方法》车万翔 等
* 《自然语言处理实战：预训练模型应用及其产品化》安库·A·帕特尔 等
* https://lilianweng.github.io/posts/2018-06-24-attention/

#### 5、Transformer

这里引用一段车教授在《自然语言处理：基于预训练模型的方法》中的一段话：

> 从本意上讲，其是将一个向量序列变换成另一个向量序列，所以可以翻译成“变换器”或“转换器”。其还有另一个含义是“变压器”，也就是对电压进行变换，所以翻译成变压器也比较形象。当然，还有一个更有趣的翻译是“变形金刚”，这一翻译不但体现了其能变换的特性，还寓意着该模型如同变形金刚一样强大。目前，Transformer还没有一个翻译的共识，绝大部分人更愿意使用其英文名。

##### 参考：

* 《自然语言处理：基于预训练模型的方法》车万翔 等
* 《自然语言处理实战：预训练模型应用及其产品化》安库·A·帕特尔 等
* https://github.com/lilianweng/transformer-tensorflow/blob/master/transformer.py

### Reference

1. https://stanford.edu/~shervine/teaching/cs-230/
2. https://www.analyticsvidhya.com/blog/2020/02/cnn-vs-rnn-vs-mlp-analyzing-3-types-of-neural-networks-in-deep-learning/
3. https://en.wikipedia.org/wiki/Recurrent_neural_network
4. https://www.telusinternational.com/insights/ai-data/article/difference-between-cnn-and-rnn
5. http://colah.github.io/posts/2015-08-Understanding-LSTMs/
6. https://zhuanlan.zhihu.com/p/52119092
7. https://katex.org/docs/supported.html
8. https://zhuanlan.zhihu.com/p/28054589
9. https://zhuanlan.zhihu.com/p/91315967
10. https://zhuanlan.zhihu.com/p/460967976
11. https://zhuanlan.zhihu.com/p/47184529
12. http://www.idryman.org/blog/2012/04/04/jekyll-graphviz-plugin/
13. http://wjhsh.net/jiangxinyang-p-9367497.html