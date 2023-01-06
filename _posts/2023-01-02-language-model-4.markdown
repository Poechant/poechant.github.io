---
layout: post
title:  麦克船长的 NLP 语言模型技术笔记 4：循环神经网络（RNN）
date:   2023-01-02 05:01:31 +0800
categories: ai
tags: [AI, 人工智能, RNN, 循环神经网络, LSTM]
description: 如果您喜欢机器学习，那么您一定会喜欢这篇文章。在这篇文章中，我们将深入探讨 RNN（循环神经网络），这是一种强大的神经网络模型，能够预测序列数据，例如文本、语音和时间序列。我们将通过生动的代码示例和实际案例来演示如何使用 RNN，并在日常生活中真实地体验它的功能。您将学习到如何使用 RNN 解决各种机器学习问题，并动手尝试运用 RNN 解决实际问题。这篇文章将为您提供一个完整的 RNN 入门指南，并使您对 RNN 有更深入的了解。
excerpt: 如果您喜欢机器学习，那么您一定会喜欢这篇文章。在这篇文章中，我们将深入探讨 RNN（循环神经网络），这是一种强大的神经网络模型，能够预测序列数据，例如文本、语音和时间序列。我们将通过生动的代码示例和实际案例来演示如何使用 RNN，并在日常生活中真实地体验它的功能。您将学习到如何使用 RNN 解决各种机器学习问题，并动手尝试运用 RNN 解决实际问题。这篇文章将为您提供一个完整的 RNN 入门指南，并使您对 RNN 有更深入的了解。
katex: True
location: 杭州
---

**本文目录**
* TOC
{:toc}

如果您喜欢机器学习，那么您一定会喜欢这篇文章。在这篇文章中，我们将深入探讨 RNN（循环神经网络），这是一种强大的神经网络模型，能够预测序列数据，例如文本、语音和时间序列。我们将通过生动的代码示例和实际案例来演示如何使用 RNN，并在日常生活中真实地体验它的功能。您将学习到如何使用 RNN 解决各种机器学习问题，并动手尝试运用 RNN 解决实际问题。这篇文章将为您提供一个完整的 RNN 入门指南，并使您对 RNN 有更深入的了解。

RNN（Recurrent Neural Network）的 R 是 Recurrent 的意思，所以这是一个贷循环的神经网络。首先要明白一点，你并不需要搞懂 CNN 后才能学习 RNN 模型。你只要了解了 MLP 就可以学习 RNN 了。

### 1、经典结构的 RNN

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

这种输入和输出数据项数一致的 RNN，一般叫做 N vs. N 的 RNN。如果我们用 PyTorch 来实现一个非常简单的经典 RNN 则如下：

```python
import torch
import torch.nn as nn

# 创建一个 RNN 实例
# 第一个参数
rnn = nn.RNN(10, 20, 1, batch_first=True)  # 实例化一个单向单层RNN

# 输入是一个形状为 (5, 3, 10) 的张量
# 5 个输入数据项（也可以说是样本）
# 3 个数据项是一个序列，有 3 个 steps
# 每个 step 有 10 个特征
input = torch.randn(5, 3, 10)

# 隐藏层是一个 (1, 5, 20) 的张量
h0 = torch.randn(1, 5, 20)

# 调用 rnn 函数后，返回输出、最终的隐藏状态
output, hn = rnn(input, h0)

print(output)
print(hn)
```

我们来解读一下这段代码：

* 这段代码实例化了一个带有 1 个隐藏层的 RNN 网络。
* 它的输入是一个形状为 (5, 3, 10) 的张量，表示有 5 个样本，每个样本有 3 个时间步，每个时间步的特征维度是 10。
* 初始隐藏状态是一个形状为 (1, 5, 20) 的张量。
* 调用 rnn 函数后，会返回输出和最终的隐藏状态。
* 输出的形状是 (5, 3, 20)，表示有 5 个样本，每个样本有 3 个时间步，每个时间步的输出维度是 20。
* 最终的隐藏状态的形状是 (1, 5, 20)，表示最后的隐藏状态是 5

但是上面的代码示例，并没有自己编写一个具体的 RNN，而是用了默认的 PyTorch 的 RNN，那么下面我们就自己编写一个：

```python
class MikeCaptainRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # 对于 RNN，输入维度就是序列数
        self.input_size = input_size

        # 隐藏层有多少个节点/神经元，经常将 hidden_size 设置为与序列长度相同
        self.hidden_size = hidden_size

        # 输入层到隐藏层的 W^{ih} 权重、bias^{ih} 偏置项
        self.weight_ih = torch.randn(self.hidden_size, self.input_size) * 0.01
        self.bias_ih = torch.randn(self.hidden_size)

        # 隐藏层到隐藏层的 W^{hh} 权重、bias^{hh} 偏置项
        self.weight_hh = torch.randn(self.hidden_size, self.hidden_size) * 0.01
        self.bias_hh = torch.randn(self.hidden_size)

    # 前向传播
    def forward(self, input, h0):

    	# 取出这个张量的形状
        N, L, input_size = input.shape

        # 初始化一个全零张量
        output = torch.zeros(N, L, self.hidden_size)

        # 处理每个时刻的输入特征
        for t in range(L):

        	# 获得当前时刻的输入特征，[N, input_size, 1]。unsqueeze(n)，在第 n 维上增加一维
            x = input[:, t, :].unsqueeze(2)  
            w_ih_batch = self.weight_ih.unsqueeze(0).tile(N, 1, 1)  # [N, hidden_size, input_size]
            w_hh_batch = self.weight_hh.unsqueeze(0).tile(N, 1, 1)  # [N, hidden_size, hidden_size]

            # bmm 是矩阵乘法函数
            w_times_x = torch.bmm(w_ih_batch, x).squeeze(-1)  # [N, hidden_size]。squeeze(n)，在第n维上减小一维
            w_times_h = torch.bmm(w_hh_batch, h0.unsqueeze(2)).squeeze(-1)  # [N, hidden_size]
            h0 = torch.tanh(w_times_x + self.bias_ih + w_times_h + self.bias_hh)
            output[:, t, :] = h0
        return output, h0.unsqueeze(0)
```

### 2、N vs.1 的 RNN

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

### 3、1 vs. N 的 RNN

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

### 4、LSTM（Long Short-Term Memory）长短时记忆网络

#### 4.1、如何理解这个 Short-Term 呢？

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
\bm{h}_t = \bm{h}_k + \bm{u}_{k+1} + \bm{h}_{k+2} + ... + \bm{u}_{t-1} + \bm{u}_t
$$
{% endraw %}

从这里我们就可以看到，第 t 时刻的隐藏层输出，直接关联到第 k 时刻的输出，t 到 k 时刻的相关性则用 {% raw %} $$ \bm{u}_{k+1} $$ {% endraw %} 到 {% raw %} $$ \bm{u}_t $$ {% endraw %} 相加表示。也就是有 t-k 的短期（Short Term）记忆。

#### 4.2、引入遗忘门 f、输入门 i、输出门 o、记忆细胞 c

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

### 5、双向循环神经网络、双向 LSTM

双向循环神经网络很好理解，就是两个方向都有，例如下图：

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
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

在 PyTorch 中使用 `nn.RNN` 就有参数表示双向：

> bidirectional – If True, becomes a bidirectional RNN. Default: False

bidirectional：默认设置为 False。若为 True，即为双向 RNN。

### 6、堆叠循环神经网络、堆叠 LSTM

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

在 PyTorch 中使用 `nn.RNN` 就有参数表示双向：

> num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1

num_layers：隐藏层层数，默认设置为 1 层。当 `num_layers` >= 2 时，就是一个 stacked RNN 了。

### 7、N vs. M 的 RNN

对于输入序列长度（长度 N）和输出序列长度（长度 M）不一样的 RNN 模型结构，也可以叫做 Encoder-Decoder 模型，也可以叫 Seq2Seq 模型。首先接收输入序列的 Encoder 先将输入序列转成一个隐藏态的上下文表示 C。C 可以只与最后一个隐藏层有关，甚至可以是最后一个隐藏层生成的隐藏态直接设置为 C，C 还可以与所有隐藏层有关。

有了这个 C 之后，再用 Decoder 进行解码，也就是从把 C 作为输入状态开始，生成输出序列。

这种的应用就非常广了，因为大多数时候输入序列与输出序列的长度都是不同的，比如最常见的应用「翻译」，从一个语言翻译成另一个语言；再比如 AI 的一个领域「语音识别」，将语音序列输入后生成所识别的文本内容；还有比如 ChatGPT 这种问答应用等等。

但是 Seq2Seq 模型有一个很显著的问题，就是当输入序列很长时，Encoder 生成的 Context 可能就会出现所捕捉的信息不充分的情况，导致 Decoder 最终的输出是不尽如人意的。

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

Reference

* https://www.cnblogs.com/engpj/p/16906911.html