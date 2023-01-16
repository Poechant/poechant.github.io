---
layout: post
title:  麦克船长 NLP 语言模型技术笔记 5：注意力机制（Attention Mechanism）
date:   2023-01-05 02:13:09 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, Attention, 注意力, AIGC, Transformer, 自注意力, Self-Attention, 多头注意力, Multiple Head Attention, 残差网络, Short-Cut, 位置编码, Bahdanau, Encoder-Decoder]
description: 基于 RNN 的 Encoder-Decoder 模型存在无法处理过长文本、并行性差的两大痛点。2015 年 Bahdanau 等人在其论文中提出 Attention 机制，再到 2017 年 Transformer 模型的论文《Attention is All You Need》横空出世，其并行速度极快，而且每两个词之间的词间距都是 1。此后 NLP 领域 Transformer 彻底成为主流。如果你已经了解 Encoder-Decoder 模型，本文将基于此带你深入浅出的搞清楚 Attention、Transformer。
excerpt: 基于 RNN 的 Encoder-Decoder 模型存在无法处理过长文本、并行性差的两大痛点。2015 年 Bahdanau 等人在其论文中提出 Attention 机制，再到 2017 年 Transformer 模型的论文《Attention is All You Need》横空出世，其并行速度极快，而且每两个词之间的词间距都是 1。此后 NLP 领域 Transformer 彻底成为主流。如果你已经了解 Encoder-Decoder 模型，本文将基于此带你深入浅出的搞清楚 Attention、Transformer。
katex: True
location: 杭州
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

## 一、为什么说 RNN 模型没有体现「注意力」？
 
Encoder-Decoder 的一个非常严重的问题，是依赖中间那个 context 向量，则无法处理特别长的输入序列 —— 记忆力不足，会忘事儿。而忘事儿的根本原因，是没有「注意力」。

对于一般的 RNN 模型，Encoder-Decoder 结构并没有体现「注意力」—— 这句话怎么理解？当输入序列经过 Encoder 生成的中间结果（上下文 C），被喂给 Decoder 时，这些中间结果对所生成序列里的哪个词，都没有区别（没有特别关照谁）。这相当于在说：输入序列里的每个词，对于生成任何一个输出的词的影响，是一样的，而不是输出某个词时是聚焦特定的一些输入词。这就是模型没有注意力机制。

人脑的注意力模型，其实是资源分配模型。NLP 领域的注意力模型，是在 2014 年被提出的，后来逐渐成为 NLP 领域的一个广泛应用的机制。可以应用的场景，比如对于一个电商平台中很常见的白底图，其边缘的白色区域都是无用的，那么就不应该被关注（关注权重为 0）。比如机器翻译中，翻译词都是对局部输入重点关注的。

所以 Attention 机制，就是在 Decoder 时，不是所有输出都依赖相同的「上下文 {% raw %} $$ \bm{C}_t $$ {% endraw %}」，而是时刻 t 的输出，使用 {% raw %} $$ \bm{C}_t $$ {% endraw %}，而这个 {% raw %} $$ \bm{C}_t $$ {% endraw %} 来自对每个输入数据项根据「注意力」进行的加权。

## 二、基于 Attention 机制的 Encoder-Decoder 模型

2015 年 Dzmitry Bahdanau 等人在论文[《Neural Machine Translation by Jointly Learning to Align and Translate》](https://arxiv.org/abs/1409.0473) 中提出了「Attention」机制，下面请跟着麦克船长，船长会深入浅出地为你解释清楚。

下图中 {% raw %} $$ e_i $$ {% endraw %} 表示编码器的隐藏层输出，{% raw %} $$ d_i $$ {% endraw %} 表示解码器的隐藏层输出

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

更进一步细化关于 {% raw %} $$ \bm{C}_t $$ {% endraw %} 部分，船长在此引用《基于深度学习的道路短期交通状态时空序列预测》一书中的图：

![image](/img/src/2023-01-04-captain-nlp-5.png)

这个图里的 {% raw %} $$ \widetilde{h}_i $$ {% endraw %} 与上一个图里的 {% raw %} $$ d_i $$ {% endraw %} 对应，{% raw %} $$ h_i $$ {% endraw %} 与上一个图里的 {% raw %} $$ e_i $$ {% endraw %} 对应。

针对时刻 {% raw %} $$ t $$ {% endraw %} 要产出的输出，隐藏层每一个隐藏细胞都与 {% raw %} $$ \bm{C}_t $$ {% endraw %} 有一个权重关系 {% raw %} $$ \alpha_{t,i} $$ {% endraw %} 其中 {% raw %} $$ 1\le i\le n $$ {% endraw %}，这个权重值与「输入项经过编码器后隐藏层后的输出{% raw %} $$ e_i（1\le i\le n） $$ {% endraw %}、解码器的前一时刻隐藏层输出 {% raw %} $$ d_{t-1} $$ {% endraw %}」两者有关：

{% raw %}
$$
\begin{aligned}
&s_{i,t} = score(\bm{e}_i,\bm{d}_{t-1}) \\
&\alpha_{i,t} = \frac{exp(s_{i,t})}{\textstyle\sum_{j=1}^n exp(s_{j,t})}
\end{aligned}
$$
{% endraw %}

常用的 {% raw %} $$ score $$ {% endraw %} 函数有：

* 点积（Dot Product）模型：{% raw %} $$ s_{i,t} = {\bm{d}_{t-1}}^T \cdot \bm{e}_i $$ {% endraw %}
* 缩放点积（Scaled Dot-Product）模型：{% raw %} $$ s_{i,t} = \frac{{\bm{d}_{t-1}}^T \cdot \bm{e}_i}{\sqrt{\smash[b]{dimensions\:of\:d_{t-1}\:or\:e_i}}} $$ {% endraw %}，可避免因为向量维度过大导致点积结果太大

然后上下文向量就表示成：

{% raw %}
$$
\begin{aligned}
&\bm{C}_t = \displaystyle\sum_{i=1}^n \alpha_{i,t} \bm{e}_i
\end{aligned}
$$
{% endraw %}

还记得 RNN 那部分里船长讲到的 Encoder-Decoder 模型的公式表示吗？

{% raw %}
$$
\begin{aligned}
e_t &= Encoder_{LSTM/GRU}(x_t, e_{t-1}) \\
\bm{C} &= f_1(e_n) \\
d_t &= f_2(d_{t-1}, \bm{C}) \\
y_t &= Decoder_{LSTM/GRU}(y_{t-1}, d_{t-1}, \bm{C})
\end{aligned}
$$
{% endraw %}

加入 Attention 机制的 Encoder-Decoder 模型如下。

{% raw %}
$$
\begin{aligned}
e_t &= Encoder_{LSTM/GRU}(x_t, e_{t-1}) \\
\bm{C}_t &= f_1(e_1,e_2...e_n,d_{t-1}) \\
d_t &= f_2(d_{t-1}, \bm{C}_t) \\
y_t &= Decoder_{LSTM/GRU}(y_{t-1}, d_{t-1}, \bm{C}_t)
\end{aligned}
$$
{% endraw %}

这种同时考虑 Encoder、Decoder 的 Attention，就叫做「Encoder-Decoder Attention」，也常被叫做「Vanilla Attention」。可以看到上面最核心的区别是第二个公式 {% raw %} $$ C_t $$ {% endraw %}。加入 Attention 后，对所有数据给予不同的注意力分布。具体地，比如我们用如下的函数来定义这个模型：

{% raw %}
$$
\begin{aligned}
\bm{e} &= tanh(\bm{W}^{xe} \cdot \bm{x} + \bm{b}^{xe}) \\
s_{i,t} &= score(\bm{e}_i,\bm{d}_{t-1}) \\
\alpha_{i,t} &= \frac{e^{s_{i,t}}}{\textstyle\sum_{j=1}^n e^{s_{j,t}}} \\
\bm{C}_t &= \displaystyle\sum_{i=1}^n \alpha_{i,t} \bm{e}_i \\
\bm{d}_t &= tanh(\bm{W}^{dd} \cdot \bm{d}_{t-1} + \bm{b}^{dd} +
				 \bm{W}^{yd} \cdot \bm{y}_{t-1} + \bm{b}^{yd} +
				 \bm{W}^{cd} \cdot \bm{C}_t + \bm{b}^{cd}) \\
\bm{y} &= Softmax(\bm{W}^{dy} \cdot \bm{d} + \bm{b}^{dy})
\end{aligned}
$$
{% endraw %}

到这里你能发现注意力机制的什么问题不？

* 这个注意力机制忽略了位置信息。比如 Tigers love rabbits 和 Rabbits love tigers 会产生一样的注意力分数。

## 三、Transformer 在 2017 年横空出世

船长先通过一个动画来看下 Transformer 是举例示意，该图来自 Google 的博客文章 [《Transformer: A Novel Neural Network Architecture for Language Understanding》](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)：

![image](/img/src/2023-01-04-language-model-5-11.gif)

中文网络里找到的解释得比较好的 blogs、answers，几乎都指向了同一篇博客：Jay Alammar 的[《The Illustrated Transformer》](http://jalammar.github.io/illustrated-transformer/)，所以建议读者搭配该篇文章阅读。

Transformer 模型中用到了自注意力（Self-Attention）、多头注意力（Multiple-Head Attention）、残差网络（ResNet）与捷径（Short-Cut）。下面我们先通过第 1 到第 4 小节把几个基本概念讲清楚，然后在第 5 小节讲解整体 Transformer 模型就会好理解很多了。最后第 6 小节我们来一段动手实践。

### 1、自注意力机制（Self-Attention）

自注意力是理解 Transformer 的关键，原作者在论文中限于篇幅，没有给出过多的解释。以下是我自己的理解，能够比较通透、符合常识地去理解 Transformer 中的一些神来之笔的概念。

#### 1.1、一段自然语言内容，其自身就「暗含」很多内部关联信息

在加入了 Attention 的 Encoder-Decoder 模型中，对输出序列 Y 中的一个词的注意力来自于输入序列 X，那么如果 X 和 Y 相等呢？什么场景会有这个需求？因为我们认为一段文字里某些词就是由于另外某些词而决定的，可以粗暴地理解为「完形填空」的原理。那么这样一段文字，其实就存在其中每个词的自注意力，举个例子：

> 老王是我的主管，我很喜欢他的平易近人。

对这句话里的「他」，如果基于这句话计算自注意力的话，显然应该给予「老王」最多的注意力。受此启发，我们认为：

> 一段自然语言中，其实暗含了：为了得到关于某方面信息 Q，可以通过关注某些信息 K，进而得到某些信息（V）作为结果。

Q 就是 query 检索/查询，K、V 分别是 key、value。所以类似于我们在图书检索系统里搜索「NLP书籍」（这是 Q），得到了一本叫《自然语言处理实战》的电子书，书名就是 key，这本电子书就是 value。只是对于自然语言的理解，我们认为任何一段内容里，都自身暗含了很多潜在 Q-K-V 的关联。这是整体受到信息检索领域里 query-key-value 的启发的。

基于这个启发，我们将自注意力的公式表示为：

{% raw %}
$$
\begin{aligned}
Z = SelfAttention(X) = Attention(Q,K,V)
\end{aligned}
$$
{% endraw %}

X 经过自注意力计算后，得到的「暗含」了大量原数据内部信息的 Z。然后我们拿着这个带有自注意力信息的 Z 进行后续的操作。这里要强调的是，Z 向量中的每个元素 z_i 都与 X 的所有元素有某种关联，而不是只与 x_i 有关联。

#### 1.2、如何计算 Q、K、V

Q、K、V 全部来自输入 X 的线性变换：

{% raw %}
$$
\begin{aligned}
Q &= W^Q \cdot X \\
K &= W^K \cdot X \\
V &= W^V \cdot X
\end{aligned}
$$
{% endraw %}

{% raw %} $$ W^Q、W^K、W^V $$ {% endraw %} 以随机初始化开始，经过训练就会得到非常好的表现。对于 {% raw %} $$ X $$ {% endraw %} 中的每一个词向量 {% raw %} $$ x_i $$ {% endraw %}，经过这个变换后得到了：

{% raw %}
$$
\begin{aligned}
q_i &= W^Q \cdot x_i \\
k_i &= W^K \cdot x_i \\
v_i &= W^V \cdot x_i
\end{aligned}
$$
{% endraw %}

#### 1.3、注意力函数：如何通过 Q、V 得到 Z

基于上面的启发，我们认为 X 经过自注意力的挖掘后，得到了：

* 暗含信息 1：一组 query 与一组 key 之间的关联，记作 qk（想一下信息检索系统要用 query 先招到 key）
* 暗含信息 2：一组 value
* 暗含信息 3：qk 与 value 的某种关联

这三组信息，分别如何表示呢？这里又需要一些启发了，因为计算机科学其实是在「模拟还原」现实世界，在 AI 的领域目前的研究方向就是模拟还原人脑的思考。所以这种「模拟还原」都是寻找某一种近似方法，因此不能按照数学、物理的逻辑推理来理解，而应该按照「工程」或者「计算科学」来理解，想想我们大学时学的「计算方法」这门课，因此常需要一些启发来找到某种「表示」。

这里 Transformer 的作者，认为 {% raw %} $$ Q $$ {% endraw %} 和 {% raw %} $$ K $$ {% endraw %} 两个向量之间的关联，是我们在用 {% raw %} $$ Q $$ {% endraw %} 找其在 {% raw %} $$ K $$ {% endraw %} 上的投影，如果 {% raw %} $$ Q $$ {% endraw %}、{% raw %} $$ K $$ {% endraw %} 是单位长度的向量，那么这个投影其实可以理解为找「{% raw %} $$ Q $$ {% endraw %} 和 {% raw %} $$ K $$ {% endraw %} 向量之间的相似度」：

* 如果 {% raw %} $$ Q $$ {% endraw %} 和 {% raw %} $$ K $$ {% endraw %} 垂直，那么两个向量正交，其点积（Dot Product）为 0；
* 如果 {% raw %} $$ Q $$ {% endraw %} 和 {% raw %} $$ K $$ {% endraw %} 平行，那么两个向量点积为两者模积 {% raw %} $$ \|Q\|\|K\| $$ {% endraw %}；
* 如果 {% raw %} $$ Q $$ {% endraw %} 和 {% raw %} $$ K $$ {% endraw %} 呈某个夹角，则点积就是 {% raw %} $$ Q $$ {% endraw %} 在 {% raw %} $$ K $$ {% endraw %} 上的投影的模。

因此「暗含信息 1」就可以用「{% raw %} $$ Q\cdot K $$ {% endraw %}」再经过 Softmax 归一化来表示。这个表示，是一个所有元素都是 0~1 的矩阵，可以理解成对应注意力机制里的「注意力分数」，也就是一个「注意力分数矩阵（Attention Score Matrix）」。

而「暗含信息 2」则是输入 {% raw %} $$ X $$ {% endraw %} 经过的线性变换后的特征，看做 {% raw %} $$ X $$ {% endraw %} 的另一种表示。然后我们用这个「注意力分数矩阵」来加持一下 {% raw %} $$ V $$ {% endraw %}，这个点积过程就表示了「暗含信息 3」了。所以我们有了如下公式：

{% raw %}
$$
\begin{aligned}
Z = Attention(Q,K,V) = Softmax(Q \cdot K^T) \cdot V
\end{aligned}
$$
{% endraw %}

其实到这里，这个注意力函数已经可以用了。有时候，为了避免因为向量维度过大，导致 {% raw %} $$ Q \cdot K^T $$ {% endraw %} 点积结果过大，我们再加一步处理：

{% raw %}
$$
\begin{aligned}
Z = Attention(Q,K,V) = Softmax(\frac{Q \cdot K^T}{\sqrt{\smash[b]{d_k}}}) \cdot V
\end{aligned}
$$
{% endraw %}

这里 {% raw %} $$ d_k $$ {% endraw %} 是 K 矩阵中向量 {% raw %} $$ k_i $$ {% endraw %} 的维度。这一步修正还有进一步的解释，即如果经过 Softmax 归一化后模型稳定性存在问题。怎么理解？如果假设 Q 和 K 中的每个向量的每一维数据都具有零均值、单位方差，这样输入数据是具有稳定性的，那么如何让「暗含信息 1」计算后仍然具有稳定性呢？即运算结果依然保持零均值、单位方差，就是除以「{% raw %} $$ \sqrt{\smash[b]{d_k}} $$ {% endraw %}」。

到这里我们注意到：

* K、V 里的每一个向量，都是

#### 1.4、其他注意力函数

为了提醒大家这种暗含信息的表示，都只是计算方法上的一种选择，好坏全靠结果评定，所以包括上面的在内，常见的注意力函数有（甚至你也可以自己定义）：

{% raw %}
$$
Z = Attention(Q,K,V) =
\begin{cases}
\begin{aligned}
&= Softmax(Q^T K) V \\
&= Softmax(\frac{Q K^T}{\sqrt{\smash[b]{d_k}}}) V \\
&= Softmax(\omega^T tanh(W[q;k])) V \\
&= Softmax(Q^T W K) V \\
&= cosine[Q^T K] V
\end{aligned}
\end{cases}
$$
{% endraw %}

到这里，我们就从原始的输入 {% raw %} $$ X $$ {% endraw %} 得到了一个包含自注意力信息的 {% raw %} $$ Z $$ {% endraw %} 了，后续就可以用 {% raw %} $$ Z $$ {% endraw %} 了。

### 2、多头注意力

到这里我们理解了「自注意力」，而 Transformer 这篇论文通过添加「多头」注意力的机制进一步提升了注意力层。我们先看下它是什么，然后看下它的优点。从本小节开始，本文大量插图引用自[《The Illustrated Transformer》](http://jalammar.github.io/illustrated-transformer/)，作者 Jay Alammar 写出一篇非常深入浅出的图解文章，被大量引用，非常出色，再次建议大家去阅读。

Transformer 中用了 8 个头，也就是 8 组不同的 Q-K-V：

{% raw %}
$$
\begin{aligned}
Q_0 = W_0^Q \cdot X ;\enspace K_0 = &W_0^K \cdot X ;\enspace V_0 = W_0^V \cdot X \\
Q_1 = W_1^Q \cdot X ;\enspace K_1 = &W_0^K \cdot X ;\enspace V_1 = W_1^V \cdot X \\
&.... \\
Q_7 = W_7^Q \cdot X ;\enspace K_7 = &W_0^K \cdot X ;\enspace V_7 = W_7^V \cdot X
\end{aligned}
$$
{% endraw %}

这样我们就能得到 8 个 Z：

{% raw %}
$$
\begin{aligned}
&Z_0 = Attention(Q_0,K_0,V_0) = Softmax(\frac{Q_0 \cdot K_0^T}{\sqrt{\smash[b]{d_k}}}) \cdot V_0 \\
&Z_1 = Attention(Q_1,K_1,V_1) = Softmax(\frac{Q_1 \cdot K_1^T}{\sqrt{\smash[b]{d_k}}}) \cdot V_1 \\
&... \\
&Z_7 = Attention(Q_7,K_7,V_7) = Softmax(\frac{Q_7 \cdot K_7^T}{\sqrt{\smash[b]{d_k}}}) \cdot V_7 \\
\end{aligned}
$$
{% endraw %}

然后我们把 {% raw %} $$ Z_0 $$ {% endraw %} 到 {% raw %} $$ Z_7 $$ {% endraw %} 沿着行数不变的方向全部连接起来，如下图所示：

![image](/img/src/2023-01-04-language-model-5-3.png){: width="464" }

我们再训练一个权重矩阵 {% raw %} $$ W^O $$ {% endraw %}，然后用上面拼接的 {% raw %} $$ Z_{0-7} $$ {% endraw %} 乘以这个权重矩阵：

![image](/img/src/2023-01-04-language-model-5-4.png){: width="135" }

于是我们会得到一个 Z 矩阵：

![image](/img/src/2023-01-04-language-model-5-5.png){: width="100" }

到这里就是多头注意力机制的全部内容，与单头注意力相比，都是为了得到一个 Z 矩阵，但是多头用了多组 Q-K-V，然后经过拼接、乘以权重矩阵得到最后的 Z。我们总览一下整个过程：

![image](/img/src/2023-01-04-language-model-5-6.png){: width="935" }

通过多头注意力，每个头都会关注到不同的信息，可以如下类似表示：

![image](/img/src/2023-01-04-language-model-5-7.png){: width="400"}

这通过两种方式提高了注意力层的性能：

* 多头注意力机制，扩展了模型关注不同位置的能力。{% raw %} $$ Z $$ {% endraw %} 矩阵中的每个向量 {% raw %} $$ z_i $$ {% endraw %} 包含了与 {% raw %} $$ X $$ {% endraw %} 中所有向量 {% raw %} $$ x_i $$ {% endraw %} 有关的一点编码信息。反过来说，不要认为 {% raw %} $$ z_i $$ {% endraw %} 只与 {% raw %} $$ x_i $$ {% endraw %} 有关。
* 多头注意力机制，为注意力层提供了多个「表示子空间 Q-K-V」，以及 Z。这样一个输入矩阵 {% raw %} $$ X $$ {% endraw %}，就会被表示成 8 种不同的矩阵 Z，都包含了原始数据信息的某种解读暗含其中。

### 3、退化现象、残差网络与 Short-Cut

#### 3.1、退化现象

对于一个 56 层的神经网路，我们很自然地会觉得应该比 20 层的神经网络的效果要好，比如说从误差率（error）的量化角度看。但是华人学者何凯明等人的论文[《Deep Residual Learning for Image Recognition》](https://arxiv.org/pdf/1512.03385.pdf)中给我们呈现了相反的结果，而这个问题的原因并不是因为层数多带来的梯度爆炸/梯度消失（毕竟已经用了归一化解决了这个问题），而是因为一种反常的现象，这种现象我们称之为「退化现象」。何凯明等人认为这是因为存在「难以优化好的网络层」。

#### 3.2、恒等映射

如果这 36 层还帮了倒忙，那还不如没有，是不是？所以这多出来的 36 个网络层，如果对于提升性能（例如误差率）毫无影响，甚至更进一步，这 36 层前的输入数据，和经过这 36 层后的输出数据，完全相同，那么如果将这 36 层抽象成一个函数 {% raw %} $$ f_{36} $$ {% endraw %}，这就是一个恒等映射的函数：

{% raw %} $$ f_{36}(x) = x $$ {% endraw %}

回到实际应用中。如果我们对于一个神经网络中的连续 N 层是提升性能，还是降低性能，是未知的，那么则可以建立一个跳过这些层的连接，实现：

> 如果这 N 层可以提升性能，则采用这 N 层；否则就跳过。

这就像给了这 N 层神经网络一个试错的空间，待我们确认它们的性能后再决定是否采用它们。同时也可以理解成，这些层可以去单独优化，如果性能提升，则不被跳过。

#### 3.3、残差网络（Residual Network）与捷径（Short-Cut）

如果前面 20 层已经可以实现 99% 的准确率，那么引入了这 36 层能否再提升「残差剩余那 1%」的准确率从而达到 100% 呢？所以这 36 层的网络，就被称为「残差网络（Residual Network，常简称为 ResNet）」，这个叫法非常形象。

而那个可以跳过 N 层残差网络的捷径，则常被称为 Short-Cut，也会被叫做跳跃链接（Skip Conntection），这就解决了上述深度学习中的「退化现象」。

### 4、位置编码（Positional Embedding）

还记得我在第二部分最后提到的吗：

> 这个注意力机制忽略了位置信息。比如 Tigers love rabbits 和 Rabbits love tigers 会产生一样的注意力分数。

#### 4.1、Transformer 论文中的三角式位置编码（Sinusoidal Positional Encoding）

现在我们来解决这个问题，为每一个输入向量 {% raw %} $$ x_i $$ {% endraw %} 生成一个位置编码向量 {% raw %} $$ t_i $$ {% endraw %}，这个位置编码向量的维度，与输入向量（词的嵌入式向量表示）的维度是相同的：

![image](/img/src/2023-01-04-language-model-5-8.png){: width="500"}

Transformer 论文中给出了如下的公式，来计算位置编码向量的每一位的值：

{% raw %}
$$
\begin{aligned}
P_{pos,2i} &= sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
P_{pos,2i+1} &= cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
\end{aligned}
$$
{% endraw %}

这样对于一个 embedding，如果它在输入内容中的位置是 pos，那么其编码向量就表示为：

{% raw %}
$$
\begin{aligned}
[P_{pos,0}, P_{pos,1}, ... , P_{pos,d_x-1}]
\end{aligned}
$$
{% endraw %}

延展开的话，位置编码其实还分为绝对位置编码（Absolute Positional Encoding）、相对位置编码（Relative Positional Encoding）。前者是专门生成位置编码，并想办法融入到输入中，我们上面看到的就是一种。后者是微调 Attention 结构，使得它可以分辨不同位置的数据。另外其实还有一些无法分类到这两种的位置编码方法。

#### 4.2、绝对位置编码

绝对位置编码，如上面提到的，就是定义一个位置编码向量 {% raw %} $$ t_i $$ {% endraw %}，通过 {% raw %} $$ x_i + t_i $$ {% endraw %} 就得到了一个含有位置信息的向量。

* 习得式位置编码（Learned Positional Encoding）：将位置编码当做训练参数，生成一个「最大长度 x 编码维度」的位置编码矩阵，随着训练进行更新。目前 Google BERT、OpenAI GPT 模型都是用的这种位置编码。缺点是「外推性」差，如果文本长度超过之前训练时用的「最大长度」则无法处理。目前有一些给出优化方案的论文，比如「[层次分解位置编码](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247515573&idx=1&sn=2d719108244ada7db3a535a435631210&chksm=96ea6235a19deb23babde5eaac484d69e4c2f53bab72d2e350f75bed18323eea3cf9be30615b#rd)」。
* 三角式位置编码（Sinusoidal Positional Encodign）：上面提过了。
* 循环式位置编码（Recurrent Positional Encoding）：通过一个 RNN 再接一个 Transformer，那么 RNN 暗含的「顺序」就导致不再需要额外编码了。但这样牺牲了并行性，毕竟 RNN 的两大缺点之一就有这个。
* 相乘式位置编码（Product Positional Encoding）：用「{% raw %} $$ x_i \odot t_i $$ {% endraw %}」代替「{% raw %} $$ x_i + t_i $$ {% endraw %}」。

#### 4.3、相对位置编码和其他位置编码

最早来自于 Google 的论文[《Self-Attention with Relative Position Representations》](https://arxiv.org/abs/1803.02155)相对位置编码，考虑的是当前 position 与被 attention 的 position 之前的相对位置。

* 常见相对位置编码：经典式、XLNET 式、T5 式、DeBERTa 式等。
* 其他位置编码：CNN 式、复数式、融合式等。

到此我们都是在讲 Encoder，目前我们知道一个 Encoder 可以用如下的示意图表示：

![image](/img/src/2023-01-04-language-model-5-12.png){: width="680"}

### 5、编码器 Encoder 和解码器 Decoder

#### 5.1、Encoder 和 Decoder 的图示结构

![image](/img/src/2023-01-04-language-model-5-15.png){: width="165"}

* 第一层是多头注意力层（Multi-Head Attention Layer）。
* 第二层是经过一个前馈神经网络（Feed Forward Neural Network，简称 FFNN）。
* 这两层，每一层都有「Add & Normalization」和 ResNet。

![image](/img/src/2023-01-04-language-model-5-14.png){: width="179"}

* 解码器有两个多头注意力层。第一个多头注意力层是 Masked Multi-Head Attention 层，即在自注意力计算的过程中只有前面位置上的内容。第二个多头注意力层买有被 Masked，是个正常多头注意力层。
* 可以看出来，第一个注意力层是一个自注意力层（Self Attention Layer），第二个是 Encoder-Decoder Attention 层（它的 K、V 来自 Encoder，Q 来自自注意力层），有些文章里会用这个角度来指代。
* FNN、Add & Norm、ResNet 都与 Encoder 类似。

#### 5.2、Decoder 的第一个输出结果

产出第一个最终输出结果的过程：

* 不需要经过 Masked Multi-Head Attention Layer（自注意力层）。
* 只经过 Encoder-Decoder Attention Layer。

![image](/img/src/2023-01-04-language-model-5-13.png){: width="695"}

这样我们就像前面的 Encoder-Decoder Attention 模型一样，得到第一个输出。但是最终的输出结果，还会经过一层「Linear + Softmax」。

#### 5.3、Decoder 后续的所有输出

从产出第二个输出结果开始：

* Decoder 的自注意力层，会用到前面的输出结果。
* 可以看到，这是一个串行过程。

#### 5.4、Decoder 之后的 Linear 和 Softmax

经过所有 Decoder 之后，我们得到了一大堆浮点数的结果。最后的 Linear & Softmax 就是来解决「怎么把它变成文本」的问题的。

* Linear 是一个全连接神经网络，把 Decoders 输出的结果投影到一个超大的向量上，我们称之为 logits 向量。
* 如果我们的输出词汇表有 1 万个词，那么 logits 向量的每一个维度就有 1 万个单元，每个单元都对应输出词汇表的一个词的概率。
* Softmax 将 logits 向量中的每一个维度都做归一化，这样每个维度都能从 1 万个单元对应的词概率中选出最大的，对应的词汇表里的词，就是输出词。最终得到一个输出字符串。

### 6、Transformer 模型整体

![image](/img/src/2023-01-04-language-model-5-16.png){: width="660"}

最后我们再来整体看一下 Transformer：

* 首先输入数据生成词的嵌入式向量表示（Embedding），生成位置编码（Positional Encoding，简称 PE）。
* 进入 Encoders 部分。先进入多头注意力层（Multi-Head Attention），是自注意力处理，然后进入全连接层（又叫前馈神经网络层），每层都有 ResNet、Add & Norm。
* 每一个 Encoder 的输入，都来自前一个 Encoder 的输出，但是第一个 Encoder 的输入就是 Embedding + PE。
* 进入 Decoders 部分。先进入第一个多头注意力层（是 Masked 自注意力层），再进入第二个多头注意力层（是 Encoder-Decoder 注意力层），每层都有 ResNet、Add & Norm。
* 每一个 Decoder 都有两部分输入。
* Decoder 的第一层（Maksed 多头自注意力层）的输入，都来自前一个 Decoder 的输出，但是第一个 Decoder 是不经过第一层的（因为经过算出来也是 0）。
* Decoder 的第二层（Encoder-Decoder 注意力层）的输入，Q 都来自该 Decoder 的第一层，且每个 Decoder 的这一层的 K、V 都是一样的，均来自最后一个 Encoder。
* 最后经过 Linear、Softmax 归一化。

### 7、Transformer 的性能

Google 在其博客于 2017.08.31 发布如下测试数据：

|![image](/img/src/2023-01-04-language-model-5-9.png)|![image](/img/src/2023-01-04-language-model-5-10.png)|
|-|-|
| | |

## 四、一个基于 TensorFlow 架构的 Transformer 实现

我们来看看一个简单的 Transformer 模型，就是比较早出现的 Kyubyong 实现的 Transformer 模型：https://github.com/Kyubyong/transformer/tree/master/tf1.2_legacy

### 1、先训练和测试一下 Kyubyong Transformer

下载一个「德语-英语翻译」的数据集：https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8

把 `de-en` 下面的 `tgz` 解压后放在 `corpora/` 目录下。如果需要先修改超参数，需要修改 `hyperparams.py`。然后运行如下命令，生成词汇文件（vocabulary files），默认到 `preprocessed` 目录下：

```shell
python prepro.py
```

然后开始训练：

```shell
python train.py
```

也可以跳过训练，直接[下载预训练过的文件](https://www.dropbox.com/s/fo5wqgnbmvalwwq/logdir.zip?dl=0)，是一个 `logdir/` 目录，把它放到项目根目录下。然后可以对训练出来的结果，运行评价程序啦：

```shell
python eval.py
```

会生成「德语-英语」测试结果文件在 `results/` 目录下，内容如下：

```
- source: Sie war eine jährige Frau namens Alex
- expected: She was a yearold woman named Alex
- got: She was a <UNK> of vote called <UNK>

- source: Und als ich das hörte war ich erleichtert
- expected: Now when I heard this I was so relieved
- got: And when I was I <UNK> 's

- source: Meine Kommilitonin bekam nämlich einen Brandstifter als ersten Patienten
- expected: My classmate got an arsonist for her first client
- got: Because my first eye was a first show

- source: Das kriege ich hin dachte ich mir
- expected: This I thought I could handle
- got: I would give it to me a day

- source: Aber ich habe es nicht hingekriegt
- expected: But I didn't handle it
- got: But I didn't <UNK> <UNK>

- source: Ich hielt dagegen
- expected: I pushed back
- got: I <UNK>

...

Bleu Score = 6.598452846670836
```

评估结果文件的最后一行是 Bleu Score：

* 这是用来评估机器翻译质量的一种度量方式。它是由几个不同的 BLEU 分数组成的，每个 BLEU 分数都表示翻译结果中与参考翻译的重叠程度。
* 一个常用的 BLEU 分数是 BLEU-4，它计算翻译结果中与参考翻译的 N 元文法语言模型 n-gram（n 为 4）的重叠程度。分数越高表示翻译结果越接近参考翻译。

### 2、Kyubyong Transformer 源码分析

* `hparams.py`：超参数都在这里，仅 30 行。将在下面 `2.1` 部分解读。
* `data_load.py`：装载、批处理数据的相关函数，代码仅 92 行。主要在下面 `2.2` 部分解读。
* `prepro.py`：为 source 和 target 创建词汇文件（vocabulary file），代码仅 39 行。下面 `2.3` 部分会为大家解读。
* `train.py`：代码仅 184 行。在下面 `2.4` 部分解读。
* `modules.py`：Encoding / Decoding 网络的构建模块，代码仅 329 行。与 `modules.py` 一起会在 `2.4` 部分解读。
* `eval.py`：评估效果，代码仅 82 行。将在 `2.5` 部分解读

总计 700 多行代码。

#### 2.1、超参数

`hyperparams.py` 文件中定义了 `Hyperparams` 超参数类，其中包含的参数我们逐一来解释一下：

* `source_train`：训练数据集的源输入文件，默认是 `'corpora/train.tags.de-en.de'`
* `target_train`：训练数据集的目标输出文件，默认是 `'corpora/train.tags.de-en.en'`
* `source_test`：测试数据集的源输入文件，默认是 `'corpora/IWSLT16.TED.tst2014.de-en.de.xml'`
* `target_test`：测试数据集的目标输出文件，默认是 `'corpora/IWSLT16.TED.tst2014.de-en.en.xml'`
* `batch_size`：设置每批数据的大小。
* `lr`：设置学习率 learning rate。
* `logdir`：设置日志文件保存的目录。
* `maxlen`
* `min_cnt`
* `hidden_units`：设置编码器和解码器中隐藏层单元的数量。
* `num_blocks`：编码器（encoder block）、解码器（decoder block）的数量
* `num_epochs`：训练过程中迭代的次数。
* `num_heads`：还记得上面文章里我们提到的 Transformer 中用到了多头注意力吧，这里就是多头注意力的头数。
* `droupout_rate`：设置 dropout 层的 dropout rate，具体 dropout 请看 2.4.1 部分。
* `sinusoid`：设置为 `True` 时表示使用正弦函数计算位置编码，否则为 `False` 时表示直接用 `position` 做位置编码。

#### 2.2、预处理

文件 `prepro.py` 实现了预处理的过程，根据 `hp.source_train` 和 `hp.target_train` 分别创建 `"de.vocab.tsv"` 和 `"en.vocab.tsv"` 两个词汇表。

```python
def make_vocab(fpath, fname):

    # 使用 codecs.open 函数读取指定文件路径(fpath)的文本内容，并将其存储在 text 变量中
    text = codecs.open(fpath, 'r', 'utf-8').read()

    # 将 text 中的非字母和空格的字符去掉
    text = regex.sub("[^\s\p{Latin}']", "", text)

    # 将 text 中的文本按照空格分割，并将每个单词存储在 words 变量中
    words = text.split()

    # words 中每个单词的词频
    word2cnt = Counter(words)

    # 检查是否存在 preprocessed 文件夹，如果不存在就创建
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:

    	# 按出现次数从多到少的顺序写入每个单词和它的出现次数
    	# 在文件最前面写入四个特殊字符 <PAD>, <UNK>, <S>, </S> 分别用于填充，未知单词，句子开始和句子结束
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    make_vocab(hp.source_train, "de.vocab.tsv")
    make_vocab(hp.target_train, "en.vocab.tsv")
    print("Done")
```

* 在主函数中调用 make_vocab 函数，在目录 `preprocessed` 生成 `de.vocab.tsv` 和 `en.vocab.tsv` 两个词汇表文件。
* 在函数 `make_vocab` 中，先使用 `codecs.open` 函数读取指定文件路径 `fpath` 的文本内容，并将其存储在 `text` 变量中，再使用正则表达式 `regex` 将 `text` 中的非字母和空格的字符去掉，接着将 `text` 中的文本按照空格分割，并将每个单词存储在 `words` 变量中。
* 接下来，使用 `Counter` 函数统计 `words` 中每个单词的出现次数，并将统计结果存储在 `word2cnt` 变量中。
* 最后所有词与词频，存储在 `de.vocab.tsv` 和 `en.vocab.tsv` 两个文件中。

#### 2.3、训练/测试数据集的加载

我们先看下 `train.py`、`data_load.py`、`eval.py` 三个文件：

* `train.py`：该文件包含了 `Graph` 类的定义，并在其构造函数中调用 `load_data.py` 文件中的 `get_batch_data` 函数加载训练数据。
* `data_load.py`：定义了加载训练数据、加载测试数据的函数。
* `eval.py`：测试结果的评价函数定义在这个文件里。

下面是函数调用的流程：

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=LR
	splines=ortho
	node [shape="box"]

	训练 -> Graph构造函数 -> get_batch_data -> load_train_data
	测试 -> eval -> load_test_data

	load_train_data -> create_data
	load_test_data -> create_data

	create_data -> load_de_vocab
	create_data -> load_en_vocab
}
{% endgraphviz %}
</div>

```python
def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word
```

将 `preprocessed/de.vocab.tsv` 和 `preprocessed/en.vocab.tsv` 中储存的德语、英语的词汇、词频，载入成 `word2idx` 和 `idx2word`。前者是通过词查询词向量，后者通过词向量查询词。

`load_de_vocab` 和 `load_en_vocab` 函数被 `create_data` 函数引用，该函数将输入的源语言和目标语言句子转换为索引表示，并对过长的句子进行截断或填充。详细的解释看下面代码里的注释。

```python
# 输入参数是翻译模型的源语言语句、目标语言语句
def create_data(source_sents, target_sents):

    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # 用 zip 函数将源语言和目标语言句子对应起来，并对句子进行截断或填充
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 

        # 将句子的词的编号，原句以及编号后的句子存储下来，以供之后使用
        if max(len(x), len(y)) <=hp.maxlen:

        	# 将 x 和 y 转换成 numpy 数组并加入 x_list 和 y_list 中
            x_list.append(np.array(x))
            y_list.append(np.array(y))

            # 将原始的 source_sent 和 target_sent 加入 Sources 和 Targets 列表中
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # 对于每个 (x, y) 对，使用 np.lib.pad 函数将 x 和 y 分别用 0 进行填充，直到长度为 hp.maxlen
    # 这样做的目的是使得每个句子长度都相等，方便后续的训练
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))

    # 返回转换后的索引表示，以及未经处理的源语言和目标语言句子
    # X 是原始句子中德语的索引
    # Y 是原始句子中英语的索引
    # Sources 是源原始句子列表，并与 X 一一对应
    # Targets 是目标原始句子列表，并与 Y 一一对应
    return X, Y, Sources, Targets

# 返回原始句子中德语、英语的索引
def load_train_data():
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y
```

下面的 `get_batch_data` 则从文本数据中读取并生成 batch：

```python
def get_batch_data():
    
    # 加载数据
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # 将 X 和 Y 转换成张量
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # 创建输入队列
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # 创建 batch 队列，利用 shuffle_batch 将一组 tensor 随机打乱，并将它们分为多个 batch
    # 使用 shuffle_batch 是为了防止模型过拟合
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()
```

#### 2.4、构建模型并训练

Graph 的构造函数流程，就是模型的构建流程，下面船长来分析这部分代码。

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=LR
	splines=ortho
	node [shape="box"]

	Graph构造函数 -> 编码器 -> 解码器 -> Linear -> Softmax
}
{% endgraphviz %}
</div>

整体这个流程，主要涉及 `train.py` 文件和 `modules.py` 文件。所有模型所需的主要函数定义，都是在 `modules.py` 中实现的。我们先看下编码器（Encoder）的流程：

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	splines=ortho
	node [shape="box"]

	embedding -> positional_encoding -> dropout -> multihead_attention -> feedforward
}
{% endgraphviz %}
</div>


下面是 `train.py` 中实现的 Transformer 流程，其中的每一段代码，船长都会做详细解释，先不用急。这个流程里，首先定义了编码器，先使用了 Embedding 层将输入数据转换为词向量，使用 Positional Encoding 层对词向量进行位置编码，使用 Dropout 层进行 dropout 操作，然后进行多层 Multihead Attention 和 Feed Forward 操作。

在构建模型前，先执行 `train.py` 的主程序段，首先 `if __name__ == '__main__'` 这句代码是在 Python 中常用的一种编写方式，它的意思是当一个文件被直接运行时，`if` 语句下面的代码会被执行。请看下面代码的注释。

```python
if __name__ == '__main__':                
    
    # 加载词汇表   
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # 构建模型并训练
    g = Graph("train"); print("Graph loaded")
    
    # 创建了一个 Supervisor 对象来管理训练过程
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)

    # 使用 with 语句打开一个会话
    with sv.managed_session() as sess:

    	# 训练迭代 hp.num_epochs 次
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break

            # tqdm 是一个 Python 库，用来在循环执行训练操作时在命令行中显示进度条
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):

            	# 每次迭代都会运行训练操作 g.train_op
                sess.run(g.train_op)

            # 获取训练的步数，通过 sess.run() 函数获取 global_step 的当前值并赋值给 gs。这样可在后面使用 gs 保存模型时用这个值命名模型
            gs = sess.run(g.global_step)

            # 每个 epoch 结束时，它使用 saver.save() 函数保存当前模型的状态
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    
    print("Done")
```

* `num_epochs` 是训练过程中迭代的次数，它表示训练模型需要在训练数据上跑多少遍。每一次迭代都会在训练数据集上进行训练，通常来说，训练数据集会被重复多次迭代，直到达到 `num_epochs` 次。这样可以确保模型能够充分地学习数据的特征。设置 `num_epochs` 的值过大或过小都会导致模型性能下降。

##### 2.4.1、编码过程

###### Embedding

`embedding` 用来把输入生成词嵌入向量：

```python
# 词语转换为对应的词向量表示
self.enc = embedding(self.x, 
                      vocab_size=len(de2idx), 
                      num_units=hp.hidden_units, 
                      scale=True,
                      scope="enc_embed")
```

* `vocab_size` 是词汇表的大小。
* `num_units` 是词向量的维度。
* `scale` 是一个布尔值，用来确定是否对词向量进行标准化。
* `scope` 是变量作用域的名称。

###### Key Masks

接着生成一个 `key_masks` 用于在之后的计算中屏蔽掉某些位置的信息，以便模型只关注有效的信息。

```python
key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1)
```

* 先对 `self.enc` 张量进行对每个元素求绝对值的操作
* 沿着最后一阶作为轴，进行 `reduce_sum` 操作，得到一个 (batch, sequence_length) 形状的张量。
* 再进行 `tf.sign` 操作，对刚得到的每个元素进行符号函数的变换。
* 最后再扩展阶数，变成形状 (batch, sequence_length, 1) 的张量。

###### Positional Encoding

下面生成 Transformer 的位置编码：

```python
# 位置编码
if hp.sinusoid:
    self.enc += positional_encoding(self.x,
                      num_units=hp.hidden_units, 
                      zero_pad=False, 
                      scale=False,
                      scope="enc_pe")
else:
    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0),
    							 [tf.shape(self.x)[0], 1]),
                      vocab_size=hp.maxlen, 
                      num_units=hp.hidden_units, 
                      zero_pad=False, 
                      scale=False,
                      scope="enc_pe")
```

如果超参数 `hp.sinusoid=True`，使用 `positional_encoding` 函数，通过使用正弦和余弦函数来生成位置编码，可以为输入序列添加位置信息。如果 `hp.sinusoid=False`，使用 `embedding` 函数，通过学习的词嵌入来生成位置编码。

位置编码生成后，用 `key_masks` 处理一下。注意 `key_masks` 的生成一定要用最初的 `self.enc`，所以在前面执行而不是这里：

```python
self.enc *= key_masks
```

这个不是矩阵乘法，而是对应元素相乘。这里乘上 `key_masks` 的目的是将 `key_masks` 中值为 0 的位置对应的 `self.enc` 中的元素置为 0，这样就可以排除这些位置对计算的影响。

###### Drop out

下面调用了 TensorFlow 的 drop out 操作：

```python
self.enc = tf.layers.dropout(self.enc, 
                            rate=hp.dropout_rate, 
                            training=tf.convert_to_tensor(is_training))
```

drop out 是一种在深度学习中常用的正则化技巧。它通过在训练过程中随机地「关闭」一些神经元来减少 **过拟合**。这样做是为了防止模型过于依赖于某些特定的特征，而导致在新数据上的表现不佳。

在这个函数中，`dropout` 层通过在训练过程中随机地将一些神经元的输出值设置为 0，来减少模型的过拟合。这个函数中使用了一个参数 `rate`，表示每个神经元被「关闭」的概率。这样做是为了防止模型过于依赖于某些特定的特征，而导致在新数据上的表现不佳。

###### Encoder Blocks: Multi-Head Attention & Feed Forward

然后看下 encoder blocks 代码：

```python
## Blocks
for i in range(hp.num_blocks):
    with tf.variable_scope("num_blocks_{}".format(i)):
        # 多头注意力
        self.enc = multihead_attention(queries=self.enc, 
                                        keys=self.enc, 
                                        num_units=hp.hidden_units, 
                                        num_heads=hp.num_heads, 
                                        dropout_rate=hp.dropout_rate,
                                        is_training=is_training,
                                        causality=False)
        
        # 前馈神经网络
        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])
```

上述代码是编码器（Encoder）的实现函数调用的流程，也是与船长上面的模型原理介绍一致的，在定义时同样使用了 Embedding 层、Positional Encoding 层、Dropout 层、Multihead Attention 和 Feed Forward 操作。其中 Multihead Attention 在编码、解码中是不一样的，待会儿我们会在 Decoder 部分再提到，有自注意力层和 Encoder-Decoder 层。

* 超参数 hp.num_blocks 表示 Encoder Blocks 的层数，每一层都有一个 Multi-Head Attention 和一个 Feed Forward。
* 这个 Encoder 中的 Multi-Head Attention 是基于自注意力的（注意与后面的 Decoder 部分有区别）
* `causality` 参数的意思是否使用 Causal Attention，它是 Self-Attention 的一种，但是只使用过去的信息，防止模型获取未来信息的干扰。一般对于预测序列中的某个时间步来说，只关注之前的信息，而不是整个序列的信息。这段代码中 `causality` 设置为了 `False`，即会关注整个序列的信息。

##### 2.4.2、解码过程

再看一下解码的流程：

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	splines=ortho
	node [shape="box"]
	decoder_attn1 [label="multihead_attention (self-attention)"]
	decoder_attn2 [label="multihead_attention (encoder-decoder attention)"]

	embedding -> positional_encoding -> dropout -> decoder_attn1 -> decoder_attn2 -> feedforward
}
{% endgraphviz %}
</div>

###### Embedding

下面我们逐一看每段代码，主要关注与编码阶段的区别即可：

```python
self.dec = embedding(self.decoder_inputs, 
                      vocab_size=len(en2idx), 
                      num_units=hp.hidden_units,
                      scale=True, 
                      scope="dec_embed")
```

* `embedding` 输入用的是 `self.decoder_inputs`
* 词汇表尺寸用翻译后的输出语言英语词汇表长度 `len(en2idx)`

###### Key Masks

```python
key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.dec), axis=-1)), -1)
```

* `key_masks` 输入变量用 `self.dec`。

###### Positional Encoding & Drop out

```python
# 位置编码
if hp.sinusoid:
    self.dec += positional_encoding(self.decoder_inputs,
                      vocab_size=hp.maxlen, 
                      num_units=hp.hidden_units, 
                      zero_pad=False, 
                      scale=False,
                      scope="dec_pe")
else:
    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
    							 [tf.shape(self.decoder_inputs)[0], 1]),
                      vocab_size=hp.maxlen, 
                      num_units=hp.hidden_units, 
                      zero_pad=False, 
                      scale=False,
                      scope="dec_pe")

self.dec *= key_masks

self.dec = tf.layers.dropout(self.dec, 
                            rate=hp.dropout_rate, 
                            training=tf.convert_to_tensor(is_training))
```

* 输入 `self.decoder_inputs`
* 指定 `vocab_size` 参数 `hp.maxlen`

###### Decoder Blocks: Multi-Head Attention & Feed Forward

```python
## 解码器模块
for i in range(hp.num_blocks):
    with tf.variable_scope("num_blocks_{}".format(i)):
        # 多头注意力（自注意力）
        self.dec = multihead_attention(queries=self.dec, 
                                        keys=self.dec, 
                                        num_units=hp.hidden_units, 
                                        num_heads=hp.num_heads, 
                                        dropout_rate=hp.dropout_rate,
                                        is_training=is_training,
                                        causality=True, 
                                        scope="self_attention")
        
        # 多头注意力（Encoder-Decoder 注意力）
        self.dec = multihead_attention(queries=self.dec, 
                                        keys=self.enc, 
                                        num_units=hp.hidden_units, 
                                        num_heads=hp.num_heads,
                                        dropout_rate=hp.dropout_rate,
                                        is_training=is_training, 
                                        causality=False,
                                        scope="vanilla_attention")

        # 前馈神经网络
        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
```

* 在用 `multihead_attention` 函数解码器模块时，注意传入的参数 `scope` 区别，先是自注意力层，用参数 `self_attention`，对应的 `queries` 是 `self.dec`，`keys` 也是 `self.dec`。再是「Encoder-Decder Attention」用的是参数 `vanilla_attention`，对应的 `queries` 来自解码器是 `self.dec`，但 `keys` 来自编码器是是 `self.enc`。

##### 2.4.3、Embedding、Positional Encoding、Multi-Head Attention、Feed Forward

###### Embedding 函数实现

```python
def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):

    	# 创建一个名为 `lookup_table`、形状为 (vocab_size, num_units) 的矩阵
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())

        # lookup_table 的第一行插入一个全零行，作为 PAD 的词向量
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

        # 在词向量矩阵 lookup_table 中查找 inputs
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        # 对输出的词向量进行除以根号 num_units 的操作，可以控制词向量的统计稳定性。
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs
```

###### Positional Encoding 函数实现

```python
def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):

    	# tf.range(T) 生成一个 0~T-1 的数组
    	# tf.tile() 将其扩展成 N*T 的矩阵，表示每个词的位置
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # 用 numpy 的 sin 和 cos 函数对每个位置进行编码
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # 将编码结果转为张量
        lookup_table = tf.convert_to_tensor(position_enc)

        # 将编码的结果与位置索引相关联，得到最终的位置编码
        if zero_pad:
        	# 如果 zero_pad 参数为 True，则在编码结果的开头添加一个全 0 的向量
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        # scale 参数为 True，则将编码结果乘上 num_units 的平方根
        if scale:
            outputs = outputs * num_units**0.5

        return outputs
```

###### Multi-Head Attention 函数实现

```python
def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        
        # Linear Projections
        # 使用三个全连接层对输入的 queries、keys 分别进行线性变换，将其转换为三个维度相同的张量 Q/K/V
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        # 按头数 split Q/K/V，再各自连接起来
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        # 计算 Q_, K_, V_ 的点积来获得注意力权重
        # 其中 Q_ 的维度为 (hN, T_q, C/h)
        # K_ 的维度为 (hN, T_k, C/h)
        # 计算出来的结果 outputs 的维度为 (h*N, T_q, T_k)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

        # Scale
        # 对权重进行 scale，这里除以了 K_ 的第三维的平方根，用于缩放权重
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        # 这里需要将 keys 的有效部分标记出来，将无效部分设置为极小值，以便在之后的 softmax 中被忽略
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:

        	# 创建一个与 outputs[0, :, :] 相同形状的全 1 矩阵
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)

            # 对 diag_vals 进行处理，返回一个下三角线矩阵
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
   			# 将 masks 为 0 的位置的 outputs 值设置为一个非常小的数
   			# 这样会导致这些位置在之后的计算中对结果产生非常小的影响，从而实现了遮盖未来信息的功能
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # 对于每个头的输出，应用 softmax 激活函数，这样可以得到一个概率分布
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        # 对于查询（queries）进行 masking，这样可以避免输入序列后面的词对之前词的影响
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1)) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts & Weighted Sum
        # 对于每个头的输出，应用 dropout 以及进行残差连接
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        # 将每个头的输出拼接起来，使用 tf.concat 函数，将不同头的结果按照第二维拼接起来
        # 得到最终的输出结果，即经过多头注意力计算后的结果
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs
```

###### Feed Forward 函数实现

下面是 **前馈神经网络层** 的定义，这是一个非线性变换，这里用到了一些卷积神经网络（CNN）的知识，我们来看下代码再解释：

```python
def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # 连接一个残差网络 ResNet
        outputs += inputs
        
        # 归一化后输出
        outputs = normalize(outputs)
    
    return outputs
```

* 先是使用了一个卷积层（conv1d）作为 inner layer、一个卷积层作为 readout layer，卷积核大小都为 1。
* `filters` 参数用来控制卷积层中输出通道数量，inner layer 的输出通道数设置为 `num_units[0]` ，readout layer 的设置为 `num_units[1]`。有时也会把这个解释为神经元数量。这两个的默认分别为 2048、512，调用时传入的是超参数的 `[4 * hidden_units, hidden_units]`。
* 其中 inner layer 用 `ReLU` 作为激活函数，然后连接一个残差网络 RedNet，把 readout layer 的输出加上原始的输入。
* 最后使用 `normalize` 归一化处理输出，再返回。下面来看下 `normalize` 函数。

```python
def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):

    	# 输入数据的形状
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
    	# 平均数、方差
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        # 拉伸因子 beta
        beta= tf.Variable(tf.zeros(params_shape))

        # 缩放因子 gamma
        gamma = tf.Variable(tf.ones(params_shape))

        # 归一化：加上一个非常小的 epsilon，是为了防止除以 0
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )

        outputs = gamma * normalized + beta
        
    return outputs
```

* 该函数实现了 Layer Normalization，用于在深度神经网络中解决数据的不稳定性问题。

##### 2.4.4、编码和解码完成后的操作

解码器后的 `Linear & Softmax`：

```python
# 全连接层得到的未经过归一化的概率值
self.logits = tf.layers.dense(self.dec, len(en2idx))

# 预测的英文单词 idx
self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
self.istarget = tf.to_float(tf.not_equal(self.y, 0))

# 正确预测数量，除以所有样本数，得到准确率
self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))

#  记录了模型的准确率的值，用于 tensorboard 可视化
tf.summary.scalar('acc', self.acc)
```

训练集数据处理时，经过 `Linear & Softmax` 之后的最后处理如下。这里用到了 `tf.nn.softmax_cross_entropy_with_logits` 交叉熵损失，来计算模型的错误率 `mean_loss`，并使用 Adam 优化器 `AdamOptimizer` 来优化模型参数。

```python
# 使用 label_smoothing 函数对真实标签进行标签平滑，得到 self.y_smoothed
self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
```

下面这段代码实现了一种叫做「label Smoothing」的技巧。

```python
def label_smoothing(inputs, epsilon=0.1):

	# 获取输入的类别数，并将其赋值给变量 K
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
```

在训练过程中，样本的标签被表示为一个二维矩阵，其中第一维表示样本的编号，第二维表示样本的标签。这个矩阵的形状就是 (样本数, 类别数)，所以类别数对应的就是最后一维。具体到这个模型用例里，第一个维度是德语样本句子数，最后一维就是英语词汇量的大小。

用于解决在训练模型时出现的过拟合问题。在标签平滑中，我们给每个样本的标签加上一些噪声，使得模型不能完全依赖于样本的标签来进行训练，从而减少过拟合的可能性。具体来说，这段代码将输入的标签 `inputs` 乘上 `1-epsilon`，再加上 `epsilon / K`，其中 `epsilon` 是平滑因子，`K` 是标签类别数（英语词汇量大小）。这样就可以在训练过程中让模型对标签的预测更加平稳，并且降低过拟合的风险。

然后我们看后续的操作。

```python
# 对于分类问题来说，常用的损失函数是交叉熵损失
self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

# Training Scheme
self.global_step = tf.Variable(0, name='global_step', trainable=False)

# Adam 优化器 self.optimizer，用于优化损失函数
self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

# 使用优化器的 minimize() 函数创建一个训练操作 self.train_op，用于更新模型参数。这个函数会自动计算梯度并应用更新
self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
   
# 将平均损失写入 TensorFlow 的 Summary 中，用于 tensorboard 可视化
tf.summary.scalar('mean_loss', self.mean_loss)

# 将所有的 summary 合并到一起，方便在训练过程中写入事件文件
self.merged = tf.summary.merge_all()
```

#### 2.5、效果评价

```python
def eval(): 
    # 创建一个处理测试数据集的 Graph 实例
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # 加载测试数据
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
     
    # Start session         
    with g.graph.as_default():

    	# TensorFlow 中用于管理训练的一个类
    	# 它可以帮助你轻松地管理训练过程中的各种资源，如模型参数、检查点和日志
        sv = tf.train.Supervisor()

        # 创建一个会话
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            # 恢复模型参数
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            # 获取模型名称
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
             
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')

            # 初始化结果文件
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []

                # 循环处理数据
                for i in range(len(X) // hp.batch_size):
                     
                    # 获取小批量数据
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    # 使用自回归推理（Autoregressive inference）得到预测结果
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                     
                    # 将预测结果写入文件
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                          
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                # 计算 BLEU 分数，并将其写入文件
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score))
                                          
if __name__ == '__main__':
    eval()
    print("Done")
```

### 3、Kyubyong Transformer 的性能表现

评估结果文件的最后一行有 Bleu Score = 6.598452846670836 表示这个翻译模型的翻译结果与参考翻译重叠程度比较高，翻译质量较好。不过需要注意的是，BLEU 分数不能完全反映翻译质量，因为它不能评估语法，语义，语调等方面的问题。

另外前面我们在代码中已经将过程数据保存在 logdir 下了，就是为了后续方便可视化，我们可以用 TensorBoard 来可视化，具体使用方法如下：

```shell
mikecaptain@local $ tensorboard --logdir logdir
```

然后在浏览器里查看 `http://localhost:6006`，示例如下：

![image](/img/src/2023-01-04-language-model-5-17.gif)

### 4、Kyubyong Transformer 模型的一些问题

我们可以看到这个 Transformer 能够较好地捕捉长距离依赖关系，提高翻译质量。然而，Kyubyong Transformer 的实现存在一些问题。该 Transformer 模型在训练过程中还需要调整许多超参数，如学习率（learning rate）、batch size 等，不同的任务可能需要不同的超参数调整。

## 参考

* https://arxiv.org/abs/1706.03762
* https://arxiv.org/abs/1512.03385
* https://github.com/Kyubyong/transformer/
* http://jalammar.github.io/illustrated-transformer/
* https://towardsdatascience.com/this-is-how-to-train-better-transformer-models-d54191299978
* 《自然语言处理：基于预训练模型的方法》车万翔 等著
* 《自然语言处理实战：预训练模型应用及其产品化》安库·A·帕特尔 等著
* https://lilianweng.github.io/posts/2018-06-24-attention/
* https://github.com/lilianweng/transformer-tensorflow/
* 《基于深度学习的道路短期交通状态时空序列预测》崔建勋 著
* https://www.zhihu.com/question/325839123
* https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer
* 《Python 深度学习（第 2 版）》弗朗索瓦·肖莱 著
* https://en.wikipedia.org/wiki/Attention_(machine_learning)
* https://zhuanlan.zhihu.com/p/410776234
* https://www.tensorflow.org/tensorboard/get_started
* https://paperswithcode.com/method/multi-head-attention
* https://zhuanlan.zhihu.com/p/48508221
* https://www.joshbelanich.com/self-attention-layer/
* https://learning.rasa.com/transformers/kvq/
* https://zhuanlan.zhihu.com/p/352898810
* https://towardsdatascience.com/beautifully-illustrated-nlp-models-from-rnn-to-transformer-80d69faf2109
* https://medium.com/analytics-vidhya/understanding-q-k-v-in-transformer-self-attention-9a5eddaa5960