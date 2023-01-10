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
---

**本文目录**
* TOC
{:toc}

### 一、为什么说 RNN 模型没有体现「注意力」？
 
Encoder-Decoder 的一个非常严重的问题，是依赖中间那个 context 向量，则无法处理特别长的输入序列 —— 记忆力不足，会忘事儿。而忘事儿的根本原因，是没有「注意力」。

对于一般的 RNN 模型，Encoder-Decoder 结构并没有体现「注意力」—— 这句话怎么理解？当输入序列经过 Encoder 生成的中间结果（上下文 C），被喂给 Decoder 时，这些中间结果对所生成序列里的哪个词，都没有区别（没有特别关照谁）。这相当于在说：输入序列里的每个词，对于生成任何一个输出的词的影响，是一样的，而不是输出某个词时是聚焦特定的一些输入词。这就是模型没有注意力机制。

人脑的注意力模型，其实是资源分配模型。NLP 领域的注意力模型，是在 2014 年被提出的，后来逐渐成为 NLP 领域的一个广泛应用的机制。可以应用的场景，比如对于一个电商平台中很常见的白底图，其边缘的白色区域都是无用的，那么就不应该被关注（关注权重为 0）。比如机器翻译中，翻译词都是对局部输入重点关注的。

所以 Attention 机制，就是在 Decoder 时，不是所有输出都依赖相同的「上下文 {% raw %} $$ \bm{C}_t $$ {% endraw %}」，而是时刻 t 的输出，使用 {% raw %} $$ \bm{C}_t $$ {% endraw %}，而这个 {% raw %} $$ \bm{C}_t $$ {% endraw %} 来自对每个输入数据项根据「注意力」进行的加权。

### 二、基于 Attention 机制的 Encoder-Decoder 模型

2015 年 Dzmitry Bahdanau 等人在论文[《Neural Machine Translation by Jointly Learning to Align and Translate》](https://arxiv.org/abs/1409.0473) 中提出了「Attention」机制，下面请跟着麦克船长，我会深入浅出地为你解释清楚。

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

更进一步细化关于 {% raw %} $$ \bm{C}_t $$ {% endraw %} 部分，我们引用《基于深度学习的道路短期交通状态时空序列预测》一书中的图：

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

还记得 RNN 那部分里我们讲到的 Encoder-Decoder 模型的公式表示吗？

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

加入 Attention 机制的 Encoder-Decoder 模型如下：

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

可以看到最核心的区别是第二个公式 {% raw %} $$ C_t $$ {% endraw %}。加入 Attention 后，对所有数据给予不同的注意力分布。具体地，比如我们用如下的函数来定义这个模型：

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

### 三、Transformer 在 2017 年横空出世

中文网络里找到的解释得比较好的 blogs、answers，几乎都指向了同一篇博客：Jay Alammar 的[《The Illustrated Transformer》](http://jalammar.github.io/illustrated-transformer/)，所以建议读者搭配该篇文章阅读。

Transformer 模型中用到了自注意力（Self-Attention）、多头注意力（Multiple-Head Attention）、残差网络（ResNet）与捷径（Short-Cut）。下面我们先通过第 1 到第 4 小节把几个基本概念讲清楚，然后在第 5 小节讲解整体 Transformer 模型就会好理解很多了。最后第 6 小节我们来一段动手实践。

#### 1、自注意力机制（Self-Attention）

自注意力是理解 Transformer 的关键，原作者在论文中限于篇幅，没有给出过多的解释。以下是我自己的理解，能够比较通透、符合常识地去理解 Transformer 中的一些神来之笔的概念。

##### 1.1、一段自然语言内容，其自身就「暗含」很多内部关联信息

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

##### 1.2、如何计算 Q、K、V

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

##### 1.3、注意力函数：如何通过 Q、V 得到 Z

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

##### 1.4、其他注意力函数

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

#### 2、多头注意力

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

我们再训练一个权重矩阵 {% raw %} $$ W^O $$ {% endraw %}，然后用上面拼接的 {% raw %} $$ Z_{0~7} $$ {% endraw %} 乘以这个权重矩阵：

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

#### 3、退化现象、残差网络与 Short-Cut

##### 3.1、退化现象

对于一个 56 层的神经网路，我们很自然地会觉得应该比 20 层的神经网络的效果要好，比如说从误差率（error）的量化角度看。但是华人学者何凯明等人的论文[《Deep Residual Learning for Image Recognition》](https://arxiv.org/pdf/1512.03385.pdf)中给我们呈现了相反的结果，而这个问题的原因并不是因为层数多带来的梯度爆炸/梯度消失（毕竟已经用了归一化解决了这个问题），而是因为一种反常的现象，这种现象我们称之为「退化现象」。何凯明等人认为这是因为存在「难以优化好的网络层」。

##### 3.2、恒等映射

如果这 36 层还帮了倒忙，那还不如没有，是不是？所以这多出来的 36 个网络层，如果对于提升性能（例如误差率）毫无影响，甚至更进一步，这 36 层前的输入数据，和经过这 36 层后的输出数据，完全相同，那么如果将这 36 层抽象成一个函数 {% raw %} $$ f_{36} $$ {% endraw %}，这就是一个恒等映射的函数：

{% raw %} $$ f_{36}(x) = x $$ {% endraw %}

回到实际应用中。如果我们对于一个神经网络中的连续 N 层是提升性能，还是降低性能，是未知的，那么则可以建立一个跳过这些层的连接，实现：

> 如果这 N 层可以提升性能，则采用这 N 层；否则就跳过。

这就像给了这 N 层神经网络一个试错的空间，待我们确认它们的性能后再决定是否采用它们。同时也可以理解成，这些层可以去单独优化，如果性能提升，则不被跳过。

##### 3.3、残差网络（Residual Network）与捷径（Short-Cut）

如果前面 20 层已经可以实现 99% 的准确率，那么引入了这 36 层能否再提升「残差剩余那 1%」的准确率从而达到 100% 呢？所以这 36 层的网络，就被称为「残差网络（Residual Network，常简称为 ResNet）」，这个叫法非常形象。

而那个可以跳过 N 层残差网络的捷径，则常被称为 Short-Cut，也会被叫做跳跃链接（Skip Conntection），这就解决了上述深度学习中的「退化现象」。

#### 4、位置编码（Positional Embedding）

还记得我在第二部分最后提到的吗：

> 这个注意力机制忽略了位置信息。比如 Tigers love rabbits 和 Rabbits love tigers 会产生一样的注意力分数。

##### 4.1、Transformer 论文中的三角式位置编码（Sinusoidal Positional Encoding）

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

##### 4.2、绝对位置编码

绝对位置编码，如上面提到的，就是定义一个位置编码向量 {% raw %} $$ t_i $$ {% endraw %}，通过 {% raw %} $$ x_i + t_i $$ {% endraw %} 就得到了一个含有位置信息的向量。

* 习得式位置编码（Learned Positional Encoding）：将位置编码当做训练参数，生成一个「最大长度 x 编码维度」的位置编码矩阵，随着训练进行更新。目前 Google BERT、OpenAI GPT 模型都是用的这种位置编码。缺点是「外推性」差，如果文本长度超过之前训练时用的「最大长度」则无法处理。目前有一些给出优化方案的论文，比如「[层次分解位置编码](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247515573&idx=1&sn=2d719108244ada7db3a535a435631210&chksm=96ea6235a19deb23babde5eaac484d69e4c2f53bab72d2e350f75bed18323eea3cf9be30615b#rd)」。
* 三角式位置编码（Sinusoidal Positional Encodign）：上面提过了。
* 循环式位置编码（Recurrent Positional Encoding）：通过一个 RNN 再接一个 Transformer，那么 RNN 暗含的「顺序」就导致不再需要额外编码了。但这样牺牲了并行性，毕竟 RNN 的两大缺点之一就有这个。
* 相乘式位置编码（Product Positional Encoding）：用「{% raw %} $$ x_i \odot t_i $$ {% endraw %}」代替「{% raw %} $$ x_i + t_i $$ {% endraw %}」。

##### 4.3、相对位置编码

最早来自于 Google 的论文[《Self-Attention with Relative Position Representations》](https://arxiv.org/abs/1803.02155)相对位置编码，考虑的是当前 position 与被 attention 的 position 之前的相对位置。

* 经典式
* XLNET 式
* T5 式
* DeBERTa 式

##### 4.4、其他位置编码

* CNN 式
* 复数式
* 融合式

#### 5、Transformer 模型整体

——> 未完待续

最后我们再来整体看一下 Transformer：

* 首先输入数据生成词的 Embedding、位置编码
* 在 Encoder 里，先进入 N 层 Attention 的处理，最后进入一个全连接层，期间可能有 Short-Cut
* 然后经过 Normalization

——> 未完待续

#### 6、来看一段用 PyTorch 实现的 Transformer 示例

### 参考

* http://jalammar.github.io/illustrated-transformer/
* 《自然语言处理：基于预训练模型的方法》车万翔 等
* 《自然语言处理实战：预训练模型应用及其产品化》安库·A·帕特尔 等
* https://lilianweng.github.io/posts/2018-06-24-attention/
* 《基于深度学习的道路短期交通状态时空序列预测》
* https://www.zhihu.com/question/325839123
* https://zhuanlan.zhihu.com/p/410776234
* https://zhuanlan.zhihu.com/p/48508221
* https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer
* https://zhuanlan.zhihu.com/p/352898810