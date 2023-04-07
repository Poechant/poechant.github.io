---
layout: post
title:  麦克船长 NLP 语言模型技术笔记 1：N 元文法（N-Gram）
date:   2023-01-18 05:14:01 +0800
categories: ai
tags: [AI, 人工智能, n-gram, N元文法]
description: 
excerpt: 
katex: True
location: 香港
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

NLP 的技术基础部分，我认为主要是这两部分：词表示法（Word Presentation）、语言模型（Language Model）。对于词表示法，这里不做详细介绍，基本的思路就是把词表示为向量（一维张量），最基本的 One-Hot、Word2Vec、GloVe、fastText 等。这部分的技术演进也在不断前进，比如在第 5 篇文章中要介绍的 Transformer 模型里，用到的词表示法是「引入上下文感知的词向量」。

语言模型的发展中，有两个关键阶段。第一个阶段是以 N 元文法为代表的经典模型。第二个阶段是基于神经网络发展的语言包模型。本文先和大家讲下这个经典模型。后续文章将逐一为大家介绍神经网络的 MLP、CNN、RNN 等。

这里先介绍一个词：超参数。一般 NLP 模型的参数都是海量的，比如 OpenAI 的 GPT-3 参数就达到了 1750 亿个。而在模型训练前就要从宏观上整体设定的一些参数，被称为「超参数（hyperparameter）」。

### 一、N 元文法语言模型（N-gram Language Model）介绍

下一个词出现的概率只依赖于它前面 n-1 个词，这种假设被称为「马尔科夫假设（Markov Assumption」。N 元文法，也称为 N-1 阶马尔科夫链。

* 一元文法（1-gram），unigram，零阶马尔科夫链，不依赖前面任何词；
* 二元文法（2-gram），bigram，一阶马尔科夫链，只依赖于前 1 个词；
* 三元文法（3-gram），trigram，二阶马尔科夫链，只依赖于前 2 个词；
* ……

通过前 t-1 个词预测时刻 t 出现某词的概率，用最大似然估计：

{% raw %}
$$
P(w_t | w_1,w_2...w_{t-1}) = \frac{C(w_1,w_2,...w_t)}{C(w_1,w_2,...w_{t-1})}
$$
{% endraw %}

进一步地，一组词（也就是一个句子）出现的概率就是：

{% raw %}
$$
P(w_1,w_2,...w_t) = P(w_t | w_1,w_2,...w_{t-1}) \cdot P(w_{t-1} | w_1,w_2,...w_{t-2}) \cdot ... \cdot P(w_1)
			      = \displaystyle\prod_{i=1}^{t-1}P(w_i | w_{1:i-1})
$$
{% endraw %}

为了解决句头、尾逇概率计算问题，我们再引入两个标记 `<BOS>` 和 `<EOS>` 分别表示 beginning of sentence 和 end of sentence，所以 {% raw %} $$ w_0 = $$ {% endraw %}\<BOS\>、{% raw %} $$ w_{length + 1} = $$ {% endraw %}\<EOS\>，其中 length 是词的数量。

具体地，比如对于 bigram，该模型表示如下：

{% raw %}
$$
\begin{aligned}
P(w_1,w_2,...w_t) &= \displaystyle\prod_{i=1}^{t-1}P(w_i | w_{i-1}) \\
P(w_t | w_{t-1}) &= \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}
\end{aligned}
$$
{% endraw %}

* 如果有词出现次数为了 0，这一串乘出来就是 0 了，咋办？
* 因为基于马尔科夫假设，所以 N 固定窗口取值，对长距离词依赖的情况会表现很差。
* 如果把 N 值取很大来解决长距离词依赖，则会导致严重的数据稀疏（零频太多了），参数规模也会急速爆炸（高维张量计算）。

上面的第一个问题，我们引入「平滑技术」来解决，而后面两个问题则是在神经网络模型出现后才更好解决的。

### 二、平滑技术：解决零概率问题

虽然限定了窗口 n 大小降低了词概率为 0 的可能性，但当 n-gram 的 n 比较大的时候会有的未登录词问题（Out Of Vocabulary，OOV）。另一方面，训练数据很可能也不是 100% 完备覆盖实际中可能遇到的词的。所以为了避免 0 概率出现，就有了让零平滑过渡为非零的补丁式技术出现。

最简单的平滑技术，就是折扣法（Discounting）。这是一个非常容易想到的办法，就是把整体 100% 的概率腾出一小部分来，给这些零频词（也常把低频词一起考虑）。基于此发展出来一些具体的方法如下：

#### 1、加 1 平滑 / 拉普拉斯平滑（Add-One Discounting、Laplace Smoothing）

加 1 平滑，就是直接将所有词汇的出现次数都 +1，不止针对零频词、低频词。如果继续拿 bigram 举例来说，模型就会变成：

{% raw %}
$$
P(w_i | w_{i-1}) = \frac{C_(w_{i-1},w_i) + 1}{\displaystyle\sum_{j=1}^n(C_(w_{i-1},w_j) + 1)} = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |\mathbb{V}|}
$$
{% endraw %}

其中 {% raw %} $$ N $$ {% endraw %} 表示所有词的词频之和，{% raw %} $$ |\mathbb{V}| $$ {% endraw %} 表示词汇表的大小。

如果当词汇表中的词，很多出现次数都很小，这样对每个词的词频都 +1，结果的偏差影响其实挺大的。换句话说，+1 对于低频词很多的场景，加的太多了，应该加一个更小的数（ 1 \< δ \< 1）。所以有了下面的「δ 平滑」技术。

#### 2、δ 平滑（Delta Smoothing）

把 +1 换成 δ，我们看下上面 bigram 模型应该变成上面样子：

{% raw %}
$$
P(w_i | w{i-1}) = \frac{C_(w_{i-1},w_i) + \delta}{\displaystyle\sum_{j=1}^n(C_(w_{i-1},w_j) + \delta)} = \frac{C(w_{i-1}, w_i) + \delta}{C(w_{i-1}) + \delta|\mathbb{V}|}
$$
{% endraw %}

δ 是一个超参数，确定它的值需要用到困惑度（Perplexity，一般用缩写 PPL）。

#### 3、困惑度（Perplexity）

对于指定的测试集，困惑度定义为测试集中每一个词概率的几何平均数的倒数，公式如下：

{% raw %}
$$
\operatorname{PPL}(\mathbb{D}_{test}) = \frac{1}{\sqrt[n]{P(w_1,w_2...w_n)}}
$$
{% endraw %}

把 {% raw %} $$ P(w_1,w_2,...w_t) = \displaystyle\prod_{i=1}^{t-1}P(w_i\text{\textbar}w_{i-1}) $$ {% endraw %} 带入上述公式，就得到了 PPL 的计算公式：

{% raw %}
$$
\operatorname{PPL}(\mathbb{D}_{test}) = (\displaystyle\prod_{i=1}^nP(w_i|w_{1:i-1}))^{-\frac{1}{n}}
$$
{% endraw %}

## 参考：

* 《自然语言处理：基于预训练模型的方法》车万翔 等
* 《自然语言处理实战：预训练模型应用及其产品化》安库·A·帕特尔 等