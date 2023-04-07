---
layout: post
title:  麦克船长的 GPT-1 解读
date:   2023-01-23 23:56:03 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, AIGC, Transformer, GPT, OpenAI]
description: 
excerpt: 
location: 杭州
author: 麦克船长
---

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-34.png)

由于 GPT 系列模型已经成为当下阶段 LLM 领域的绝对领导者，因此本文中关于 GPT 系列的每个模型细节的着墨都会稍多一些。

### 1、GPT 出现的背景：监督学习模型为主，但标注和泛化成为卡点

在介绍 GPT 之前，我们说一下它出现的那个时间点的 NLP 模型发展情况。那时 NLP 领域绝大多数的 SOTA 模型，都是针对特定类型任务进行监督学习训练得到的，而监督学习模型有两个严重的限制：

1. **标注成本极高**：面对特定任务，需要大量的标注数据用于训练。
2. **泛化能力极差**：除了训练过的特定任务，模型很难泛化去做其他任务。

这两个问题不解决，AI 技术在 NLP 领域很难带来应用的广泛性，更不要提准确性问题了。但是 GPT 的出现，拉开了 NLP 领域预训练大模型对任务大一统的大幕。而后 T5 模型则明确提出了这个断言，而后续 GPT-3 基本实现了这一点。

OpenAI 官方所称的 GPT，根据其官方定义是 Generative Pre-trained Transformer 缩写，这里大家注意并不是一些文中误传的 Generative Pre-Trained，因为架构的核心理念是 Transformer。

### 2、GPT-1：基于 Transformer Decoder 的自监督训练语言模型

在 2018 年 1 月，Google Brain 团队在文章[《Generating Wikipedia by Summarizing Long Sequences》](https://arxiv.org/abs/1801.10198)中提出了一种基于 Transformer 改进，但只有 Decoder 架构的模型，也可以用于构建语言模型。相应地，因为没有 Encoder，这种架构里自然去掉了 Decoder 中的 Encoder-Decoder 注意力层。Google AI Language 团队在[《Character-Level Language Modeling with Deeper Self-Attention》](https://arxiv.org/abs/1808.04444)论文的工作中验证了用类似架构构建的语言模型可以逐一生成词。

OpenAI 在 2018 年 6 月其博客上发布一篇名为[《Improving Language Understanding with Unsupervised Learning》](https://openai.com/blog/language-unsupervised/)的文章提出了一种模型，该模型的打造方法包括**生成式预训练（Generative Pre-training）**和**判别式微调（Discriminative Fine-tuning）**两个关键阶段，并在一系列不同语言任务上获得了 SOTA 的结果。其实这种方法，早在 GPT-1 推出几年前在 CV（计算机视觉）领域就已经很主流了，其中预训练环节在 CV 领域用的都是大名鼎鼎的 ImageNet。而因为 NLP 领域没有类似 ImageNet 这种海量标注数据可用。最初，大家甚至不知道该称呼它叫什么，所以那段时间提到它的相关文章里你会看到 **Fine-tune Transformer**、**Fine-tuned Transformer**、**OpenAI Transformer** 等等，后来 GPT-2 推出后，大家也就叫 2018 年这个为 GPT-1 了，下文也用该说法。

<!-- 主要讲的是：在 Google 的 Transformer 架构之上采用无监督预训练方法得到了一个可扩展的、任务无偏（也叫任务无关）的语言模型 -->

GPT-1 与当时主流的 NLP 模型最大的区别是什么？

首先是基础架构（Architecture），与 Google Brain 团队 2018 年 1 月提出的模式一样，下图是 GPT 采用的 Transformer 模型变体，也用的是 Transformer Decoder，同样因为没有 Encoder 自然也就移除了 Encoder-Decoder Attention，而只采用多头自注意力（Multi-headed Self-Attention，关于此的介绍可见本文的前篇[《人工智能 LLM 革命前夜：一文读懂横扫自然语言处理的 Transformer 模型》](https://www.mikecaptain.com/2023/01/22/captain-aigc-1-transformer/)的第 7、8 节），中间的 transformer blocks 一共用了 12 层（作为对比，后续的 GPT-2、GPT-3 分别达到了 48 层、96 层）。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-41.png){: width="200"}

其次是核心方法（Method），GPT-1 采用的是「无监督预训练 + 监督精调」，当时 OpenAI 把这种方法仍然归类为半监督学习，但后来学界和业界都把这种叫做「自监督学习（Self-Supervised Learning）」，后面要介绍的 BERT 也采用这种方法。

关于架构和这两段式的训练方法，我们分四个小节来看下。

### 3、GPT-1 为什么用 Transformer 而不是 LSTM？

回到 2017 年的背景下（虽然是 2018 年初发布，但 GPT-1 的研发是从 2017 年 Transformer 发布后就开始的），哪怕知道 Transformer 是个不错的基础模型，但是当时一个新的语言模型采用 LSTM/RNN 还是 Transformer 并没有现在这么显而易见。为什么选择 Transformer？OpenAI 给出了两个原因。但是船长认为，按学界的通常情况，都是试出了结果再尝试归因，肯定不是分析出哪个模型更有效然后指哪打哪的。这并非吐槽，因为与理论科学不同，实验科学其实就要这样。

我们来看下 OpenAI 给出的归因。

首先一个原因，是 Transformer 有更结构化的记忆可以处理长距离依赖关系（可以理解为更能搞定长文本），这样就意味着不仅是句子维度，甚至段落维度、文章维度上的信息也可以被 Transformer 学习到。

其次，OpenAI 在做迁移学习的时候，TODO

### 4、GPT-1 的无监督预训练（Unsupervised Pre-training）

能用未标注的数据做无监督的预训练，已经很不容易了，尽管 GPT-1 并不是第一个这样做的（2014 年的 Word2Vec 词嵌入模型也是用的大量无标注文本）。下面我们来理解一下无监督预训练的过程。$$ \mathcal{U}={u_1, ..., u_n} $$ 是一个无监督词序列语料，那么语言模型给出这样一个词序列的概率是：

{% raw %}
$$
\begin{aligned}
P(\mathcal{U}) = P(u_1)P(u_2|u_1)P(u_3|u_1,u_2)...p(u_n|u_1,u_{n-1}) = \prod_i^n P(u_i|u_1, ..., u_{i-1})
\end{aligned}
$$
{% endraw %}

如果模型的上下文窗口（Context Windows）大小是 $$ k $$ 的话，则上式可近似转化为：

{% raw %}
$$
\begin{aligned}
P(\mathcal{U}) = \prod_i P(u_i | u_{i-k}, ..., u_{i-1})
\end{aligned}
$$
{% endraw %}

我们的目标就是让这个概率 $$ P(\mathcal{U}) $$ 最大化，因此我们定义一下目标，即最大化对数似然函数。再将模型的参数 $$ \Theta $$ 也考虑进来，则其定义如下：

{% raw %}
$$
\begin{aligned}
L_1(\mathcal{U}) = \sum_{i} \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta)
\end{aligned}
$$
{% endraw %}

明确了上面目标函数后，我们来看下 GPT-1 预训练模型。$$ U = (u_{-k}, ..., u_{-1}) $$ 是考虑了上下文的输入词向量矩阵，$$ W_e $$ 是词嵌入矩阵，$$ W_p $$ 是位置编码（或叫位置嵌入）矩阵。所有隐藏层都是 transformer_block，第一个隐藏层的输入是 $$ h_0 $$，每 i 个隐藏层的输出是 $$ h_i $$。那么 GPT-1 预训练模型可以表示为：

{% raw %}
$$
\begin{aligned}
h_0 &= U W_e + W_p \\
h_l &= \operatorname{transformer\_block}(h_{l-1}) \quad \forall i\in [1, n] \\
P(u) &= \operatorname{softmax}(h_n W_e^T)
\end{aligned}
$$
{% endraw %}

如果你在麦克船长的前篇文章中已经理解了位置编码（Positional Encoding），那么在 GPT-1 模型里，$$ U $$ 经过 $$ W_e $$ 处理后每一行就得到了特征（特征抽取）。文本的有序性决定了「位置」本身就是有信息量的，因此叠加 $$ W_p $$ 则保留了位置相关（position-wise）信息。

以最大化 $$ L_1 $$ 为目标，经过这样学习大量文本语料后，就得到了一个预训练模型。

### 5、GPT-1 的监督微调（Supervised Fine-Tuning，SFT）

现在我们已经有了一个预训练好的模型了，这一步就是要 fine-tune 它的参数，来适应下游的监督学习任务。对于不同的任务，在 fine-tune 阶段将所有任务的输入结构都转换成 token 序列，喂给已经预训练好的模型来 fine-tune，然后再接一个 linear+softmax。流程结构上表达如下：

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-42.png){: width="640"}

设我们有一个标注过的数据集 $$ \mathcal{C} $$，组成它的每个样本都包括一个文本序列 $$ x = x^1, ..., x^m $$ 和一个标签 $$ y $$。微调时，就是把输入 $$ x $$ 经过预训练模型后在最后一个 Decoder 输出的 $$ y $$，进行线性变换和 softmax 回归：

{% raw %}
$$
\begin{aligned}
P(y|x^1, ..., x^m) = \operatorname{softmax}(y W_y)
\end{aligned}
$$
{% endraw %}

这个过程中，就学习到了 $$ W_l \in \mathbb{R}^{m\times c} $$ 参数矩阵，其中 $$ c $$ 是下游任务目标类别的数量，比如情感分类（positive、neutral、negative）的 $$ c $$ 为 3。在模型使用时，最后得到的 $$ \operatorname{softmax}(y W_y) $$ 就能得到一组关于目标类别的概率分布了，其中最大的概率值即可看做是结果。

监督微调的目标，也是最大化对数似然函数：

{% raw %}
$$
\begin{aligned}
L_2(\mathcal{C}) = \sum_{(x,y)}\log P(y|x^1, ..., x^m)
\end{aligned}
$$
{% endraw %}

这样整体看，我们把两个训练过程（无监督预训练、监督训练）联合起来。其中在无监督预训练过程中，我们顺手完成了语言建模，它其实相当于我们的一个辅助目标。我们发现这个辅助目标有两个好处：

1. 提升了监督模型的泛化能力；
2. 加速模型收敛。

这并非 GPT-1 首次提出，此前剑桥大学的 Marek Rei 在 2017 年 4 月 24 日发表的论文[《Semi-supervised Multitask Learning for Sequence Labeling》](https://arxiv.org/abs/1704.07156)中就得出过同样的结论：

> We found that the additional language modeling objective provided consistent performance improvements on every benchmark.

同样在 2017 年，AI2 的 Matthew E. Peters 等四位学者在 4 月 29 日发表的论文[《Semi-supervised sequence tagging with bidirectional language models》](https://arxiv.org/abs/1705.00108) 中也提到了半监督预训练一个语言模型后在 NER 和 Chunking 数据集上都有显著的表现提升：

> In this paper, we proposed a simple and general semi-supervised method using pre-trained neural language models to augment token representations in sequence tagging models. Our method significantly outperforms current state of the art models in two popular datasets for NER and Chunking.

在这样的「无监督预训练 + 监督训练」方法下，目标函数就是最大化下面这个组合（引入一个 $$ \lambda $$ 超参数控制无监督预训练权重）：

{% raw %}
$$
\begin{aligned}
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda * L_1(\mathcal{C})
\end{aligned}
$$
{% endraw %}

以上整个架构与方法，为后来 GPT 的发展确定了基本的模式，甚至包括后来的商业化。OpenAI 在 2020 年 6 月开放了 GPT 的 API（不过那时候已经不是 GPT-1 了）后，其实提供的是预训练后的模型，另外还给开发者提供了 SFT 的 API。

### 6、GPT-1 的预训练数据集

在最初 GPT-1 的论文中，对于预训练数据集的来源和内容，只轻描淡写地提了一句：

> We use the BooksCorpus dataset of training the language model.

论文中所说的这个 BooksCorpus，其实是把 BookCorpus 拼写错误了。BookCorpus 也被称为 Toronto BookCorpus，是一个包含未出版的、免费的书籍内容的数据集，这些书籍都是来自 SmashWords 电子书网站（这个网站自称是全球最大独立电子书分销商，独立电子书的概念可以类比独立电影、独立游戏）。2018 年 OpenAI 训练 GPT-1 时，OpenAI 称该数据集包含 7000 多本未出版的书籍，4.6 GB 数据。

为什么用未出版的小说书籍训练呢？在船长的理解里，这是为了在一个相对隔离的数据集上训练，然后在真实世界中我们可能遇到的问题上做测试，这样可以更好地检验模型的泛化能力。因为 BookCorpus 这些书都是未公开的，而且小说又不像其他书籍，理论上构成的语料也都是原创性的，这样就能更好地检验泛化能力。

在 2021 年时，一份对 BookCorpus 当时 11038 本书籍各类目分布的统计分析如下（来自 Alan D. Thompson, March 2022, What'S IN MY AI? ），从这个分布里大概能推测 GPT-1 用其中 7000 本书都学了什么：

|    |**书籍类别** |**书籍数**|**占比**| ||    |**书籍类别** |**书籍数**|**占比**| |
| 1  |言情        | 2880    | 26.1% |  || 11 | Adventure   | 390   | 3.5%  |
| 2  |Fantasy    | 1502     | 13.6% | || 12 | Other       | 360   | 3.3%  |
| 3  |科幻        | 823     | 7.5%  |  || 13 | Literature  | 330   | 3.0%  |
| 4  |New Adult   | 766    | 6.9%   | || 14 | Humor       | 265   | 2.4%  |
| 5  |Young Adult | 758    | 6.8%   | || 15 | 历史        | 178    | 1.6%  |
| 6  |Thriller    | 646    | 5.9%   | || 16 | Themes      | 51    | 0.5%   |
| 7  |Mystery     | 621    | 5.6%   | ||    |**Total**   |**11038**|**100%**|
| 8  |吸血鬼       | 600    | 5.4%   |
| 9  |Horror      | 448    | 4.1%   | || 
| 10 |Teen        | 430    | 3.9%   | || 

### 7、小结

从性能表现上来看，在如下这些数据集上的表现大都是超越此前的模型的：

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-45.png){: width="720"}

GPT-1 给 NLP 领域带来了两个重要启示与指引：

1. GPT-1 基于 Transformer 架构，不同于当时主流模型采用的 LSTM。具体说，是 Transformer 的 Decoder 部分，并移除 Transformer 定义的 Encoder-Decoder Attention（毕竟没有 Encoder）。这样的架构，先天地可以实现无监督训练，让世界上所有自然语言（甚至代码）都有了成为其语料的可能。
2. 尽管并非 GPT-1 首创，但是它采用自监督学习的训练方法，具体是语言建模的无监督预训练 + 监督微调训练，为模型带来了更强的泛化能力、更快的收敛速度。

虽然那个时间点学界与业界觉得 GPT-1 规模不小，但现在回头看它的各维度都还不算暴力，其的基本信息如下：

<div class="table-wrapper" markdown="block">

| **模型**               | GPT-1   |
| **发布时间**            | 2018 年 6 月|
| **参数量**              | 1.17 亿  |
| **预训练数据量**         | 4.6 GB    |
| **层数**                | 12      |
| **上下文 tokens 数限定** | 512     |
| **词向量长度**           | 768     |
| **开源/闭源**		      | 开源    |

</div>

<br/>

另外，OpenAI 也公布了 [GPT-1 的源码和训练好的模型](https://github.com/openai/finetune-transformer-lm)，那时的 OpenAI 还是很 Open 的。

而就在 OpenAI 发布 GPT-1 后没多久，提出 Transformer 模型的 Google 发布了后来几年产生深远影响的、基于 Transformer Encoder 架构的语言模型 —— **BERT**。

### 参考

* https://openai.com/blog/language-unsupervised/
* https://transformer.huggingface.co/doc/gpt
* https://huggingface.co/docs/transformers/model_doc/openai-gpt
* https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
* https://lifearchitect.ai/whats-in-my-ai/
* https://zhuanlan.zhihu.com/p/362929145
* https://medium.com/the-artificial-impostor/notes-improving-language-understanding-by-generative-pre-training-4c9d4214369c
* https://arxiv.org/abs/1705.00108
* https://arxiv.org/abs/1704.07156