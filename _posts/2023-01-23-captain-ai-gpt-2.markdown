---
layout: post
title:  麦克船长的 GPT-2 解读
date:   2023-01-23 23:57:03 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, AIGC, Transformer, GPT, OpenAI]
description: 
excerpt: 
location: 杭州
author: 麦克船长
---

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-36.png)

虽然 BERT 似乎在结果上「打败」了 GPT-1，但是用 Transformer Encoder（更容易）并且数据规模和参数规模都显著提升，多少有点「胜之不武」，OpenAI 自然不服。BERT 发布后又过了 4 个月，OpenAI 发布了比 BERT 更大的 GPT-2，俨然进入了军备竞赛。前面船长介绍 GPT-1 时仍以结构理念、训练方法为主，现在介绍这个扩展自 GPT-1 的 GPT-2，则我们主要以它在结构、方法、数据等方面改进了什么为讲解线索。

### 1、GPT-2 是对 GPT-1 的直接扩展，但更笃定地追逐「通用语言模型」的理想

在 2018 年 6 月，Salesforce Research 团队在论文[《The Natural Language Decathlon: Multitask Learning as Question Answering》](https://arxiv.org/abs/1806.08730 )中提出「**通用 NLP 模型是无法诞生于一个只着眼在单一度量、数据集和任务的范式中的**」，同时提出将 NLP 的十项全能任务（Natural Language Decathlon，decaNLP），主张所有 NLP 任务可以转换成问答任务，并且提出了一个基于该思路的实验模型 MQAN 来挑战 decaNLP，尽管性能表现还有差距。

DeepMind 团队对 Salesforce Research 提出的假设非常认同，并在 2019 年 1 月发表的论文[《Learning and Evaluating General Linguistic Intelligence》](https://arxiv.org/abs/1901.11373)中提到：

> A perfect language model should in theory be able to do any linguistic task.<br/>理论上，完美的语言模型应该能够执行任何语言任务。<br/><br/>We believe that continued progress on generative language models will drive progress on general linguistic intelligence.<br/>我们相信，对生成式语言模型的持续进展，将推动通用语言智能的发展。

OpenAI 也认同这些理念假设，所以期待构建出能解决任何 NLP 任务的语言模型。OpenAI 在 GPT-2 的工作中，把这些理念假设概括为「由于（所训练的）这些任务是通用语言建模的子集，因此我们可以预期随着更多算力和训练数据的使用，性能将进一步提高。」

2019 年的情人节，OpenAI 在其官方发布了一篇 blog[《Better Language Models and Their Implications》](https://openai.com/blog/better-language-models/)，后又发布了介绍 GPT-2 的论文[《Language Models are Unsupervised Multitask Learners》](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。GPT-2 是 GPT-1 的直接扩展，因此还是基于 Transformer Decoder 架构，但是参数扩 10 倍左右，训练数据集扩 10 倍左右。GPT-2 的训练目标也很简单，就是基于一段文本中前面的所有词，预测下一个词。训练方法上，GPT-2 没有对任何任务的任何数据集做针对性的训练，都是直接评估并作为最终结果。

GPT-2 模型的基本信息如下表，其中可以看出 117M 参数版本的 GPT-2 是对标 BERT-Base，15 亿参数版本的 GPT-2 是对标 BERT-Large。

<div class="table-wrapper" markdown="block">

| **模型**                | 参数规模 | 层数 | 词向量长度 | 
| GPT-2 Small            | 117M    | 12  | 768       | 
| GPT-2 Medium           | 345M    | 24  | 1024      |
| GPT-2 Large            | 762M    | 36  | 1280      |
| GPT-2 Extra Large      | 1542M   | 48  | 1600      |
| BERT-Base              |         | 12  | 768       |
| BERT-Large             |         | 24  | 1024      |

</div>
<!-- | **模型**                |GPT-2-1542M|GPT-2-762M |GPT-2-345M | GPT-2-117M| BERT-Large  | BERT-Base   | GPT-1      |
| **发布时间**            |2019 年 2 月|2019 年 2 月|2019 年 2 月|2019 年 2 月|2018 年 10 月|2018 年 10 月| 2018 年 6 月|
| **参数量**              | 15 亿     |            |           | 1.17 亿    |3.4 亿      | 1.1 亿       | 1.17 亿    |
| **预训练数据量**         |  40 GB    |            |           |           |             |             | 4.6 GB     |
| **层数**                |  48       |  36 层     | 24  层    | 12 层      | 24 层       | 12 层       | 12 层      |
| **上下文 tokens 数限定** |  1024     |            |           |            |            |             | 512        |
| **词向量长度**           |  1600     | 1280       | 1024      | 768        | 1024       | 768         | 768       |
| **开源/闭源**            | 开源      | 开源        | 开源       | 开源       | -->

在 AI2 网站上可以在线试用基于 GPT-2 的 Next-Token 的语言模型：[https://demo.allennlp.org/next-token-lm](https://demo.allennlp.org/next-token-lm)。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-53.png){: width="640"}

GPT-2 模型架构在 GPT-1 做了一些优化，如下几点：

* 
* 
* 

但更重要的亮点在于其对更先进训练方法的成功验证。

### 2、GPT-2 大幅改进训练方法

GPT-2 的核心亮点，都体现在其论文标题「Language Models are Unsupervised Multitask Learners」中。第一个亮点即「Unsupervised」，可不只是预训练过程无监督，整个学习过程都可以无监督。第二个亮点是「Multitask」，在无监督的情况下还可以把多种不同的任务混合起来学。

#### 2.1、Zero-Shot：无需监督微调即可执行下游任务，不用 fine-tune

GPT-2 的首个重要改进，就是其论文摘要中的前两句话总结：

> Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText.

GPT-1 及当时所有语言模型的局限性在于，即便采取无监督预训练，仍然需要对特定任务进行监督微调。而 OpenAI 在 GPT-2 上验证了基于数以百万计网页上的无监督学习后就可以执行多种语言任务，比如问答、及其翻译、阅读理解、文本摘要。

#### 2.2、Multitask Learning：多任务学习共享参数更新

卡耐基梅隆大学 Rich Caruana 在 1997 年提出了 [Multitask Learning（多任务学习）](https://link.springer.com/article/10.1023/A:1007379606734) 这样一个提升模型泛化能力的学习框架，但是经过了二十年发展，NLP 在多任务的训练探索上仍然不成熟。

不过有两个取得了一定突破的技术路线，值得关注。一个是，2018 年 OpenAI 提出的 GPT-1 与 Google 提出的 BERT 都验证了「不需针对特定任务而只需要增加自注意力即可」的架构可行性。但是这样的技术方案，依然需要用到监督学习，泛化性依然受局限。另一个是，在无监督或极少量监督数据的情况下，在特定任务上也都能取得很好表现，例如常识推理（华盛顿大学几位学者于 2017 年在[《Story Cloze Task: UW NLP System》](https://maartensap.com/pdfs/schwartz2017story.pdf)研究中验证）、情感分析（OpenAI 团队 2017 年在[《Learning to Generate Reviews and Discovering Sentiment》](https://arxiv.org/abs/1704.01444)研究中验证）。

OpenAI 团队受到两条研究路线的启发并采用更通用的迁移方法，实现无监督预训练后，在 Zero-Shot 情况下完成多任务，多有任务共享更新同一个 Transformer Decoder 架构之上的模型参数。

### 3、GPT-2 的预训练数据集：高质量、多样性的 WebText

GPT-1 是拿一堆书预训练的，其实很明显多样性是不足的，尤其只用了小说。

GPT-2 则用了 40GB 的 WebText 语料（800 万个网页）。具体地，这些网页都是来自 Reddit 的网页中包含的出站链接（Outbound Links），并且获得了至少 3 个 karma，这两点门槛让 OpenAI 认为得到了一些比较高质量的网页（明显质量比 CommonCrawl 整来那些乱七八糟的要高不少）。而且这样得到的数据集，具有非常好的多样性，因此很多任务的示例会自然地被学习到。OpenAI 在论文中提到：

> …… These findings suggest a promising path towards building language processing systems which learn to perform tasks from their naturally occurring demonstrations.<br/>…… 这些发现为构建语言处理系统指明了一条道路，就是从文本语料中自然出现的样本示例来学习并完成任务。

当 OpenAI 在研发 GPT-2 时这样认为，已经预示着两点：1）ICL（In-Context Learning），甚至 Prompt Engineering，势必会在语言模型通用性变得更强后，成为一个人人都能参与的研究热点。2）如果各类任务的模式是「隐式」出现在语料中的，那么大规模训练数据就意味着可以覆盖更多任务类型，进而暴力军备竞赛有了理论上的动力。

从实验表现上，用 WebText 预训练的 GPT-2 优于在 Wikipedia、新闻或书籍上训练的其他语言模型，而无需使用这些训练数据集。

### 4、如果预训练直接喂生数据，最终的效果怎样？

对于预训练数据集的处理，GPT-2 采用了最简单直接、符合我们目标期待的方式：不作任何处理，直接喂生数据（raw text）。

#### 4.1、生文本「隐式」包含任务模式，上下文「显式」提示具象任务

GPT-2 直接从原始文本开始学习，而不需要针对任务准备的训练数据，也不需要任何微调，尤其对于问答、阅读理解、summarization、translation 这些任务上，只需要以正确的方式提示经过训练的模型就能得到令人惊讶的结果。当然离 SOTA 还有区别，但作者们表示，从实验表现上看，如果有足够大的未标记数据集和算力，模型在这些任务上也会取得领先表现。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-43.png)
↑ 标注 (+) 的项，表示分数越高越好；标注 (–) 的项，表示分数越低越好。

这里已经基本预示着，投喂生文本数据，让模型「囫囵吞枣」地学会了不少东西，就像一个小孩子到父亲的书房里翻了很多很多书，知识都学杂了。但是如果启发教育问的好，给一些上下文提示语，模型就能给出很不错的响应。这也就引出了 In-Context Learning、Prompt Engineering 等一系列话题。关于这些的探讨，我们将在「In-Context Learning」那一章详细介绍这方面的研究发展和技术尝试。

但是显然在 GPT-2 这个阶段，其表现还没有让 OpenAI 这么笃定这件事。比如整体看，OpenAI 发现需要对 GPT-2 多尝试几次才能获得好的样本，尝试的次数取决于模型对上下文的熟悉程度：1）当提示数据中有非常具体的主题（比如英国脱欧、指环王等）时，GPT-2 能在一半的时间内生成合理的样本。2）对于高度技术性或深奥类型的内容，GPT-2 就表现不太行了。比如对于数据集 Natural Questions 上，OpenAI 给出测试问答的例子：

| 任务   | **Question Answering** |
| 数据集 | Natural Questions  |
| 示例   | Who wrote the book the origin of species?<br/><br/>**Correct answer: Charles Darwin**<br/>Model answer: Charles Darwin<br/><br/>What is the largest state in the U.S. by land mass?<br/><br/>**Correct answer: Alaska**<br/>Model answer: California |

#### 4.2、LLM 军备竞赛的序幕拉开

整体来看，GPT-2 在以下这些数据集上执行的对应任务，虽然很多没到 SOTA，但效果还可以，毕竟是没有针对性的任务数据拿来训练的。还是上面提到的那句，OpenAI 认为「由于（所训练的）这些任务是通用语言建模的子集，因此我们可以预期随着更多算力和训练数据的使用，性能将进一步提高」。这也为 GPT-3 留下了巨大的空间：

| 任务                                                          | 数据集                     |
|---------------------------------------------------------------|---------------------------|
| Summarization: summarize news articles                       | CNN and Daily Mail dataset |
| Machine Translation: translate French sentences to English   | WMT-14 Fr-En               |
| Reading Comprehension: answer questions about given passages | CoQA                       |
| Common Sense Reasoning: resolution of an ambiguous pronoun   | Winograd Schema Challenge  |
| Question Answering                                           | Natural Questions          |
| Language Modeling of Broad Contexts: predict the last word of a passage | LAMBADA         |

在 GPT-2 的开发与公布阶段，OpenAI 就已经跃跃欲试 LLM 下一阶段的可能范式：基于扩展性极好的 Transformer Decoder 架构上（撑得起巨量参数规模）构建模型，并投喂足够多的数据（海量数据已经潜在包括各种任务模式）进行无监督预训练（所有任务都「隐式」地变成从左至右的生成训练） —— 从而拉开了「大模型、大数据、大算力」的军备竞赛大幕。

### 5、OpenAI 初步预见了 LLM 可能带来的影响

其实如果大家对 2019 年的科技新闻还有印象的话，一定记得当时说有一家美国公司搞了个 AI 模型编造假新闻给大家忽悠得一愣一愣的，也就是 GPT-2。另外 GPT-2 在发布时只放出来一个小模型，所以也就是从这时开始 OpenAI 被人调侃为 ClosedAI 的。

当时 OpenAI 认为 LLM 正变得越来越可扩展、可定制、生成连贯，而这可以给各行各业带来很大积极价值，也能拿来作恶。因此 OpenAI 最初发布博客和论文时没有放出完整的模型，只给出了一个用于研究的实验小版本：https://github.com/openai/gpt-2 。具体地，GPT-2 没有放出数据集、训练代码、GPT-2 模型参数。OpenAI 认为那些可能的危害是必须预防的，所以要预期到。

#### 5.1、OpenAI 在 2019 年就倡议政府监管

> It is not possible to control research in these domains without slowing down the progress of AI as a whole.<br/>如果管控这些领域的研究，就不可能不减缓整个人工智能的进展。

OpenAI 希望大家对此要有预期，不要抵触可能出现的管控。甚至 OpenAI 早在 GPT-2 出现的 2019 年这个时间点就提出倡议：

> 政府应考虑扩大或启动更系统地监测 AI 技术的社会影响和扩散，并要能够量化 AI 系统能力的迭代发展情况。

这其实就预示着后来关于「对齐（Alignment）」这个议题被更重视的提出，顺带必然要面对的「对齐税（Alignment Tax）」，也因此有为了 Alignment 而出现的 InstructGPT，进而孕育出 ChatGPT。

#### 5.2、同年 5 月公布 3.45 亿参数版本，并暗示了后来微软对 OpenAI 不 Open 的影响

同年 5 月，OpenAI 公开发布了 GPT-2 的 3.45 亿参数版本，因为 OpenAI 认为很多机构已经能够训练同等能力的模型，所以风险不大。

另外，对那些致力于促进社会为 LLM 的广泛影响做好准备的合作伙伴与安全社区，OpenAI 开放了 7.62 亿参数和 15 亿参数的版本。这里 OpenAI 提到：

> These research partnerships will be a key input to our decision-making on larger models.<br/>这些研究方面的合作关系将是我们在决定发布更大模型时的关键因素。

这个在 2019 年时间节点上的表态，暗示了此后 OpenAI 与 Microsoft 的合作关系，对于 OpenAI 后续模型的公开程度的影响，给各方在打预防针。后来在 2023 年 1 月 23 日 OpenAI 博客上也发布了[《OpenAI 与 Microsoft 扩大合作伙伴关系》](https://openai.com/blog/openai-and-microsoft-extend-partnership/)一文提到 Microsoft 从 2019 年开始对 OpenAI 的投资。

而在 5 月这次发布的内容，还包括[GPT-2 output dataset](https://github.com/openai/gpt-2-output-dataset)。

### 6、GPT-2 小节

GPT-2 的亮点和洞察：

* 自回归语言模型虽然难但正确，可以隐式地从语料中学到各类任务概念（有研究认为其包含隐式马尔科夫模型，详见本文 XXXX TODO），继续显著提高性能及泛化能力的方法，就是加大训练数据规模、模型参数规模。
* 无需监督微调阶段：处理下游任务时不需要 fine-tune，预示未来在 LLM 上以 Prompt 方式完成任务可能成为一种新范式。

### 参考

* https://openai.com/blog/better-language-models/
* https://jalammar.github.io/illustrated-gpt2/
* https://github.com/openai/gpt-2
* https://github.com/openai/gpt-2-output-dataset
* https://zhuanlan.zhihu.com/p/56869079
* https://demo.allennlp.org/next-token-lm
* https://openai.com/blog/openai-and-microsoft-extend-partnership/
* https://link.springer.com/article/10.1023/A:1007379606734
* https://arxiv.org/abs/1806.08730
* https://arxiv.org/abs/1901.11373
* https://maartensap.com/pdfs/schwartz2017story.pdf
* https://arxiv.org/abs/1704.01444
* https://www.mikecaptain.com/2023/01/22/captain-aigc-1-transformer/