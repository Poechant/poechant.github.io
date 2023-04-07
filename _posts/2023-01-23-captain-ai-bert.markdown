---
layout: post
title:  麦克船长的 BERT 解读
date:   2023-01-23 23:56:03 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, AIGC, Transformer, BERT, Google, Gen-AI, 生成式 AI, GPT, OpenAI]
description: 
excerpt: 
location: 杭州
author: 麦克船长
---

OpenAI 发布 GPT-1 后，Google 大受震撼。在其发布 4 个月后的 2018 年 10 月，Google 终于推出了 BERT，它有两个版本 BERT-Base 和 BERT-Large。从性能表现上看，参数规模相当的情况下，BERT-Base 超越 GPT-1，而参数规模更大的 BERT-Large 又远好于 BERT-Base，可以说 Google 又穿上了黄色领骑衫。

BERT 的名字，在麦克船长看来也是作者们硬凑出来的，就是为了呼应 ELMo 这种芝麻街角色名字系列。BERT 是 Bidirectional Encoder Representations from Transformers 的缩写。下面是 BERT 的基本信息。

<div class="table-wrapper" markdown="block">

| **模型**   | **参数量** | **层数** | **词向量长度** | **注意力头数** |
| BERT-Large| 3.4 亿     | 24 层   | 1024          | 16            |
| BERT-Base | 1.1 亿     | 12 层   | 768           | 12            |
| GPT-1     | 1.17 亿    | 12 层   | 768           | 12            |

</div>

BERT 公布了[源码和训练好的模型参数供下载](https://github.com/google-research/bert)。Google 团队希望 BERT 能够让业内人士，用几个小时甚至几十分钟，就能训练好一个 SOTA 小模型。Google 在论文中称可以在 11 个 NLP 任务上取得 SOTA 结果，甚至包括非常挑战的 SQuAD v1.1 数据集。

BERT 发布一年半后的 2020 年 3 Google 又为 BERT 发布了[一系列 24 种小模型](https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip)共各种场景使用，开源精神一直都贯穿在 Google 的技术路线上。这些小模型有不同的层数和自注意力头数，具体如下：

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-60.png){: width="640"}

以下阅读前需要你初步了解 **Transformer 基本架构**。如果你此前不了解 Transformer Encoder、Decoder 的架构特点，可以看船长在本文的前篇[《人工智能 LLM 革命前夜：一文读懂横扫自然语言处理的 Transformer 模型》](http://www.mikecaptain.com/2023/01/22/captain-aigc-1-transformer/)中的第二章。

下面我们来看看在 Transformer 模型基础上研发的 BERT 吧。

### 1、BERT 的模型架构：双向

与 GPT 不同，BERT 采用的是 Transformer 的编码器。但是这样在技术路线上选择的分野，带来的影响非常的大。Transformer 的编码器就像完形填空，在预测每个词时，是知道前后（过去和未来）的文本内容的；但是 Transformer 的解码器仅知道前面的文本（过去）来预测词，相当于在预测未来。

在 ELMo 一节我们介绍过双向语言模型，这里将单向与双向作对比，举例来说，对于同一句话「I accessed the bank account」，GPT 单向语言模型的学习方法是：

```
I -> I accessed
I accessed -> I access the
I accessed the -> I accessed the bank
I accessed the bank -> I accessed the bank account
```

而 BERT、ELMo 双向语言模型的学习方法是：

```
I accessed the [MASK] account -> [MASK]=bank
```

这里就涉及到两个关键点：首先，上面在 ELMo 提到的，双向语言模型需要解决的「**See-Itself** 或 **See-Themselves**」问题。其次，BERT 如何挖词来做完形填空，即 Corruption Technique。

可以说 OpenAI 做了一个价值更大、难度更大的技术选型，因此如果在类似数据规模、模型规模、训练方法的情况下，GPT 是难有超过 BERT 的表现的。BERT 问世后的几年内，学界与业界的很多人都以为 BERT 是一统江湖的正途，甚至都认为 OpenAI 的 GPT 选择错了技术路线还硬着头皮坚持。这与 2022 年底 ChatGPT 彻底轰动世界形成了鲜明的对比。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-61.png)

上图展示了 BERT 与当时另外两大主流 NLP 模型 GPT-1、ELMo 的对比。BERT 与 GPT 的共同点是都基于 Transformer 架构，而 BERT 与 ELMo 的共同点是都用了双向架构。

#### 1.1、BERT 是深度双向（Deeply Bidirectional），ELMo 是浅度双向（Shallowly Bidrectional）

由于自注意力层的加持，Transformer 有着极其强大的特征提取能力，这使得 Google 在其官方博客上有底气说 BERT 是深度双向模型，而 ELMo 基于双向 LSTM 提取特征的能力只算是浅度双向模型，提出 ELMo 的 AI2 对此也无法辩驳。

同样基于 Transformer、得益于自注意力，双向模型比单向模型对自然语言有更好的理解，因此在 NLU（Natural Langauge Understanding，NLU）问题上可以轻松取得比单向模型好得多的表现，这也是 GPT-1 相对吃亏的地方。

#### 1.2、基于 Transformer Encoder 之上 BERT 做了哪些架构改进

首先是 Input Embedding 处理得到优化，BERT 的 Input Embedding 是三种 Embedding 的求和。

![](/img/src/2023/02/bert-official-blog-5.png)

* 单词嵌入（Token Embedding）。
* 位置嵌入（Position Embedding）：在 NLP 任务中，词的位置信息非常有影响。
* 片段嵌入（Segment Embedding）：或者叫「句子嵌入」，增加对句子结构的理解。

### 2、BERT 的训练方法

BERT 也是采用「无监督预训练 + 监督微调」的方法，与 GPT-1 相同。但毕竟是双向语言模型，BERT 的预训练任务与 GPT-1 不同，有如下两个：Masked Language Modeling（在某些文献中也叫 Mask Language Modeling，MLM）和 Next Sentence Prediction（NSP）

#### 2.1、Masked Language Modeling（MLM）预训练任务

BERT 具体采用的方法是，随机选择 15% 的 tokens 出来，但是并非把它们全部都 MASK 掉，而是：

* 其中的 80% 被替换为 \[MASK\]，类似 `my dog is hairy -> my dog is [MASK]`。
* 其中的 10% 被替换为一个随机 token，类似 `my dog is hairy -> my dog is apple`。
* 剩余的 10% 不变。

这个 80-10-10 是怎么定出来的，Google 团队也是脑拍了几种数字组合试出来的，如下图：

![](/img/src/2023/02/bert-official-blog-6.png){: width="480"}

* MNLI 任务：Multi-Genre Natural Language Inference 是一个大规模的众包蕴含（entailment）分类任务（[Williams et al., 2018](https://arxiv.org/abs/1704.05426)）。给定一对句子，预测第二个句子相对于第一个句子是蕴含、矛盾还是中性。
* NER 任务：Named Entity Recognition 命名实体识别任务，比如对于输入语句「擎天柱回到赛博坦」得到输出「B-PER, I-PER, E-PER, O, O, B-LOC, I-LOC, E-LOC」，其中 B、I、E 分别表示开始、中间、结束，PER、LOC 分别表示人物、地点，O 表示其他无关。

#### 2.2、Next Sentence Prediction（NSP）预训练任务

许多 NLP 任务（比如问答、推理等等）都涉及到句子之间关系的理解，这不会被一般性的语言建模过程学习到。因此 Google 想用预训练阶段的 NSP 任务来解决这个痛点。NSP 预训练任务所准备的数据，是从单一语种的语料库中取出两个句子 $$ S_i $$ 和 $$ S_j $$，其中 50% 的情况下 B 就是实际跟在 A 后面的句子，50% 的情况下 B 是随机取的。这样语言模型就是在面对一个二元分类问题进行预训练，例如：

```
INPUT: [CLS] the man went to [MASK] store [SEP]
       he bought a gallon [MASK] milk [SEP]
LABEL: IsNext
```

```
INPUT: [CLS] the man [MASK] to the store [SEP]
       penguin [MASK] are flight ##less birds [SEP]
LABEL: NotNext
```

CLS 是一个表示「classification」的 token，SEP 是一个表示「separate」的 token。这样的预训练任务，让 BERT 在词维度的语言知识外，也让 BERT 学习到一些句子维度的语言结构。

### 3、BERT 的哪些改进是带来最显著性能提升的？

BERT 与其他几个主流模型的性能对比如下：

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-46.png){: width="720"}

可以看到 BERT 在当时有着极其出色的表现。那么对于这么出色的表现，从上面 BERT 的架构特色到训练方法，到底什么改进对 BERT 性能的积极影响是最大的？这就要依赖**消融研究**（Ablation Studies，也可以叫消融实验）了。什么是消融研究？你在一些论文中会经常看到，就是指**删除模型或算法的某些「功能」并查看其如何影响性能**，也就是物理实验中大家最熟悉的「控制变量」法，我们下面具体看下 Google 对 BERT 做的消融研究实验：

![](/img/src/2023/02/bert-official-blog-7.png){: width="480"}

上表中对如下四项做了对比：

* BERT-Base：这个是 baseline，其他所有「变量」都基于此。
* No NSP：在 BERT-Base 上移除「Next Sentence Prediction」预训练任务。
* LTR & No NSP：LTR 就是 Left To Right，也就是变成了 GPT 的 Auto-Regression Model 架构，同时也把 NSP 预训练任务移除。
* +BiLSTM：在 fine-tuning 期间，基于「LTR & No NSP」架构之上增加随机初始化的 BiLSTM。

可以看到，「LTR & No NSP」与「No NSP」对比，在 5 个任务中的 4 个都显著大幅下降，说明**双向结构的正向影响是最显著的**。而单独移除 NSP 后，各任务上的表现只小幅下降，但其中在 QNLI 任务上大幅下降（QNLI，Question Natural Language Inference 是基于 Stanford Question Dataset 之上的一个测试推理能力的二元分类任务），这说明**增加句子维度的学习对「推理」有帮助**。再看在「LTR & No NSP」上加「BiLSTM」也未能拯救性能（只有 SQuAD 提升多一些），说明 **Transformer 特征抽取能力比 BiLSTM 强很多**。

### 4、BERT 的数据集

BERT 比 GPT-1 用的训练数据集要大得多。BERT 同样也用了 BookCorpus（并且继承了 GPT-1 在论文中的拼写谬误「BooksCorpus」）约含 8 亿个词，以及英文维基百科（English Wikipedia）约含 25 亿词。整体来看，BERT 训练数据集大小差不多是 GPT-1 训练数据集的 4 倍左右。

### 5、BERT 小节

1. **用足够硬的打榜成绩夯实了「预训练 + 微调」学习范式**：过去我们都是针对某个任务进行训练，让模型成为这个任务领域的专家。但是 NLP 的很多知识是有交叉的，比如语言知识、推理能力等等，各个任务的边界并不泾渭分明，因此总是为了更好解决特定任务而要学习补充其他知识。逐渐地，领域知识的边界越来越模糊，知识范围越来越广，就逐渐自然地向着大语言模型的方向发展了，于是就出现了 GPT、BERT 这种「**预训练 + 微调**」的学习范式。但是 BERT 对特定任务微调后，由于参数被更新，相应地在其他一些任务上的表现可能就会下降，这就导致**模型的泛化能力受到局限**。而后来的 GPT-3、InstructGPT 到 ChatGPT，则是在预训练完成后并不针对任何下游任务更新参数。这样的好处是模型泛化能力很好，但是针对到特定任务身上，很肯定没有监督精调的 BERT 好，尤其是在 NLU 类型的任务上。
2. **开源并开放各种规格的模型下载**：成为了 2018 到几乎 ChatGPT 出现之前 NLP 领域研究的核心模型。
3. Transformer Encoder 双向模型的特征抽取能力，被充分认可。但其实双向语言模型在生成类任务上并不符合人类自然的语言文字「从前向后」的交互模式，这也为后来 GPT 系列反超埋下伏笔。
4. 掀起了模型轻量化的研究热点，尤其在 2020 年推出 24 个小模型后。
5. NSP 预训练任务增加了句子层面的语言结构理解。

### 6、动手小实践

BERT 是主流大模型里，开放源代码和模型参数最好的。我们在本小节用 bert-as-service 来跑个简单的例子，为了让大家在任何个人电脑上都跑的起来，这个例子比较小，我们主要是为了简单实践一下找找感觉。

#### 6.1、安装 BERT 所需要的各种依赖

```shell
conda install tensorflow==1.14.0
```

验证 tensorflow 是否安装正确：

```python
import tensorflow as tf
print(tf.__version__)
```

#### 6.2、下载一个预训练（Pre-Train）过的 BERT 模型

官方的模型在这里浏览：https://github.com/google-research/bert#pre-trained-models

也有一些中文的模型，以下是 ChatGPT 推荐的三个：

* BERT-Base, Chinese：这是 Google 官方提供的中文 BERT 模型，在中文 NLP 任务中表现良好。你可以从 这里下载这个模型。
* ERNIE：这是由中科院自然语言所提供的中文 BERT 模型，包含了额外的语义信息。你可以从 这里下载这个模型。
* RoBERTa-wwm-ext：这是由清华大学自然语言处理实验室提供的中文 BERT 模型，在多种中文 NLP 任务中表现良好。你可以从 这里下载这个模型。

#### 6.3、安装 BERT 的服务端和客户端

这里我们使用 bert-as-service，bert-as-service 是一种将 BERT 模型部署为服务的方式。该工具使用 TensorFlow Serving 来运行 BERT 模型，并允许通过 REST API 进行调用。根据 bert-as-service 的文档，它已经在 TensorFlow 1.14.0 上测试过。

在你激活的 conda 环境里，安装 `bert-as-service`：

```shell
# 安装服务端和客户端
# 更多关于 bert-serving-server 的信息可以参考：https://bert-serving.readthedocs.io/en/latest/index.html
conda install bert-serving-server bert-serving-client 
验证 bert-as-service 是否安装成功
bert-serving-start -h
```

#### 6.4、启动 BERT 服务端

```shell
# 命令行下启动BERT服务
# -num_worker 表示启动几个worker服务，即可以处理几个并发请求，超过这个数字的请求将会在LBS（负载均衡器）中排队等待
bert-serving-start -model_dir /模型/的/绝对/路径 -num_worker=4
```

#### 6.5、编写程序实现 BERT 客户端

这里有一些客户端例子可以参考：https://cloud.tencent.com/developer/article/1886981

```python
from bert_serving.client import BertClient
import numpy as np

# 定义类
class BertModel:
    def __init__(self):
        try:
            self.bert_client = BertClient(ip='127.0.0.1', port=5555, port_out=5556)  # 创建客户端对象
            # 注意：可以参考API，查看其它参数的设置
            # 127.0.0.1 表示本机IP，也可以用localhost
        except:
            raise Exception("cannot create BertClient")

    def close_bert(self):
        self.bert_client.close()  # 关闭服务

    def sentence_embedding(self, text):
        '''对输入文本进行embedding
          Args:
            text: str, 输入文本
          Returns:
            text_vector: float, 返回一个列表，包含text的embedding编码值
        '''
        text_vector = self.bert_client.encode([text])[0]
        return text_vector  # 获取输出结果

    def caculate_similarity(self, vec_1, vec_2):
        '''根据两个语句的vector，计算它们的相似性
          Args:
            vec_1: float, 语句1的vector
            vec_2: float, 语句2的vector
          Returns:
            sim_value: float, 返回相似性的计算值
        '''
        # 根据cosine的计算公式
        v1 = np.mat(vec_1)
        v2 = np.mat(vec_2)
        a = float(v1 * v2.T)
        b = np.linalg.norm(v1) * np.linalg.norm(v2)
        cosine = a / b
        return cosine


if __name__ == "__main__":
    # 创建bert对象
    bert = BertModel()
    while True:
        # --- 输入语句 ----
        input_a = input('请输入语句1: ')

        if input_a == "N" or input_a == "n":
            bert.close_bert()  # 关闭服务
            break

        input_b = input('请输入语句2: ')

        # --- 对输入语句进行embedding ---

        a_vec = bert.sentence_embedding(input_a)
        print('a_vec shape : ', a_vec.shape)

        b_vec = bert.sentence_embedding(input_b)
        print('b_vec shape : ', b_vec.shape)

        # 计算两个语句的相似性
        cos = bert.caculate_similarity(a_vec, b_vec)
        print('cosine value : ', cos)

        print('\n\n')

        # 如果相似性值大于0.85，则输出相似，否则，输出不同
        if cos > 0.85:
            print("2个语句的含义相似")
        else:
            print("不相似")
```

#### 6.6、测试效果

在使用 `bert-serving-client` 连接 `bert-serving-server` 时，你需要确保 `bert-serving-server` 使用的模型和 `bert-serving-client` 使用的模型是匹配的，否则会出现错误。

程序正常运行后，将要求你输入两句话，然后 BERT 计算两句话的相似性。

```shell
请输入语句1: 
请输入语句2: 
```

两句输入好确认后，得到如下形式的结果：

```
a_vec shape :  (768,)
b_vec shape :  (768,)
cosine value :  0.8691698561422959
```
#### 本小节参考

* https://nlp.stanford.edu/seminar/details/jdevlin.pdf
* https://zhuanlan.zhihu.com/p/49271699
* https://arxiv.org/abs/2302.09419
* https://zhuanlan.zhihu.com/p/530524533
* https://arxiv.org/abs/1704.05426
* https://github.com/hanxiao/bert-as-service
* https://cloud.tencent.com/developer/article/1886981