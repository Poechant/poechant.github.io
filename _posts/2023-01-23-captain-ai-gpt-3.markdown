---
layout: post
title:  麦克船长的 GPT-3 解读
date:   2023-01-23 23:58:13 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, AIGC, Transformer, GPT, OpenAI]
description: 
excerpt: 
location: 杭州
author: 麦克船长
---

在 GPT-2 发布 1 年零 3 个月后的 2020 年 5 月，OpenAI 团队发布 GPT-3，从其论文[《Language Models are Few-Shot Learners》](https://arxiv.org/abs/2005.14165)可以看到，其最大亮点是数据规模、参数规模都比 GPT-2 大 100 倍。这么简单粗暴的办法，带来的是 GPT-3 表现上的炸裂。而其论文标题也点出了本文的主题：语言模型就是要无监督预训练 + Few-Shot Prompt。

### 1、GPT-3 表明 OpenAI 进一步收紧其技术开放度

首先要明白 OpenAI 在技术开放上的收紧策略，自 GPT-2 发布时就已经给公众打了预防针了，这点船长在「TODO」中已经提到。因此我们能看到 GPT 系列上 OpenAI 变得越来越 Closed：GPT-1、GPT-2 发布时，OpenAI 都在其官网发布了 blog，到了表现远超过 GPT-1、GPT-2 的 GPT-3 时，OpenAI 反而连一篇博客都没写。而其论文更是对关键的训练、模型、数据做了哪些重要工程表现得轻描淡写，花大篇幅着墨于实验及结果上。

GPT-3 没有放出源码、预训练好的模型参数等等，俨然变成了各国网友们调侃的 ClosedAI 了。

对于模型架构，OpenAI 声称 GPT-3 与 GPT-2 是一样的。GPT-3 依然延续了此前 GPT-2 的基本架构和预训练方法：构建基于 Transformer Decoder 的自回归语言模型，然后进行无监督预训练，无针对特定任务的微调。

### 2、GPT-3 的模型参数

我们可以看到共 8 个参数规模的模型如下。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-52.png){: width="720"}

在模型**参数规模** $$ n_{params} $$ 的军备竞赛方面，可以看到 GPT-3 Small 是与 GPT-1（1.17 亿参数）、BERT-Base（1.1 亿参数）对应的；GPT-3 Medium 是与 BERT-Large（3.4 亿参数）对应的；其余六个的参数规模都直接超过了这些大模型，尤其 GPT-3 1750 亿参数的版本，直接从 GPT-2 最大的 15.4 亿版本基础上拉升了 100 多倍规模！也比以往任何一个非稀疏模型至少大 10 倍！

模型的 Transformer decoder block 的**层数** $$ n_{layers} $$，也常叫**模型深度**。可以看到随着模型参数规模的急速拉升，OpenAI 团队并没有把模型深度急剧增加，与 GPT-2 152M 版本相比，比它大一百多倍的 GPT-3 175B 的层数也只不过是其两倍（96 vs. 48）。

模型的**词向量长度**，也常叫**模型宽度**，一般表示为 $$ d_{model} $$，区分于注意力头的宽度 $$ d_{head} $$。与 GPT-2 对比就能看到，类似层数的模型情况下，GPT-3 要宽很多，可以说整体 GPT-3 的模型要胖一些，而不是在提升规模时更多去加深（加层数）。

GPT-3 的 **batch size** 都非常大，哪怕 Small 版本也有 0.5M，175B 版本更是达到 3.2M，这对内存要求非常高。如果 batch 处理并行的话，那么内存就是就是 3.2M 除以并行的数量。注意一般 batch 指的是每次训练迭代中模型处理的样本数，而不是具体的数据量，即 3,200,000 个样本，而不是 3.2MB 数据。把 batch size 做大的好处是，降低模型训练时的通讯量。不过对于小模型，batch size 太大的话很容易过拟合。对于参数量大的模型，batch size 自然也要相匹配的变大。

**LR（Learning Rate，学习率）**在 GPT-3 里是随着 batch size 增加而下降的，这与一些研究的结果是相反的，比如 Meta AI 在论文《》中提出 batch size 增加时 LR 也要随之变大。

### 3、GPT-3 的训练数据

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-6.png){: width="640" }

上面这个表格，是 GPT-3 用到的训练数据集。关于 GPT-3 的训练数据集解读，很多中文文章里说的是错的，主要是对论文中该表的数据理解错误，下面我们分别来看下各个数据集和整体数据处理做的核心工作。

#### 3.1、训练数据集

GPT-3 的数据源来自五部分组成，包括一个大型的来自 CommonCrawl.org 数据集、扩展的 WebText 数据集、两个互联网上的书籍语料库（一般认为书籍的语料质量是非常高的）和英文的维基百科。表中各列的含义如下：

* **第二列 Quantity**：每个数据源本身的数据规模，单位是 tokens（根据 OpenAI 官方文档里提到的一个大概的经验是，通常英文文本里 1 token 有 4 个字母或者 0.75 个单词）。
* 训练期间从给定数据集中提取的部分，就是**第三列 Weight in training mix（数据混合训练时的权重）**，因为不同数据集质量不同，基于此考虑 OpenAI 结合数据质量因素做的配比。最后，GPT-3 整体上是设定了 3000 亿 tokens 的训练数据集。
* **第四列**表示各数据集在**训练时出现的次数**，最高的 wikipedia 是 3.4 次。这样其实是为了高质量训练，稍微接受一点点的过拟合。

说到这里，不得不提中文 LLM 的数据集问题。中文整体上是缺乏这些非营利组织语料库、高质量文本内容社区/百科平台内容的。类似 StackOverflow 这种技术问答社区、WikiPedia 高质量百科。

#### 3.2、提高数据质量的处理准备工作

在上述数据集基础上，GPT-3 使用如下方式提高了数据集的质量。

首先，用一个高质量数据集作为正例，用 LogisticRegression 过滤了 CommonCrawl 的数据。这个高质量数据集是什么呢？还记得 GPT-2 里采用的 WebText 数据（Reddit 外链，且 Karma 大于 3）吗？OpenAI 认为这是质量比较高的数据，就是将其作为正例来过滤的。过滤前的文本压缩规模是 45TB，过滤后的是 570GB。

其次，在文档级别上用 LSH 算法去除重复的数据。LSH 是 Locality Sensitive Hashing（局部敏感哈希），是信息检索领域的一个常用算法，可以快速判断一个词集合（文章就是一个词集合）和一个很大集合之间的相似度。

第三，再额外加一些高质量的数据，上面表格中 WebText2 就是基于 GPT-2 用的 WebText 扩展而来的，另外还有 English Wikipedia、两个电子书数据集。

### 4、GPT-3 的训练开销

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-58.png){: width="720"}

上图展示了 GPT-3 与 BERT、RoBERTa、T5 的训练算力对比。可以看到 GPT-3 的算力开销有多么惊人。注意图标的纵轴是非线性的、指数级间隔。那么消耗这么多的算力，性能表现如何呢？我们看下面这张表。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-59.png){: width="640"}

上表中横轴是算力（对应上上表中的纵轴，注意同样是指数增长），纵轴是训练期间的验证损失（注意也是指数增长）。可以看到每一条曲线都有一个大致的拐点，在拐点之后继续增加算力并不能显著提升性能。所有这个最优拐点连起来，是完全符合 Scaling Law 的（关于 Scaling Law 可以看本文的「第 XXXX TODO 章」）。

OpenAI 官方并没有公开讲过花了多少钱训练 GPT-3，市面上流传的说法「460 万美元」目前考证来看是一家云服务厂商用其最低价格 GPU 云服务估算而写的一篇软广，并不可信。也有一些其他组织或个人做了测算，整体上都是想表达训练一次很贵。至少从 OpenAI 的论文里我们能看出来，这个训练花费已经贵到研究人员即使发现多个 bug 也没舍得重新训练的地步：

> 论文第 9 页：Unfortunately, a **bug** in the filtering caused us to ignore some overlaps, and **due to the cost** of training it was not feasible to retrain the model.<br/>中文翻译：不幸的是，过滤中的一个 BUG 导致我们忽略了一些重叠，而考虑到训练成本，重新训练模型是不可行的。

> 论文第 31 页：Unfortunately, a **bug** resulted in only partial removal of all detected overlaps from the training data. Due to the cost of training, it wasn’t feasible to retrain the model.<br/>中文翻译：不幸的是，一个 BUG 导致仅删除了训练数据中检测到的所有重叠的一部分。考虑到训练成本，重新训练模型是不可行的。

> 论文第 44 页：Due to a **bug** revealed by this analysis, filtering described above failed on long documents such as books. Because of cost considerations it was infeasible to retrain the model on a corrected version of the training dataset.<br/>中文翻译：由于此分析揭示的 BUG，上述过滤在长文档（比如书籍）上是失败的。出于成本考虑，在训练数据集的修正版本上重新训练模型是不可行的。

作为对比，我们看下 Meta AI 在同样参数规模的[《OPT: Open Pre-trained Transformer Language Models》](https://arxiv.org/abs/2205.01068) 模型上，用约 1000 个 80G A100 GPU 上训练至少两个月时间，就可想而知这花费有多高昂了。

一些估算开销的链接：
* https://zhuanlan.zhihu.com/p/608181241
* https://www.zhihu.com/question/412295638/answer/1387457462
* https://arxiv.org/pdf/2005.14165.pdf^What
* https://mc.ai/what-is-gpt-3-how-can-you-use-it/^OpenAI’s
* https://venturebeat.com/2020/06/01/ai-machine-learning-openai-gpt-3-size-isnt-everything/^OpenAI 
* https://venturebeat.com/2020/06/11/openai-launches-an-api-to-commercialize-its-research/^GPT-3
* https://lambdalabs.com/blog/demystifying-gpt-3/^GPT-3
* https://www.reddit.com/r/MachineLearning/comments/h0jwoz/d_gpt3_the_4600000_language_model/

### 5、In-Context Learning

OpenAI 在 GPT-3 发布中显式地提出了 In-Context Learning，即在无监督训练好的 GPT-3，使用时用少量示例就可以得到有较好的输出反馈，这就叫 Few-Shot Prompt。只有 1 个示例的时候就叫 One-Shot Prompt，没有示例的时候就叫 Zero-Shot。对于在使用时出现在输入中的这些示例，模型是不会更新其参数来做 fine-tune 的。那么模型是怎么从这些示例学到东西的呢？我们把这样的学习方法叫 In-Context Learning，即模型从无监督的训练文本上下文里，完成了非显性的学习。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-57.png)

In-Context Learning 与 Fine-Tune 两者区别在于是否更新模型参数。In-Context Learning 这种神奇的能力为什么会 work，船长将和你在本文「第 XXX TODO 章」一起来初步探索下（目前这个领域还没有完全清晰明确的理论证明）。OpenAI 评估模型性能时，对示例不同做了区分：

1. Few-Shot Learning，对每个子任务提供 10-100 个样本。
2. 其中一个特殊情况是只给 1 个样本，我们叫 One-Shot Learning。
3. Zero-Shot Learning 顾名思义无样本。

Fine-Tune

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-48.png){: width="480"}

Few-Shot

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-51.png){: width="480"}

One-Shot

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-50.png){: width="480"}

Zero-Shot

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-49.png){: width="480"}

#### 5.1、与 ICL 有关的一些实验结论

模型性能表现，随着示例样本数增加而增加，如下图所示。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-56.png){: width="640"}

无论是 Few-Shot、One-Shot 还是 Zero-Shot，模型的性能表现都随着参数规模的增加而增加，如下图所示。

![](/img/src/2023/2023-01-23-captain-aigc-2-llm-47.png){: width="640"}

#### 5.2、Few-Shot Prompt

* 沿着 Few Shot 的方向，机器学习的过程有点类似于人类的学习：大量无监督输入，针对特定任务只需极少量有监督输入。

### 6、GPT-3 的亮点与局限

亮点

* 展示了不用微调的可能性！

局限

* 训练成本太高。
* 样本有效性差，目前训练方法还是让 GPT-3 看了太多数据了（相比人类一生能看的文本量，GPT 看的量太大了）。未来提高样本有效性也是一个重要工作。
* 缺少 Alignment，可能导致被用来撒布不实消息、生成垃圾邮件或者钓鱼、造假论文；内容可能带有性别偏见、种族评价、宗教歧视；缺少对政治敏感的兼容，但这也是最复杂的。
* 能耗太高，不环保。
* GPT 在 few-shot learning 时到底是现学的，还是找到原来学过的相似的东西找出来。如果是后者，那真的是在拼训练数据大小了。但是对比人类，我们应该要做到前者才对。
* 如果要补全一段，还可以。如果要一直续写小说，GPT 可能不太行。
* 训练学习时，对每个词都是一样对待的，就不像人类其实是有重点的。
* 这里也有问题：1）当你在下游任务真有一大组样本（比如 1 万条）想给模型时，Few-Shot 真的给模型那么多数据么，那每次使用都要带着也太麻烦了，效率也不高。2）哪怕只有 1 条样本想 Prompt，不用它效果就不好，但是每次使用模型都要把这一条带着也不优雅。

### 7、GPT-3 的 API 及其应用

GPT-3 问世后，出现了不少基于它的应用。

### 8、GPT-3 小节

* GPT-3 跟很多深度学习模型一样，都是无法解释的
* **LMM 一定是下一个军备竞赛打卡点**。LMM 即 Large Multimodal Models（大型多模态模型），目前的 LLM 只是在读书（读文本），缺少其他体验，比如视频到底是什么鬼，比如真实物理世界的交互；而 LMM 就不同了，可以与人类进行不同模态的交互，可以读懂人类给它的文本、视频、语音等等模态内容，也能根据需要给人类生成文档、图片、视频等等。
* GPT-3 论文试图重新定义 Meta Learning，但发表后并没有引起大家的认同。

### 参考

* https://arxiv.org/abs/2205.01068
* 





<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=TB
	splines=ortho
	node [shape="box"]

	cd2[label="code-davinci-002"]
	td2[label="text-davinci-002"]
	td3[label="text-davinci-003"]

	cd2 -> td2
	td2 -> td3
}
{% endgraphviz %}
</div>












#### GPT-3 的论文要点

* 最近这些年 NLP 领域都是用预训练好的语言模型再做微调，这当然是有问题的。我们对每个子任务，需要一个跟任务相关的数据集，而且要一个跟任务相关的微调。
—— 1、需要一个大数据集，需要标注
—— 2、当一个样本没有出现在数据分布里面时，大模型的泛化性效果不一定比小模型好。所以对于原来那种老模式，如果微调后效果很好，也不能说明预训练后的泛化性好。很有可能是过拟合了预训练数据。
—— 3、人类是不需要一个很大的监督数据集去做一个任务的，基本 few-shot 就 ok 了。


论文 2.3、训练过程

论文 2.4、Evaluation

论文 3、结果

Compute-验证损失 曲线，最佳值的点连起来是一个 power law 分布：损失线性下降，计算量指数增长

参数量-精度，Zero、Few、One 都可以超过目前最好的算法，其中用 Few shot 甚至可以毕竟人类

* 一些研究表明，当模型越来越大时，似乎过拟合没有那么严重。
