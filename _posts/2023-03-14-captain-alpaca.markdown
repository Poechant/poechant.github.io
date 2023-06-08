---
layout: post
title: Stanford 研究团队发布基于 LLaMA 微调的 Alpaca 模型，仅花费不到 500 美元
date:   2023-03-14 12:40:13 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, 语言模型, 大模型, LLaMA, Alpaca, Stanford, 斯坦福, 大学, 微调, finetune, Model, Lanaguage Model]
description: 
excerpt: 
katex: True
location: 杭州
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

Meta 的 LLaMA 模型发布后不久，在 2023 年 3 月 13 日，一个叫 Alpaca 的模型被发布。Alpaca 是一个在 LLaMA-7B 基础上用 5.2 万条的「instruction-following」微调得到的 LLM，由 Stanford 大学的基础模型研究中心（Center for Research on Foundation Models，CRFM）团队发布，训练总花费约不到 600 美元。

为啥名字叫 Alpaca？LLaMA 是来自缩写（当然也是硬拗的），而 Alapaca 就纯粹是因为羊驼家族的名字了。作为羊驼家族的成员，llama 和 alpaca 都是来自南美。在南美有四类羊驼，分别是 llama（Lama glama）、alpaca（Vicugna pacos）、vicuña（Vicugna vicugna）、guanaco（或者叫 huanaco，Lama guanicoe）。从外形上，llama 和 alpaca 其实是长得有点像的，区别的方法一般是通过耳朵。耳朵比较长的，还有点弯弯的，是 llama；耳朵是直的，短一些的是 alpaca。

OK，不扯远了，如何分别各种草泥马的事情，还是交给搞动物学的朋友们，我们继续聊人工智能。Stanford CRFM 团队在其官方主页上发布了相关文章介绍 Alpaca，官方的一些相关参考资料如下：

* Blog：[《Alpaca: A Strong, Replicable Instruction-Following Model》](https://crfm.stanford.edu/2023/03/13/alpaca.html)
* DEMO：https://crfm.stanford.edu/alpaca/
* Code：[GitHub - Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

## 1、为什么做 Alpaca？

尽管像 ChatGPT、Claude、New Bing 的 Chatbot 这些已经很厉害了，但是 CRFM 团队认为这些模型依然存在许多不足之处，尤其是它们可能生成虚假信息、传播社会刻板印象、产生有害语言等等。为了在这方面的研究上取得一些推进，学术界的参与至关重要。但不幸的是，在学术界进行关于指令跟随模型的研究一直很困难，因为没有一个容易获取且能够与 GPT-3.5 等闭源模型相媲美的模型。

刚好 Meta 发布了 LLaMA，CRFM 团队就基于 LLaMA-7B 微调了除了 Alpaca。

## 2、用的什么数据微调？

CRFM 团队自己花钱使用了 OpenAI 的 text-davinci-003 模型 API 生成了 52,000 条 Self-Instruct 的数据，然后用这些数据对 Alpaca 进行 Finetune。效果性能评估后，CRFM 团队声称 Alpaca 展现了许多与 OpenAI 的 text-davinci-003 相接近的表现，但 Alpaca 的规模要比 text-davinci-003 小太多了，而且微调的话费仅仅是不到 600 美元，所有人都可以低成本的复现。

* 训练数据：5.2 万条的「instruction-following」。
* 关于训练数据：发布在 https://github.com/tatsu-lab/stanford_alpaca#data-release
* 数据生成过程 https://github.com/tatsu-lab/stanford_alpaca#data-generation-process

当然实际效果后来大家也都知道，并没有 CRFM 王婆卖瓜所说的那么好，但是这已经是一个不小的启发了。尽管大家知道微调的成本是远低于预训练的，但是 CRFM 团队直接给出了一个具体量化的结果，而且这个流程可实操落地，未来可优化的空间就自然很大。

## 3、Self-Instruct

2022 年 12 月底，在 ChatGPT 发布后不到一个月，Yizhong Wang 等几位学者共同发布了论文《SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions》。这篇论文主要是指出「可以用现成的、强大的大语言模型，来自动生成微调所需要的数据」，这样的好处，是大量地省去了人工编写 Instruction-Following 数据的工作量，因为这种数据通常在数量、多样性和创造性方面受到人工编写规模的限制，很难搞出非常多，这也就间接地限制了继续推进 LLM 的泛化能力拓展。这篇论文的作者引入 Self-Instruct，是一个用 LLM 来造数据用于微调的框架。整个流程从 LLM 生成指令、输入和输出样本，然后在使用它们对目标模型进行 finetune 之前，过滤无效或相似的样本。

论文作者将这个方法应用于原始的 GPT-3，性能测试结果显示在 Super-NaturalInstructions 任务上相对于原始模型实现了 33% 的绝对改进，与使用私有的用户数据和人工注释进行训练的 InstructGPT-001 的性能相当。为了进一步评估，论文作者还搞了一组专家编写的用于新任务的指令，并通过人工评估，结果显示使用 Self-Instruct 对 GPT-3 进行调优明显优于使用现有的公共指令数据集，而且作者自称是运用 Self-Instruct 框架的结果仅与 InstructGPT-001 相差 5% 的绝对差距。Self-Instruct 提供了一种几乎无需人工标注的方法，可以使预训练语言模型与指令对齐。

该篇论文的研究团队还公开了他们的合成数据集 `https://github.com/yizhongw/self-instruct`（截止本文编写时，GitHub 上已有 2.5K 的 stars 了）。

## 4、Alpaca 的算力开销及微调方法

Alpaca 的微调总花费约不到 600 美元，具体是在 8 个 80GB 的 A100 上微调 3 个小时。其大致的微调流程如下：

![](/img/src2/2023/06/04/1.jpg){: width="720"}

首先，使用 self-instruct 种子集中的 175 个人工编写的「instruction-following」输出对。CRFM 团队还放出了这 175 个种子数据（`https://github.com/yizhongw/self-instruct/blob/main/data/seed_tasks.jsonl`），大家可以浏览下大致格式：

```json
{
    "id": "seed_task_0",
    "name": "breakfast_suggestion",
    "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?",
    "instances":
        [
            {
                "input": "",
                "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup watter, totalling about 550 calories. The 4 strips of bacon contains about 200 calories."
            }
        ], 
    "is_classification": false
}
```

这个 Self-Instruct 的生成流程大致如下：

![](/img/src2/2023/06/04/2.jpg){: width="720"}

然后，CRFM 团队使用种子集作为提示，让 OpenAI 的 text-davinci-003 生成更多指令。他们简化了生成流程（详细信息请参阅 GitHub），大大降低了成本。数据生成过程产生了52K个唯一指令及其对应的输出，使用 OpenAI API 的成本不到 500 美元。

## 5、Alpaca 的局限性

CRFM 团队在其官方文章中特别强调「Alpaca 仅用于学术研究，任何商业用途都是被禁止的」，并解释其原因有三个因素：

1. Alpaca 基于 Meta AI 发布的 LLaMA 项目，因此也就继承了其非商用的特点。
2. 用于微调的数据，是基于 OpenAI 的 text-davinci-003 模型生成的，其使用条款是禁止其生成的数据用于开发与 OpenAI 竞争的模型。
3. CRFM 团队没有设计足够的安全措施（safety measures），因此 Alpaca 无法应对通用型的使用场景，只适合拿来做研究。

## 6、Alpaca 带来的启发及优化空间

Alpaca 是被公开放出的第一个基于 LLaMA 的微调，吹响了 LLaMA 生态被开源社区添砖加瓦繁荣起来的号角。Alpaca 虽然性能一般，但是按照同样的范式，可以有大量的优化研究可做：

1. 使用更好的数据来 finetune。
2. 探索如何批量制造更好的数据。
3. 训练的方法存在的优化空间也很大。
4. 除了 LLaMA-7B 模型，还有 13B、33B、65B 的潜力等待挖掘。

## 7、Alpaca.cpp 发布

Stanford CRFM 团队的 Alpaca 发布后的四天，一个在多种个人计算设备上就能跑起来的 Alpaca.cpp 被发布，由加州的 Kevin Kwok 将 LLaMA.cpp 与 Alpaca 做了整合。

Alpaca.cpp 的项目地址为 `https://github.com/antimatter15/alpaca.cpp`。 

## 参考

* https://crfm.stanford.edu/2023/03/13/alpaca.html
* https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform
* https://pujols.pixnet.net/blog/post/23619904
* https://arxiv.org/abs/2212.10560
 https://github.com/yizhongw/self-instruct