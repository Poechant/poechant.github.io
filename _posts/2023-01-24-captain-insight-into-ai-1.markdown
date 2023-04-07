---
layout: post
title: 人工智能大模型发展时间脉络梳理
date: 2023-01-23 05:00:33 +0800
categories: ai
tags: [AI, 人工智能, AC, 人工意识, Artificial Intelligence, Artificial Consciousness, NLP, CV, 自然语言处理, 计算机视觉, Transformer, ChatGPT, OpenAI, 微软, Microsoft, Meta, 神经网络, Neural Network, Attention, 注意力机制, GPT, BERT, Google, 大模型, 大数据, 大算力, GPU, TPU, 并行计算, 图灵测试, 图灵机, 预训练, AGI, 通用人工智能, 生成式AI, 监督学习, 半监督学习, 无监督学习, Finetune, Fine Tune]
description: 
excerpt: 
katex: True
location: 香港
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

## 数据集

* Wikipedia 数据集：https://meta.wikimedia.org/wiki/Data_dump_torrents
* C4（Colossal Clean Crawled Corpus）：https://www.tensorflow.org/datasets/catalog/c4

## 一、模型理念

### 生成模型（生成器）

Generative Against Network
Boltsmann Machine
Variational Auto-Encode
GPT

### 缩放定律（The Scaling Law）

* （OpenAI）的研究者认为语言模型的性能与模型尺寸的关系可以通过对数线性曲线预测，即模型尺寸呈指数增长时，性能会随之线性增加。这种现象被称为语言模型的缩放定律，正如 Kaplan 等人在2020年最初的GPT3文章中讨论的那样。

### 思维链（Chain of Thought，CoT，2022 年 1 月）

正如作者所展示的那样，思维链提示在性能-比例曲线中表现出明显的**相变**。当模型尺寸足够大时，性能会显著提高并明显超越比例曲线。

当使用思维链进行提示时，大模型在复杂推理上的表现明显优于微调，在知识推理上的表现也很有竞争力，并且分布鲁棒性也存在一定的潜力。要达到这样的效果只需要8个左右的示例，意味着范式可能会转变。在 ChatGPT 上线之后，整个领域为之震撼，意识到范式已经转变了。[1]

## 二、训练方法

### 强化学习（Reinforcement Learning）

* 无法使用数据集训练，只能通过真实环境或模拟器产生的数据来学习
* 应用：游戏 AI（Game AI）、AlphaGo

### 深度神经网络

* 使用神经网络构建强化学习的方法，就是深度强化学习

### 自指令（Self-Instruct，2022 年 12 月）

* 在高质量的人类标注数据上，微调基础语言模型。
* 论文地址：[《Self-Instruct: Aligning Language Model with Self Generated Instructions》](https://arxiv.org/abs/2212.10560)

### 指令微调 (Instruction Fine-Tuning，IFT）

* few-shot prompting，指令微调可以看作是有监督微调的一个子集。
* instruction + input + output 组成，前面两个组起来是 instance
* input、output 组成的 instance 是可选的，也可以没有，比如开放式生成文本
* IFT 的 training data 通常是人工编写 instruction + bootstrap（语言模型自举）得到的 instances

### 有监督微调（Supervised Fine-Tuning，SFT）

* SFT 阶段经常被用于提高响应的**安全性**，而不是接在 IFT 后面提高指令相应的具体性。
* 指令微调可以看作是有监督微调的一个子集。

### 人类反馈强化学习（Reinforcement Learning from Human Feedback，RLHF）

* 使用了 RLHF 的模型：OpenAI 的 InstructGPT、DeepMind 的 Sparrow 和 Anthropic 的 Constitutional AI。
* 偏好模型：RLHF 中包含一个偏好模型，该模型用于返回 RL 优化器的标量奖励。
* RLHF：根据人类反馈来对模型的输出响应进行排序标注，用这些带标注的输出响应来训练「偏好模型」，最后通过强化学习训练 Dialogue Agent 来模拟偏好模型。

RLHF 的步骤：

* 预训练一个语言模型 (LM) ；
* 聚合问答数据并训练一个奖励模型 (Reward Model，RM) ；
* 用强化学习 (RL) 方式微调 LM。

![image](/img/src/2023/2023-01-24-sft-sparrow-rules.png)

### AI 反馈强化学习（Reinforcemen Learning from AI Feedback，RLAIF）

* https://arxiv.org/abs/2212.08073

### 自监督（Self-Supervised）

### Conventional Fine-Tuning

* 论文链接：

## 三、模型发展

XLNet、RoBERTa、ALBert、Google T5

### Google - BERT

* 论文地址：https://aclanthology.org/N19-1423.pdf

### Google - PaLM & Pathways（2022 年 2~4 月）

最能体现 Google 技术眼光

### Google - LaMDA

* 论文地址：https://arxiv.org/abs/2201.08239

### Meta - Galactica

* 问题：发现会生成错的或有偏见的，这个很危险。
* 上线后 3 天下架（2022 年 11 月 18 日）

### Meta - BlenderBot

* 论文地址：https://arxiv.org/abs/2208.03188

### OpenAI - GPT

#### GPT-3（2020 年年中）

<div style="text-align: center;">
{% graphviz %}
digraph G {
	rankdir=BT
	splines=ortho

	mtd003[label="text-davinci-003"]
	mtd002[label="text-davinci-002"]
	mtd001[label="text-davinci-001"]

	mtc001[label="text-curie-001"]

	mtb001[label="text-babbage-001"]

	mcd001[label="code-davinci-001"]
	mcd002[label="code-davinci-002"]

	mdib[label="davinci-instruct-beta1"]

	mcd002 -> mtd002
	mtd002 -> mtd003
}
{% endgraphviz %}
</div>

GPT-3 它不仅仅是一项具体的技术，其实体现的是 LLM 应该往何处去的一个发展理念。自此之后，差距拉得越来越远，ChatGPT 只是这种发展理念差异的一个自然结果。

* 训练数据集：45 TB
* 参数规模：1750 亿，大小超过 700G

### OpenAI - InstructGPT（2022 年）

* 在性能不妥协的情况下，InstructGPT 只用了 13 亿参数（vs. GPT-3 用了 1750 亿参数），并且采用 RLHF 技术使模型对齐（align）人类，即减少有害内容的输出。
* 标注员（labelers）基于 OpenAI 的客户提交给 API 的提示（prompts）进行强化学习。具体地，用的是 2021 年 1 月部署的 Playground（用到 InstructGPT）收到的用户提交的提示（prompts），对这些 prompts 标注员提供一些示例（相当于 few-shot fine-tuning），这样来训练，然后对模型输出的结果排序，我们基于这些输出排序结果来精调 GPT-3 得到了 InstructGPT。

#### 基于 GPT-3 以 SFT 方法训练出来的 InstructGPT

* 
* 模型：davinci-instruct-beta

#### 基于 GPT-3 以 FeedME（Feedback Made Easy）方法训练出来的 InstructGPT

* 监督精调（人类写范例）+ RLHF（机器生成结果被人类打分 1~7）
* 模型：text-davinci-001, text-davinci-002, text-curie-001, text-babbage-001

#### 基于 GPT-3 以 PPO 方法训练

* 用人类比较反馈的数据作为 RM（奖励模型）进行强化学习
* 模型：text-davinci-003

### OpenAI - ChatGPT（2022 年 11 月底）

在 2023 年 1 月，Altman 接受的一次采访中提到「ChatGPT 的基本模型已经在 API 中存在很长时间了，大概 10 个月吧，甚至更久。我认为其中一个令人惊讶的情况就是，如果你做出一点微调，让（模型）以特定的方式对人们有所用途，并找出正确的交互范式，那么你就可以得到这个结果。实际上，这并不是一项全新的技术。（让它产生这个效果的）是其他的调整。我认为这一点还没有得到很好的理解。比如，很多人仍然不相信我们，他们认为这一定是GPT-4。」

### DeepMind - Sparrow

* 论文地址：https://arxiv.org/abs/2209.14375

### Anthropic - Claude

* 论文地址：https://arxiv.org/abs/2204.05862
* Claude vs. ChatGPT：https://scale.com/blog/chatgpt-vs-claude

### LAION - Open Assistant

* https://github.com/LAION-AI/Open-Assistant

### Google - Bard

## 四、各大公司

在 LLM 这个事情上，感觉梯队很明显，Google 应该是排在第二位，最能体现 Google 技术眼光的是 PaLM 和 Pathways，推出时间大概在 22 年 2 月到 4 月间，同一时期，OpenAI 推出的却是 InstructGPT，从这里就可以看出 Google 和 OpenAI 的差距了，至于为何这么说，你看了我后面的正文后大概能理解。DeepMind 之前的重心一直在强化学习攻克游戏和 AI for science 这些方面，切入LLM 其实很晚，应该是21 年才开始重视这个方向，目前也处于追赶状态。Meta 就更不用说了，重心一直不在 LLM 上，目前感觉也发力开始追赶。

### DeepMind：

### Meta：

### Google：

### Anthropic:

2019 年 OpenAI 与微软完成交易后，OpenAI 研究副总裁 Dario Amodei 不认同公司委身于微软，后于 2021 年包括 Amodei、GPT3 的 Lead Engineer Tom Brown 在内的 11 名 OpenAI 员工离职创立了 Anthropic。2022 年晚些时候，Google 向 Anthropic 投资了 3 亿美元。

与批评者认为 OpenAI 过于莽撞地推出 ChatGPT 而没有考虑足够的安全等问题形成对比的是，Anthropic 强调他们致力于建设「可靠、可控、可解释」的 AI 系统。

### OpenAI：

* **Alignment**: How can we understand what objective, if any, a model is best understood as pursuing? How do we increase the extent to which that objective is aligned with human preferences, such as via prompt design or fine-tuning?
* **Fairness and Representation**: How should performance criteria be established for fairness and representation in language models? How can language models be improved in order to effectively support the goals of fairness and representation in specific, deployed contexts?
* **Interdisciplinary Research**: How can AI development draw on insights from other disciplines such as philosophy, cognitive science, and sociolinguistics?
* **Interpretability / Transparency**: How do these models work, mechanistically? Can we identify what concepts they’re using, or extract latent knowledge from the model, make inferences about the training procedure, or predict surprising future behavior?
* **Misuse Potential**: How can systems like the API be misused? What sorts of ‘red teaming’ approaches can we develop to help us and other AI developers think about responsibly deploying technologies like this?
* **Model Exploration**: Models like those served by the API have a variety of capabilities which we have yet to explore. We’re excited by investigations in many areas including model limitations, linguistic properties, commonsense reasoning, and potential uses for many other problems.
* **Robustness**: Generative models have uneven capability surfaces, with the potential for surprisingly strong and surprisingly weak areas of capability. How robust are large generative models to "natural" perturbations in the prompt, such as phrasing the same idea in different ways or with/without typos? Can we predict the kinds of domains and tasks for which large generative models are more likely to be robust (or not robust), and how does this relate to the training data? Are there techniques we can use to predict and mitigate worst-case behavior? How can robustness be measured in the context of few-shot learning (e.g. across variations in prompts)? Can we train models so that they satisfy safety properties with a very high level of reliability, even under adversarial inputs?

## 参考

* https://huggingface.co/blog/dialog-agents
* https://yaofu.notion.site/A-Closer-Look-at-Large-Language-Models-Emergent-Abilities-493876b55df5479d80686f68a1abd72f
* [《ChatGPT 背后的“功臣”——RLHF 技术详解》](https://mp.weixin.qq.com/s/TLQ3TdrB5gLb697AFmjEYQ)
* https://platform.openai.com/docs/model-index-for-researchers
* https://openai.com/blog/instruction-following/#rfref18