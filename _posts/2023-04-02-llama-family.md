---
layout: post
title: LLaMA 羊驼家族：大模型世界里的 Linux 内核
date:   2023-03-24 12:40:13 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, 语言模型, 大模型]
description: 
excerpt: 
katex: True
location: 杭州
author: 麦克船长
---

85060909

## 一、LLaMA

目前基于预训练 LLM 研发性能较好的对话机器人，其难点之一是高质量数据。目前调教开源模型，基本都在用 ChatGPT 作为老师，常用的方式就两种：

1. 自己准备问题，让老师给答案：研发团队准备一组问题（instruction），然后用 ChatGPT 根据这些问题生成答案（following），这样得到一组 instruction-following 数据作为微调的训练数据。这样做的一个前提是认为 ChatGPT 本身就是高质量对齐的模型。

2. 用公开收集的 ChatGPT 的高质量对话作为微调的训练数据，这里的一个好的参考是 ShareGPT，它是一个大量用户分享 ChatGPT 对话内容的网站。

### 1、LLaMA

* Blog：https://ai.facebook.com/blog/large-language-model-llama-meta-ai/

#### 1.1、在手机上跑一个 LLaMA.cpp

下载 [Android NDK](https://developer.android.com/ndk) 到你的本地电脑上，编译一个支持 Android 平台运行的 LLaMA 出来：

```shell
mikecaptain@local % mkdir build-android
mikecaptain@local % cd build-android
mikecaptain@local % export NDK=<your_ndk_directory>
mikecaptain@local % cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod ..
mikecaptain@local % make
```

在 Android 设备上安装 [Termux](https://termux.dev/) 

```shell
mikecaptain@local % git clone https://github.com/ggerganov/llama.cpp
mikecaptain@local % cd llama.cpp
mikecaptain@local % make
```

### 2、LLaMA 家族的 Alpaca-7B（2023 年 3 月 13 日）

Alpaca 是一个在 LLaMA-7B 基础上用 5.2 万条的「instruction-following」微调得到的 LLM，由 Stanford 大学的一个研究团队发布，训练总花费约不到 600 美元。

* Blog：https://crfm.stanford.edu/2023/03/13/alpaca.html
* DEMO：https://crfm.stanford.edu/alpaca/
* Code：https://github.com/tatsu-lab/stanford_alpaca

![](https://crfm.stanford.edu/static/img/posts/2023-03-13-alpaca/alpaca_main.jpg){: width="720"}

* 关于训练数据：发布在 https://github.com/tatsu-lab/stanford_alpaca#data-release ，数据生成过程 https://github.com/tatsu-lab/stanford_alpaca#data-generation-process

#### 2.1、Alpaca-7B

### 3、LLaMA 家族的 GPT4ALL

### 4、LLaMA 家族的 Vicuna-13B（2023 年 3 月 31 日）

* Blog：https://vicuna.lmsys.org/
* DEMO：https://chat.lmsys.org/
* 示例：https://vicuna.lmsys.org/eval/

Vicuna 是一个在 LLaMA-13B 基础上用了 7 万条 ShareGPT 的对话微调得到的 LLM，由加州大学伯克利分校（UCBerkeley）的一个研究团队主导，联合卡耐基梅隆大学（CMU）、斯坦福大学和加州大学圣地亚哥分校（UCSD）发布，训练总花费约 300 美元。

* 关于模型参数：截止 2023 年 4 月 3 日暂未发布。
* 关于训练数据：根据其官方网页上的声明，研发团队没有发布训练数据的计划。



![](/img/src/2023/04/2023-04-02-llama-01.png){: width="720"}

Clone this repository and navigate to FastChat folder:

```shell
mikecaptain@localhost % git clone https://github.com/lm-sys/FastChat.git
mikecaptain@localhost % cd FastChat
```

Install the latest main branch of huggingface/transformers:

```shell
mikecaptain@localhost % pip3 install git+https://github.com/huggingface/transformers
```

安装依赖

```shell
mikecaptain@localhost % pip3 install -e .
```

Launch a controller:

```shell
mikecaptain@localhost % python3 -m fastchat.serve.controller
```

Launch a model worker

```shell
mikecaptain@localhost % python3 -m fastchat.serve.model_worker --model-path facebook/opt-1.3b
```

本段参考

* https://zhuanlan.zhihu.com/p/618389519
* https://vicuna.lmsys.org/

### 5、LLaMA 的局限性

### 6、LLaMA 的其他版本

* Chinese Vicuna：https://github.com/Facico/Chinese-Vicuna





## 二、Cerebras-GPT：真·开源（2023 年 3 月 28 日）

3 月 28 日，芯片初创公司 Cerebras 发布开源大模型 Cereras-GPT 系列，包括 111M、256M、590M、1.3B、2.7B、6.7B、13B 七个参数规模的模型。

![](https://www.cerebras.net/wp-content/uploads/2023/03/Scaling-laws-blog-banner.png)

### 1、真·开源

模型架构、训练数据、预训练好的模型权重参数、Checkpoints、计算优化训练、License 等全部开源。

![](https://www.cerebras.net/wp-content/uploads/2023/03/Scaling-laws-blog-comparison.png)

### 2、计算优化训练 —— 训练效率提升

船长在[《麦克船长 LLM 革命系列 2：破晓》](https://www.mikecaptain.com/2023/03/06/captain-aigc-2-llm/)一文中介绍了 2020 年 1 月 OpenAI 提出的「Scaling Law」，指出了训练算力、训练数据规模、模型参数规模指数增长，模型性能表现线性提升的规律。

![](https://www.mikecaptain.com/img/src/2023/2023-01-23-captain-aigc-2-llm-10.png)

2022 年 DeepMind 在提出 Chinchilla（中文意思「龙猫」）模型时，顺便指出 LLM 的训练数据规模（training data）与训练算力（compute）之间的最优关系。

## 三、BloombergGPT（2023 年 3 月 30 日）

## 四、ChatRWKV：一个匹敌 Transformer 表现的并行化 RNN 模型

## 五、GLM

## 背景

### 1、Chinchilla AI

2022 年 3 月 DeepMind 在[《Training Compute-Optimal Large Language Models》](https://arxiv.org/abs/2203.15556)一文中提到：

> By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. <br/> 通过在 50~5000 亿个训练数据上（具体说是指 tokens 规模）训练 400 多个语言模型，模型参数规模从 7000 万到超过 160 亿个参数，DeepMind 发现对于计算优化训练（Compute-Optimal Traning），模型规模和训练数据规模应该等比缩放：模型规模翻倍时，训练数据量也应翻倍。

DeepMind 认为当时很多 LLM 的训练都是不充分的，在暴力堆大模型参数规模时，很多 LLM 的训练数据规模并没有相应放大。DeepMind 顺便在发布该文时提出了 Chinchilla 模型，其参数规模、训练数据规模对比如下：

| 模型 			| 模型参数规模 	| 训练数据规模 	|
|---------------|---------------|---------------|
| LaMDA 		| 137B			| 168B			|
| GPT-3 		| 175B			| 300B			|
| Jurassic		| 178B			| 300B			|
| Gopher 		| 280B			| 300B			|
| MT-NLG 530B	| 530B			| 270B			|
| Chinchilla 	| 70B 			| 1.4T			|

此前 DeepMind 还发布过一个名为 Gopher 的 LLM（在 Chinchilla 发布前两个月），发布 Chinchilla 时用了与 Gopher 同等规模的预算，但在用更小的模型参数规模情况下，用更多的训练数据，Chinchilla 取得了更好的效果。

发布大语言模型 Chinchilla，官方声称其表现可以超过 GPT-3，并且用更少的

### 2、MoE（Mixture of Experts）

混合专家系统（Mixture of Experts，简称 MoE）是把处理不同任务的不同神经网络专家系统（Expert Model），通过门控模型（Gating Model），组合成一个系统。针对下游任务，门控模型来决定使用哪个神经网络会得到更好表现。这个技术整体算是一种集成学习技术（Emsemble Learning）。

### 3、Physics of AI

#### 参考：

* https://www.youtube.com/watch?v=XLNmgviQHPA

## 参考

1. https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/
2. https://arxiv.org/pdf/2203.15556.pdf
3. https://vicuna.lmsys.org/
4. https://mp.weixin.qq.com/s/Q1jHC5b7NMvrOVTd9dIJCw
5. https://zhuanlan.zhihu.com/p/618776565
6. https://crfm.stanford.edu/2023/03/13/alpaca.html
7. https://medium.com/geekculture/list-of-open-sourced-fine-tuned-large-language-models-llm-8d95a2e0dc76