---
layout: post
title: 上船跑模型之一键运行 Alpaca.cpp：笔记本上就能跑的 LLaMA！
date:   2023-03-18 00:40:13 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, 语言模型, 大模型]
description: Alpaca 是 Stanford 的一个研究团队在 LLaMA 基础上用少量语料微调得到的开源模型，GitHub 上的 antimatter15/alpaca.cpp 是其 C++ 一键运行版本
excerpt: Alpaca 是 Stanford 的一个研究团队在 LLaMA 基础上用少量语料微调得到的开源模型，GitHub 上的 antimatter15/alpaca.cpp 是其 C++ 一键运行版本
katex: True
location: 杭州
author: 麦克船长
---

Clone 项目到本地

```shell
mikecaptain@CVN % git clone https://github.com/antimatter15/alpaca.cpp
mikecaptain@CVN % cd alpaca.cpp
```

下载训练好的数据集：

```shell
mikecaptain@CVN % wget -O ggml-alpaca-7b-q4.bin -c https://gateway.estuary.tech/gw/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC
```

编译、运行

```shell
mikecaptain@CVN % make chat
mikecaptain@CVN % ./chat
```