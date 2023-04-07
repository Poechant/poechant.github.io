---
layout: post
title: 笔记本上就能跑的 ChatGPT-like 模型！
date:   2023-03-15 00:40:13 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, 语言模型, 大模型]
description: 
excerpt: 
katex: True
location: 杭州
author: 麦克船长
---

```
git clone https://github.com/antimatter15/alpaca.cpp
cd alpaca.cpp
```

下载训练好的数据集：

```
wget -O ggml-alpaca-7b-q4.bin -c https://gateway.estuary.tech/gw/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC
```

```
make chat
./chat
```