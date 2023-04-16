---
layout: post
title: 上船跑模型之 MacBook 上运行 LLaMA 7B 和 13B 原始模型
date:   2023-03-12 00:40:13 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, 语言模型, 大模型, LLaMA, Meta, GPT, 本地, MacBook, ChatGPT, 平替, OpenAI, 开源, 免费, 推理, 模型]
description: 简单几步，让你在你的 MacBook 上运行 LlaMA 7B 和 13B。
excerpt: 简单几步，让你在你的 MacBook 上运行 LlaMA 7B 和 13B。
katex: True
location: 杭州
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

## 一、LLaMA 预训练模型下载

LLaMA 是什么？关于 LLaMA 的介绍，看这篇[《Meta 推出开源 LLaMA，用 1/10 参数规模打败 GPT-3，群"模"乱舞的 2023 拉开序幕》](https://www.mikecaptain.com/2023/02/25/meta-llama/)。

聊了 LLaMA 后，接下来，下载 LLaMA 模型文件，这里略去下载地址（网上很多，但普遍地址有效生命周期不长）。

```
─ models
 ├── 7B
 │   ├── checklist.chk
 │   ├── consolidated.00.pth
 │   └── params.json
 ├── 13B
 │   ├── checklist.chk
 │   ├── consolidated.00.pth
 │   ├── consolidated.01.pth
 │   └── params.json
 ├── 30B
 │   ├── checklist.chk
 │   ├── consolidated.00.pth
 │   ├── consolidated.01.pth
 │   ├── consolidated.02.pth
 │   ├── consolidated.03.pth
 │   └── params.json
 ├── 65B
 │   ├── checklist.chk
 │   ├── consolidated.00.pth
 │   ├── consolidated.01.pth
 │   ├── consolidated.02.pth
 │   ├── consolidated.03.pth
 │   ├── consolidated.04.pth
 │   ├── consolidated.05.pth
 │   ├── consolidated.06.pth
 │   ├── consolidated.07.pth
 │   └── params.json
 └── tokenizer.model
```

因为文件太大，估计会出现多次断点续传，所以一定要校验下文件是否正确，正确与否与 checklist.chk 文件对比。比如：

```shell
mikecaptain@CVN % md5sum 7B/consolidated.00.pth
6efc8dab194ab59e49cd24be5574d85e
```

## 二、用 ggerganov/llama.cpp 运行

### 1、LLaMA 7B 版本

#### 1.1、下载 LLaMA.cpp 项目

```shell
mikecaptain@CVN % git clone git@github.com:ggerganov/llama.cpp.git
mikecaptain@CVN % cd llama.cpp
mikecaptain@CVN % make
```

#### 1.2、准备 llama 7B 环境

```shell
mikecaptain@CVN % conda install pytorch numpy sentencepiece 
```

The first script converts the model to "ggml FP16 format":

```shell
mikecaptain@CVN % python convert-pth-to-ggml.py models/7B/ 1
```

![](/img/src/2023/03/2023-03-12-llama-cpp-1.png)

这样会得到一个 13GB 的文件 `models/7B/ggml-model-f16.bin`，然后再运行脚本 `quantize` 用来把 `models/7B/ggml-model-f16.bin` 转为 `4-bits` 版本：

```shell
mikecaptain@CVN % ./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin 2
```

![](/img/src/2023/03/2023-03-12-llama-cpp-2.png)

这样会生成一个 3.9GB 的 `models/7B/ggml-model-q4_0.bin` 文件。

#### 1.3、运行 LLaMA 7B 模型

脚本 `main` 用于启动，使用刚得到的 `models/7B/ggml-model-q4_0.bin` 模型文件。

```shell
mikecaptain@CVN % ./main -m ./models/7B/ggml-model-q4_0.bin \
  -t 8 \
  -n 128 \
  -p 'The first man on the moon was '
```

![](/img/src/2023/03/2023-03-12-llama-cpp-4.png)

可以看到续写的内容是：

>  The first man on the moon was 38-years old.
Astronaut Neil Armstrong, at age 24, became a naval aviator flying fighter jets in World War II. He later flew for Pan American Airlines as a test pilot and flight engineer before he joined NASA's Mercury space program.
"Houston, Tranquillity Base here. The Eagle has landed." Armstrong spoke those words on July 20, 1969, as the United States beat Russia in the Cold War race to put a man on the moon. [end of text]

可以用 `./main --help` 来查看多有哪些参数可以设置。

![](/img/src/2023/03/2023-03-12-llama-cpp-3.png)

### 2、LLaMA 13B 版本

#### 2.1、准备 LLaMA 13B 环境

运行如下命令，生成一个 `ggml-model-f16.bin` 文件：

```shell
mikecaptain@CVN % python convert-pth-to-ggml.py models/13B/ 1
```

![](/img/src/2023/03/2023-03-12-llama-cpp-5.png)

再运行如下命令，生成文件 `./models/13B/ggml-model-q4_0.bin`：

```shell
mikecaptain@CVN % ./quantize ./models/13B/ggml-model-f16.bin   ./models/13B/ggml-model-q4_0.bin 2
```

![](/img/src/2023/03/2023-03-12-llama-cpp-6.png)

#### 2.2、运行 LLaMA 13B 模型

```shell
mkkecaptain@CVN % ./main \
  -m ./models/13B/ggml-model-q4_0.bin \
  -t 8 \
  -n 128 \
  -p 'Some good pun names for a coffee shop run by beavers:
-'
```

![](/img/src/2023/03/2023-03-12-llama-cpp-7.png)


<!-- ## 三、用 soulteary/llama-docker-playground 运行

### 1、下载 llama-docker-playground 项目

```shell
mikecaptain@CVN % git clone https://github.com/soulteary/llama-docker-playground.git
mikecaptain@CVN % cd llama-docker-playground
```

### 2、准备环境

### 3、运行模型 -->

## 参考

* [Running LLaMA 7B and 13B on a 64GB M2 MacBook Pro with llama.cpp](https://soulteary.com/2023/03/09/quick-start-llama-model-created-by-meta-research.html)
* [ggerganov - llama.cpp](https://github.com/ggerganov/llama.cpp)