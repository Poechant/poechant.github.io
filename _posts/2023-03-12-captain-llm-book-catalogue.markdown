---
layout: post
title: 书籍大纲《麦克船长 LLM 革命未来》
date:   2023-03-12 03:40:13 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, 语言模型, 大模型]
description: 
excerpt: 
katex: True
location: 上海
author: 麦克船长
---

<!-- 推荐语：
* 张凯夫
* 赵德丽：达摩院计算机视觉负责人、资深算法专家
* 熊皓：淘宝直播算法负责人，资深技术专家
* 黄眉：
* 三七：企业智能事业部总经理
* 锡泽：阿里集团资深
* 韩锷：资深技术专家
* 张斯成：前钉钉副总裁
* 罗璇
* 刘江？
* 赖晓春，上海科技大学研究员、联影首席科学家
* 牛力，上海交通大学
* 王猛，Google 自动驾驶 AI 研究员
* 田飞，Meta
* 赵明夷，Notion
* 孙鹏，世界之窗浏览器创始人
* 王宇航，核桃编程创始人
* 陆铭，宝云科技创始人


大模型革命：自然语言处理篇
演化：大语言模型编年史

--- -->

## 目录

#### 先导篇 · 自然语言处理的基本背景
* **第 1 章：人工智能皇冠的明珠**：自然语言处理
* **第 2 章：自然语言处理的任务**
<br/>

#### 第一部分 · 革命前夜：从统计语言模型到 Transformer

* **第 3 章：自然语言处理的「史前」时代**：2017 年之前模型
	* 统计语言模型
	* 感知机（Perceptron）
	* 卷积神经网络（CNN）
	* 循环神经网络（RNN）
	* 为什么说 RNN 模型没有体现「注意力」？
	* 基于 Attention 机制的 Encoder-Decoder 模型
	* 本章小节
* **第 4 章：Transformer 横空出世**：开启 NLP 新纪元的 2017 年
	* 自注意力机制（Self-Attention）
	* 多头注意力（Multi-Head Attention）
	* 退化现象、残差网络与 Short-Cut
	* 位置编码（Positional Embedding）
	* Transformer 的编码器
	* Transformer 的解码器
	* 本章小节
* **第 5 章：亲手实现 Transformer**：基于 TensorFlow 架构的 Python 实现
	* 先训练和测试一个 Transformer 实例
	* 超参数、预处理、数据加载、模型构建及训练
	* 编码器（Encoder）
	* 解码器（Decoder）
	* 编码和解码完成后的操作
	* 效果评价
	* 本章小节
<br/>

#### 第二部分 · 革命破晓：从 Transformer 到 GPT-4

* **第 6 章：神经语言模型的范式演进**
	* 第一阶段：完全监督学习（Fully Supervised Learning）范式
	* 第二阶段：预训练（Pre-train）范式 —— 为了更好的泛化性（Generalization）
	* 第三阶段：「预训练-人工反馈强化学习-提示（Pre-train, RLHF and Prompt）」学习范式
	* 本章小节

* **第 7 章：预训练大行其道**：从 ELMo 到 GPT-3（2018-2021）
	* **ELMo**：词所在的上下文很重要（2018 年 2 月）
	* **GPT**（2018 年 6 月）
	* **BERT**（2018 年 10 月）
	* **GPT-2**（2019 年 2 月）
	* **T5**：提出所有 NLP 任务可统一为文本生成任务（2019 年 10 月）
	* **缩放定律**（Scaling Law）：AI 时代的摩尔定律（2020 年 1 月）
	* **GPT-3**（2020 年 5 月）
	* 本章小节
* **第 8 章：大模型推理能力**：神奇的上下文学习（In-Context Learning）
	* ICL 能力的直接应用：Prompt Engineering
	* ICL 能力的底层假设：贝叶斯推理
	* ICL 是如何工作的？
	* 思维链（Chain of Thought，CoT）
	* 本章小节
* **第 9 章：预训练对齐人类**：从 InstructGPT 到 ChatGPT
	* **InstructGPT**：为对齐（Alignment）而生的指令式 GPT（2022 年 3 月）
	* **ChatGPT**：基于 RLHF 训练的对话式 GPT 模型（2022 年 11 月底）
	* 本章小节
* **第 10 章：多模态语言模型**
	* Visual ChatGPT
	* GPT-4
	* 本章小节
<br/>

#### 番外篇 · 革命征程：展望 AGI


<!-- <br/><br/>

---



* **神经网络模型训练的基本方法**

```
MLP 多层感知机、CNN 卷积神经网络、RNN 循环神经网络、LSTM 长短时记忆网络、基于注意力机制的 Encoder-Decoder
```
```
ELMo（2018）、GPT-1（2018）、BERT（2018）、GPT-2（2019）、T5（2019）、GPT-3（2020）
```
```
Prompt Engineering（PET、Hard Prompt & Soft Prompt、Prompt Tuning）、贝叶斯推理、WHY can it work、HOW does it work、思维链、涌现
```
```
InstructGPT、ChatGPT、GPT-4
```
 -->