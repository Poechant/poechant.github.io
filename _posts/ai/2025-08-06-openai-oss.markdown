---
layout: post
title: 时隔六年再开源，OpenAI 发布两款语言模型，主打端侧场景
date: 2025-08-06 08:01:10 +0800
categories: ai
tags: [AI, 人工智能, AGI, Agent, 大模型, 基础模型, 模型, OpenAI, OSS, 开源, 开放权重, Open Source, Open Weight, 端侧模型, Frontier]
description: 距离 OpenAI 上一次开源语言模型已经过去了 6 年 —— 北京时间 2025 年 8 月 6 日凌晨 00:22，美国人工智能公司 OpenAI 在官网发文，声称发布两款开放权重推理模型 ``gpt-oss-120b`` 和 ``gpt-oss-20b``，也是 ``GPT-2`` 之后的第一个开放权重语言模型。
excerpt: 距离 OpenAI 上一次开源语言模型已经过去了 6 年 —— 北京时间 2025 年 8 月 6 日凌晨 00:22，美国人工智能公司 OpenAI 在官网发文，声称发布两款开放权重推理模型 ``gpt-oss-120b`` 和 ``gpt-oss-20b``，也是 ``GPT-2`` 之后的第一个开放权重语言模型。
katex: True
location: 杭州
author: 麦克船长
pinned: no
---

![](/resources/2025/08/06/3.jpg)

距离 OpenAI 上一次开源语言模型已经过去了 6 年 —— 北京时间 2025 年 8 月 6 日凌晨 00:22，美国人工智能公司 OpenAI 在官网发文，声称发布两款开放权重推理模型 ``gpt-oss-120b`` 和 ``gpt-oss-20b``，也是 ``GPT-2`` 之后的第一个开放权重语言模型。

#### 先看看开源了啥？

- 模型权重：``gpt-oss-120b``、``gpt-oss-20b``（含 MXFP4 量化版），许可证 ``Apache-2.0``；
- 推理脚本：官方 PyTorch 版 + Apple Metal 优化版；
- 开发组件：Python/Rust 版 Harmony renderer；
- 分词器：``o200k_harmony``（≈20 万词），与 ``gpt-o4-mini`` / ``gpt-4o`` 系列同源，用于长上下文与工具调用。

#### 两款模型概览：120b 参数和 20b 参数、128k 上下文长度

- ``gpt-oss-120b``：36 层 Transformer + MoE，总参数 116.8b，每个 token 激活 5.1b 参数；单张 80 GB GPU 即可推理。
- ``gpt-oss-20b``：24 层，总参数 20.9b，每个 token 激活 3.6b 参数；只需 16 GB 内存即可运行。
- 二者均支持最长 128k 上下文窗口，采用旋转位置编码（RoPE）与分组多查询注意力（group size = 8）。
- 所有权重以 ``Apache-2.0`` 许可证发布，并已在 Hugging Face 上线。官方同时给出了 ``MXFP4`` 量化版（4.25 bit），分别缩减至 60.8 GB 与 12.8 GB，方便单机部署。

#### 性能：20b 即追平 o3-mini，120b 挑战 o4-mini

官方基准测试显示，``gpt-oss-120b`` 在 Codeforces、MMLU、HLE 等推理评测上可与 OpenAI 自家商用模型 ``gpt-o4-mini``「打成平手」，并在 AIME 竞赛数学、HealthBench 医疗问答上反超。体量仅为其六分之一的 ``gpt-oss-20b`` 亦与 ``gpt-o3-mini`` 水平相当，在部分数学与医疗任务中甚至略胜一筹。

更值得关注的是工具使用与代理式推理：在 Tau-Bench 评测套件中，两款模型都展示了强大的浏览器、Python 代码和函数调用能力，为未来本地化 Agent 场景奠定了基础。

#### 架构与训练细节：Transformer-MoE + 三档推理力度

- MoE 设计：120b 含 128 个专家、20b 含 32 个专家，每 token 仅激活 4 个专家，有效降低计算开销。
- 训练数据：以英文为主，偏重 STEM 与代码语料；使用全新的 ``o200k_harmony`` 分词器（同步开源）。
- 后训练：沿用 ``gpt-o4-mini`` 的「监督 SFT + 高算力 RL」流程，并额外加入「可变推理力度」训练，使模型能在低、中、高三档模式间切换，开发者仅需一行 system 指令即可指定 latency / accuracy 取舍。
- Harmony Chat Format：模型默认支持 OpenAI 最新的提示层级体系与结构化输出，便于与 Responses API 及外部 Agent 框架衔接。

#### 安全评估：首次公开「恶意微调」极限测试

伴随模型发布，OpenAI 同步发出 30 来页的《Estimating Worst-Case Frontier Risks of Open-Weight LLMs》。论文提出「Malicious Fine-Tuning（MFT）」方法：

- 先利用 RL 将模型的拒绝策略完全「拆除」；
- 再在生物、网络安全两个高风险域内，通过浏览器、终端等工具环境进行能力最大化训练。

结果显示，即便在 OpenAI 自研顶级训练栈下，MFT 后的 ``gpt-oss-120b`` 仍低于 Preparedness Framework 的 「High」 能力阈值，综合实力落后于闭源模型 o3。这一结论为「可公开发布」提供了直接佐证。

此外，OpenAI 宣布设立 50 万美元「Red Teaming Challenge」奖金池，邀请全球研究者挖掘新型安全问题，并将在赛后开源评测集与报告。

#### 生态与部署：从桌面到云端的「全链路打通」

- 合作伙伴覆盖云（Azure、AWS、Databricks 等）、推理框架（vLLM、llama.cpp、Ollama、LM Studio）、CDN/Edge（Cloudflare、Vercel）以及硬件厂商（NVIDIA、AMD、Cerebras、Groq）。
- 微软同步在 Windows 端上线 gpt-oss-20b 的 ONNX Runtime 版本，VS Code AI Toolkit 内可一键本地推理。

#### 结语

从性能、成本到安全规范，``gpt-oss`` 的发布无疑是 2025 年开源大模型领域最重磅的事件之一。它不仅象征着 OpenAI 重启「权重公开」路线，更向业界展示了「高能力 + 高安全」并非零和。随着更多社区开发者与企业加入测试与精调，``gpt-oss`` 能否像当年的 ``GPT-2`` 那样掀起新一轮创新浪潮，值得持续关注。

#### 参考
- Blog：``https://openai.com/index/introducing-gpt-oss/``
- Paper：``https://cdn.openai.com/pdf/231bf018-659a-494d-976c-2efdfc72b652/oai_gpt-oss_Model_Safety.pdf``
- Model Card：``https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf``
- Red Teaming Challenge：``https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/``
- HuggingFace：``https://huggingface.co/openai/gpt-oss-120b``


