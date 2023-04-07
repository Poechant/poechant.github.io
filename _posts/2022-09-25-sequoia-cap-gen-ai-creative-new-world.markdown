---
layout: post
title:  生成式 AI：一个创造性的新世界 from 红杉资本
date:   2022-09-25 07:14:01 +0800
categories: ai
tags: [AI, 人工智能, n-gram, N元文法]
description: 
excerpt: 
katex: True
location: 杭州
author: SONYA HUANG, PAT GRADY | [译] AI & 麦克船长
---

**本文目录**
* TOC
{:toc}

* 英文原文：https://www.sequoiacap.com/article/generative-ai-a-creative-new-world/
* 原文作者：SONYA HUANG, PAT GRADY
* 原文标题：**Generative AI: A Creative New World**

一种强大的新型大型语言模型使机器能够以可信的、有时甚至是超人的结果进行书写、编码、绘图和创造。

Humans are good at analyzing things. Machines are even better. Machines can analyze a set of data and find patterns in it for a multitude of use cases, whether it’s fraud or spam detection, forecasting the ETA of your delivery or predicting which TikTok video to show you next. They are getting smarter at these tasks. This is called “Analytical AI,” or traditional AI. 

人类善于分析事物。机器甚至更好。机器可以分析一组数据并在其中为多种用例找到模式，哪怕是电信诈骗或者垃圾邮件检测，预测您的交付时间或预测下一个要向您展示的 TikTok 视频。他们在这些任务上变得越来越聪明。这叫「分析型 AI（Analytical AI）」或者「传统 AI（traditional AI）」。

但人类不仅善于分析事物——我们也善于创造。 我们写诗、设计产品、制作游戏和编写代码。 过去，机器还没有机会在创造性工作（creative work）中与人类竞争 —— 它们只能从事分析和死记硬背的认知劳动（cognitive labor）。 但就在当下，机器开始善于创造感性和美丽的事物（sensical and beautiful things）。这个新类型的 AI 被称为「生成式 AI（Generative AI）」，这意味着机器正在生成新的东西，而不是分析已经存在的东西。

生成式 AI is well on the way to becoming not just faster and cheaper, but better in some cases than what humans create by hand. Every industry that requires humans to create original work—from social media to gaming, advertising to architecture, coding to graphic design, product design to law, marketing to sales—is up for reinvention. Certain functions may be completely replaced by generative AI, while others are more likely to thrive from a tight iterative creative cycle between human and machine—but generative AI should unlock better, faster and cheaper creation across a wide range of end markets. The dream is that generative AI brings the marginal cost of creation and knowledge work down towards zero, generating vast labor productivity and economic value—and commensurate 市值.

生成式人工智能不仅变得更快、更便宜，而且在某些情况下比人类手工创造的更好。 每个需要人类创造原创作品的行业——从社交媒体到游戏、广告到建筑、编码到平面设计、产品设计到法律、市场营销到销售——都需要重塑。 某些功能可能会被生成式 AI 完全取代，而其他功能则更有可能在人机之间紧密的迭代创意周期中蓬勃发展——但生成式 AI 应该在广泛的终端市场中解锁更好、更快和更便宜的创作。 梦想是生成式人工智能将创造和知识工作的边际成本降至零，产生巨大的劳动生产率和经济价值——以及相应的市值。

The fields that generative AI addresses—knowledge work and creative work—comprise billions of workers. Generative AI can make these workers at least 10% more efficient and/or creative: they become not only faster and more efficient, but more capable than before. Therefore, Generative AI has the potential to generate trillions of dollars of economic value.

生成式人工智能解决的领域——知识工作和创造性工作——包括数十亿工人。 生成式 AI 可以使这些员工的效率和/或创造力至少提高 10%：他们不仅变得更快、更有效率，而且比以前更有能力。 因此，生成式人工智能有可能产生数万亿美元的经济价值。

## Why Now? 

> Sure enough, as the models get bigger and bigger, they begin to deliver human-level, and then superhuman results.
> 果然，随着模型变得越来越大，它们开始提供人类水平，然后是超人的结果。

生成式 AI has the same “why now” as AI more broadly: better models, more data, more compute. The category is changing faster than we can capture, but it’s worth recounting recent history in broad strokes to put the current moment in context. 

生成式 AI 与更广泛的 AI 具有相同的“为什么是现在”：更好的模型、更多的数据、更多的计算。 该类别的变化速度快于我们可以捕捉到的速度，但值得粗略地回顾一下最近的历史，以便将当前时刻放在背景中。

**第 1 波：小模型占主导地位（2015 年之前）** 5 年多以前，小模型被认为是理解语言的 SOTA 技术。这些小型模型擅长分析任务，并被部署用于从交货时间预测（delivery time prediction）到欺诈检测（fraud classification）的工作。然而，它们对于通用生成任务的表现力不够。生成人类水平的写作或代码仍然是白日梦（a pipe dream）。

**第 2 波：规模化竞赛（2015 年至今）** Google Research 的一篇具有里程碑意义的论文（Attention is All You Need）描述了一种用于自然语言理解的新神经网络架构，称为 transformers，它可以生成高质量的语言模型，同时更可并行化，需要更少的训练时间。这些模型是少样本学习者，可以相对容易地针对特定领域进行定制。

![image](/img/src/2022/large-ai-models-vs-human-performance-920.png.jpg)

随着 AI 模型逐渐变大，它们已经开始超越主要的人类在很多任务上的表现水平。资料来源：© 伦敦经济学人报业有限公司，2022 年 6 月 11 日。保留所有权利； SCIENCE.ORG/CONTENT/ARTICLE/COMPUTERS-ACE-IQ-TESTS-STILL-MAKE-DUMB-MISTAKES-CAN-DIFFERENT-TESTS-HELP

**第 3 波：更好、更快、更便宜（2022+）** 计算变得更便宜。扩散模型等新技术可降低训练和运行推理所需的成本。研究界继续开发更好的算法和更大的模型。 开发人员访问权限从封闭测试版（closed beta）扩展到开放测试版（open beta），或者在某些情况下，开放源代码。

the floodgates are now open for exploration and application development. Applications begin to bloom.

对于那些一直渴望能用上 LLM 的 developers 而言，闸门现在已经打开，可以进行探索和应用程序开发。 申请开始大量涌现。

![image](/img/src/2022/sonya_cute_robots_singing_music_fantasy_cityscape_with_flowers__508554ad-db40-421d-87b6-4a188604f226.png.jpg)

用 MIDJOURNEY 生成的插图

**第 4 波：杀手级应用出现（现在）** 随着平台层的巩固，模型继续变得更好/更快/更便宜，以及模型访问趋向于免费和开源，应用层已经成熟，可以爆发创造力。（With the platform layer solidifying, models continuing to get better/faster/cheaper, and model access trending to free and open source, the application layer is ripe for an explosion of creativity.）

正如移动设备通过 GPS、相机和移动连接等新功能释放出新型应用程序一样，我们预计这些大型模型将激发新一波生成式 AI 应用出现。正如十年前移动的拐点为少数杀手级应用创造了市场空缺一样，我们预计杀手级应用也会出现在生成 AI 领域。The race is on！

## 市场格局（Market Landscape）

下面的示意图描述了将为每个类别提供支持的平台层，以及将在其上构建的潜在应用程序类型。

![image](/img/src/2022/sequoiacap-01.png)

#### 模型（Models）

* **Text** is the most advanced domain. However, natural language is hard to get right, and quality matters. Today, the models are decently good at generic short/medium-form writing (but even so, they are typically used for iteration or first drafts). Over time, as the models get better, we should expect to see higher quality outputs, longer-form content, and better vertical-specific tuning.

* **Images** are a more recent phenomenon, but they have gone viral: it’s much more fun to share generated images on Twitter than text! We are seeing the advent of image models with different aesthetic styles, and different techniques for editing and modifying generated images.
* **Speech synthesis** has been around for a while (hello Siri!) but consumer and enterprise applications are just getting good. For high-end applications like film and podcasts the bar is quite high for one-shot human quality speech that doesn’t sound mechanical. But just like with images, today’s models provide a starting point for further refinement or final output for utilitarian applications.
* **Video and 3D models** are coming up the curve quickly. People are excited about these models’ potential to unlock large creative markets like cinema, gaming, VR, architecture and physical product design. The research orgs are releasing foundational 3D and video models as we speak. 


* **文本（Text）** 是最高级的域。但是想把自然语言搞正确真的很难，质量实在太重要了。 今天，这些模型非常擅长通用的短/中型写作（但即便如此，它们通常用于迭代或初稿）。 随着时间的推移，随着模型变得更好，我们应该期望看到更高质量的输出、更长格式的内容和更好的垂直特定调整。
* **代码生成（Code Generation）** 很可能在短期内对开发人员的生产力产生重大影响，如 GitHub CoPilot 所示。 它还将使非开发人员更容易创造性地使用代码。
* **图片（Images）** 是最近才出现的现象，但它们已经开始病毒式传播了：在 Twitter 上分享生成的图片比分享文字有趣得多！ 我们看到了具有不同美学风格的图像模型的出现，以及用于编辑和修改生成图像的不同技术。
* **语音合成（Speech Synthesis）** 已经存在了一段时间（Hello Siri！），但消费者和企业应用程序才刚刚好起来。 对于像电影和播客这样的高端应用，对于听起来不机械的一次性人类质量语音来说，标准是相当高的。 但就像图像一样，今天的模型为进一步完善或实用应用程序的最终输出提供了一个起点。
* **视频和 3D 模型（Video and 3D models）**正在快速上升。 人们对这些模型打开电影、游戏、虚拟现实、建筑和实体产品设计等大型创意市场的潜力感到兴奋。 正如我们所说，研究机构正在发布基础 3D 和视频模型。
* **其他领域**：从音频和音乐，到生物学和化学（生成蛋白质和分子），许多领域都在进行基础模型研发。

The below chart illustrates a timeline for how we might expect to see fundamental models progress and the associated applications that become possible. 2025 and beyond is just a guess.

下图说明了我们如何期望看到基本模型的进展和相关应用成为可能的时间表。 2025 年及以后只是一个猜测。

![image](/img/src/2022/sequoiacap-02.png)

#### 应用（Applications）

以下是一些我们感到兴奋的应用程序。远远超过我们在此页面上捕获的内容（and we are enthralled by the creative applications that founders and developers are dreaming up.），我们被创始人和开发人员梦想的创意应用程序所吸引。

* **Copywriting**: The growing need for personalized web and email content to fuel sales and marketing strategies as well as customer support are perfect applications for language models. The short form and stylized nature of the verbiage combined with the time and cost pressures on these teams should drive demand for automated and augmented solutions.
* **Vertical specific writing assistants**: Most writing assistants today are horizontal; we believe there is an opportunity to build much better generative applications for specific end markets, from legal contract writing to screenwriting. Product differentiation here is in the fine-tuning of the models and UX patterns for particular workflows. 
* **Code generation**: Current applications turbocharge developers and make them much more productive: GitHub Copilot is now generating nearly 40% of code in the projects where it is installed. But the even bigger opportunity may be opening up access to coding for consumers. Learning to prompt may become the ultimate high-level programming language.
* **Art generation**: The entire world of art history and pop cultures is now encoded in these large models, allowing anyone to explore themes and styles at will that previously would have taken a lifetime to master.
* **Gaming**: The dream is using natural language to create complex scenes or models that are riggable; that end state is probably a long way off, but there are more immediate options that are more actionable in the near term such as generating textures and skybox art.  
* **Media/Advertising**: Imagine the potential to automate agency work and optimize ad copy and creative on the fly for consumers. Great opportunities here for multi-modal generation that pairs sell messages with complementary visuals.
* **Design**: Prototyping digital and physical products is a labor-intensive and iterative process. High-fidelity renderings from rough sketches and prompts are already a reality. As 3-D models become available the generative design process will extend up through manufacturing and production—text to object. Your next iPhone app or sneakers may be designed by a machine.
* **Social media and digital communities**: Are there new ways of expressing ourselves using generative tools? New applications like Midjourney are creating new social experiences as consumers learn to create in public.

* **文案写作**：对个性化 Web 和电子邮件内容的需求不断增长，以推动销售和营销策略以及客户支持，这些都是语言模型的完美应用。 措辞的简短形式和程式化性质，加上这些团队面临的时间和成本压力，应该会推动对自动化和增强解决方案的需求。
* **垂直特定的写作助手**：今天的大多数写作助手都是水平的； 我们相信有机会为特定的终端市场构建更好的生成应用程序，从法律合同编写到编剧。 这里的产品差异化在于针对特定工作流的模型和 UX 模式的微调。
* **代码生成（Code Generation）**：当前的应用程序可加速开发人员并提高他们的工作效率：GitHub Copilot 现在在安装它的项目中生成近 40% 的代码。 但更大的机会可能是为消费者开放编码。 学习提示可能会成为最终的高级编程语言。
* **艺术一代**：整个艺术史和流行文化世界现在都被编码在这些大型模型中，允许任何人随意探索以前需要一生才能掌握的主题和风格。
* **游戏**：梦想是使用自然语言来创建可操纵的复杂场景或模型； 最终状态可能还有很长的路要走，但有更直接的选择，在短期内更具可操作性，例如生成纹理和天空盒艺术。
* **媒体/广告**：想象一下自动化代理工作并为消费者即时优化广告文案和创意的潜力。 多模态生成的绝佳机会将销售信息与互补的视觉效果配对。
* **设计**：制作数字和实体产品的原型是一个劳动密集型的迭代过程。 草图和提示的高保真效果图已经成为现实。 随着 3D 模型的出现，生成式设计过程将延伸到制造和生产——从文本到对象。 你的下一个 iPhone 应用程序或运动鞋可能是由机器设计的。
* **社交媒体和数字社区**：是否有使用生成工具表达自我的新方式？ 随着消费者学会在公共场合创作，像 Midjourney 这样的新应用正在创造新的社交体验。

![image](/img/src/2022/randobot_How_to_make_an_amazing_generative_ai_app_in_an_early_modern_style_1-920.jpg.jpg)

用 MIDJOURNEY 生成的插图

> The best Generative AI companies can generate a sustainable competitive advantage by executing relentlessly on the flywheel between user engagement/data and model performance.
> 最好的生成式 AI 公司可以通过在用户参与/数据和模型性能之间的飞轮上不懈地执行来产生可持续的竞争优势。

## 生成式 AI 应用剖析（Anatomy of a Generative AI Application ）

What will a generative AI application look like? Here are some predictions.  

生成式 AI 应用程序会是什么样子？ 这里有一些预测。

#### 智能和模型微调（Intelligence and model fine-tuning）

Generative AI apps are built on top of large models like GPT-3 or Stable Diffusion. As these applications get more user data, they can fine-tune their models to: 1) improve model quality/performance for their specific problem space and; 2) decrease model size/costs.

生成式 AI 应用程序构建在 GPT-3 或 Stable Diffusion 等大型模型之上。 随着这些应用程序获得更多的用户数据，他们可以微调他们的模型以：1）针对他们的特定问题空间提高模型质量/性能； 2) 减小模型尺寸/成本。

We can think of Generative AI apps as a UI layer and “little brain” that sits on top of the “big brain” that is the large general-purpose models.  

我们可以将生成式 AI 应用程序视为 UI 层和位于大型通用模型“大脑”之上的“小大脑”。

#### 构成因素（Form Factor）

Today, Generative AI apps largely exist as plugins in existing software ecosystems. Code completions happen in your IDE; image generations happen in Figma or Photoshop; even Discord bots are the vessel to inject generative AI into digital/social communities.  

如今，生成式 AI 应用程序主要作为现有软件生态系统中的插件存在。 代码补全发生在您的 IDE 中； 图像生成发生在 Figma 或 Photoshop 中； 甚至 Discord 机器人也是将生成 AI 注入数字/社交社区的工具。

还有少量独立的生成式 Web 应用，例如用于文案写作的 Jasper 和 Copy.ai、用于视频编辑的 Runway 和用于记笔记的 Mem。

A plugin may be an effective wedge into bootstrapping your own application, and it may be a savvy way to surmount the chicken-and-egg problem of user data and model quality (you need distribution to get enough usage to improve your models; you need good models to attract users). We have seen this distribution strategy pay off in other market categories, like consumer/social.

插件可能是引导您自己的应用程序的有效楔子，它可能是克服用户数据和模型质量“先有鸡还是先有蛋”问题的明智方法（您需要分发以获得足够的使用率来改进您的模型；您需要 好的模型才能吸引用户）。 我们已经看到这种分销策略在其他市场类别中得到了回报，例如消费者/社交。

#### 交互范式（Paradigm of Interaction）

Today, most Generative AI demos are “one-and-done”: you offer an input, the machine spits out an output, and you can keep it or throw it away and try again. Increasingly, the models are becoming more iterative, where you can work with the outputs to modify, finesse, uplevel and generate variations.  

如今，大多数生成式 AI 演示都是“一劳永逸”的：你提供一个输入，机器吐出一个输出，你可以保留它或扔掉它，然后再试一次。 模型的迭代性越来越强，您可以在其中使用输出来修改、优化、升级和生成变体。

and they are great at suggesting first drafts that need to be finessed by a user to reach the final state (e.g. blog posts or code autocompletions). As the models get smarter, partially off the back of user data, we should expect these drafts to get better and better and better, until they are good enough to use as the final product.  

如今，生成式 AI 的输出被用作原型或初稿。 这些应用非常擅长吐出多种不同的想法来推动创意过程（例如，给出 logo 或者建筑设计的不同选择），并且它们非常擅长建议需要用户精心设计才能达到最终状态的初稿（例如 博客文章或代码自动完成）。 随着模型变得越来越智能，部分脱离了用户数据的支持，我们应该期望这些草稿会变得越来越好，直到它们足够好用作最终产品。

#### 可持续的 Category 领导力（Sustained Category Leadership）

The best Generative AI companies can generate a sustainable competitive advantage by executing relentlessly on the flywheel between user engagement/data and model performance. To win, teams have to get this flywheel going by 1) having exceptional user engagement → 2) turning more user engagement into better model performance (prompt improvements, model fine-tuning, user choices as labeled training data) → 3) using great model performance to drive more user growth and engagement. They will likely go into specific problem spaces (e.g., code, design, gaming) rather than trying to be everything to everyone. They will likely first integrate deeply into applications for leverage and distribution and later attempt to replace the incumbent applications with AI-native workflows. It will take time to build these applications the right way to accumulate users and data, but we believe the best ones will be durable and have a chance to become massive. 

最好的生成人工智能公司可以通过在用户参与/数据和模型性能之间的飞轮上不懈地执行来产生可持续的竞争优势。 为了获胜，团队必须通过 1) 拥有出色的用户参与度 → 2) 将更多的用户参与度转化为更好的模型性能（及时改进、模型微调、用户选择作为标记的训练数据）→ 3) 使用出色的模型 性能以推动更多的用户增长和参与。 他们可能会进入特定的问题领域（例如，代码、设计、游戏），而不是试图成为所有人的一切。 他们可能会首先深入集成到应用程序中以进行利用和分发，然后尝试用 AI 原生工作流替换现有应用程序。 以正确的方式构建这些应用程序以积累用户和数据需要时间，但我们相信最好的应用程序将经久耐用，并有机会变得庞大。

> Generative AI is still very early. The platform layer is just getting good, and the application space has barely gotten going. 
> 生成式 AI 还很早。 平台层刚刚好起来，应用空间才刚刚起步。

## 障碍和风险（Hurdles and Risks）

Despite Generative AI’s potential, there are plenty of kinks around business models and technology to iron out. Questions over important issues like copyright, trust & safety and costs are far from resolved.

尽管生成式 AI 具有潜力，但围绕商业模式和技术还有很多问题需要解决。 关于版权、信任和安全以及成本等重要问题的问题远未解决。

## 睁大眼睛（Eyes Wide Open）

生成式 AI 领域仍然处于早期。平台层刚刚好起来，应用空间才刚刚起步。

To be clear, we don’t need large language models to write a Tolstoy novel to make good use of Generative AI. These models are good enough today to write first drafts of blog posts and generate prototypes of logos and product interfaces. There is a wealth of value creation that will happen in the near-to-medium-term.

需要明确的是，我们不需要大型语言模型来编写托尔斯泰小说来充分利用生成式人工智能。 这些模型在今天足以编写博客文章的初稿并生成徽标和产品界面的原型。 中短期内将创造大量价值。

This first wave of Generative AI applications resembles the mobile application landscape when the iPhone first came out—somewhat gimmicky and thin, with unclear competitive differentiation and business models. However, some of these applications provide an interesting glimpse into what the future may hold. Once you see a machine produce complex functioning code or brilliant images, it’s hard to imagine a future where machines don’t play a fundamental role in how we work and create. 

第一波生成式 AI 应用程序类似于 iPhone 刚问世时的移动应用程序格局——有些噱头和单薄，竞争差异化和商业模式不明确。 然而，其中一些应用程序提供了一个有趣的一瞥未来可能会发生什么。 一旦您看到一台机器生成复杂的功能代码或出色的图像，就很难想象未来机器不会在我们的工作和创造方式中发挥基础性作用。

If we allow ourselves to dream multiple decades out, then it’s easy to imagine a future where Generative AI is deeply embedded in how we work, create and play: memos that write themselves; 3D print anything you can imagine; go from text to Pixar film; Roblox-like gaming experiences that generate rich worlds as quickly as we can dream them up. While these experiences may seem like science fiction today, the rate of progress is incredibly high—we have gone from narrow language models to code auto-complete in several years—and if we continue along this rate of change and follow a “Large Model Moore’s Law,” then these far-fetched scenarios may just enter the realm of the possible. 

如果我们允许自己梦想几十年后，那么很容易想象一个未来，在这个未来，生成式人工智能将深深植根于我们的工作、创造和娱乐方式：自己写的备忘录； 3D 打印任何你能想象到的东西； 从文本到皮克斯电影； 类似于 Roblox 的游戏体验，可以像我们想象的那样快速生成丰富的世界。 虽然这些经历在今天看起来像是科幻小说，但进展速度非常快——我们已经在几年内从狭义语言模型发展到代码自动完成——如果我们继续这种变化速度并遵循“大模型摩尔” 法”，那么这些牵强附会的情景可能就进入了可能的境界。

## 创业征集（Call for Startups）

We are at the beginning of a platform shift in technology. We have already made a number of investments in this landscape and are galvanized by the ambitious founders building in this space.

我们正处于技术平台转变的开端。 我们已经在这一领域进行了大量投资，并受到雄心勃勃的创始人在这一领域的建设的鼓舞。

If you are a founder and would like to meet, shoot us a note at sonya@sequoiacap.com and grady@sequoiacap.com.

如果您是创始人并希望见面，请通过 sonya@sequoiacap.com 和 grady@sequoiacap.com 给我们留言。

We can’t wait to hear your story.

我们迫不及待想听听您的故事。

> PS: This piece was co-written with GPT-3. GPT-3 did not spit out the entire article, but it was responsible for combating writer’s block, generating entire sentences and paragraphs of text, and brainstorming different use cases for generative AI. Writing this piece with GPT-3 was a nice taste of the human-computer co-creation interactions that may form the new normal. We also generated illustrations for this post with Midjourney, which was SO MUCH FUN!
> PS：这篇文章是与 GPT-3 合写的。 GPT-3 并没有吐出整篇文章，但它负责打击写作障碍，生成整个句子和文本段落，并为生成 AI 集思广益不同的用例。 用 GPT-3 写这篇文章是对可能形成新常态的人机共创交互的一种很好的体验。 我们还使用 Midjourney 为这篇文章制作了插图，这非常有趣！