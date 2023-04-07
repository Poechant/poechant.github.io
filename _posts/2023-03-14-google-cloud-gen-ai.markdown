---
layout: post
title: Google Cloud 为开发者、企业和政府带来了生成式 AI
date:   2023-03-12 03:40:13 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, 语言模型, 大模型]
description: 
excerpt: 
katex: True
location: 上海
author: 麦克船长
---

* 原文标题：Google Cloud brings generative AI to developers, businesses, and governments
* 原文链接：https://cloud.google.com/blog/products/ai-machine-learning/generative-ai-for-businesses-and-governments

Generative AI is poised to usher in a new wave of interactive, multimodal experiences that transform how we interact with information, brands, and one another. Harnessing the power of decades of Google’s research, innovation, and investment in AI, Google Cloud is bringing businesses and governments the ability to generate text, images, code, videos, audio, and more from simple natural language prompts. 

生成式 AI（Gen-AI）有望迎来新一波交互式、多模态体验，改变我们与信息、品牌和彼此互动的方式。 利用 Google 数十年来在 AI 方面的研究、创新和投资，Google Cloud 使企业和政府能够根据简单的自然语言提示生成文本、图像、代码、视频、音频等。

Realizing the potential of this technology means putting it in the hands of every developer, business, and government. To date, it has been difficult for organizations to access generative AI, let alone customize it, and at times the technology is prone to producing inaccurate information that could undermine trust. A decade ago, mobile ecosystems soared as businesses and developers gained safe, secure, and powerful tools suited to new form factors, interfaces, and interactions — and likewise, for generative AI to blossom, organizations need a new generation of tools that make it simple to build generative AI applications, or gen apps. 

实现这项技术的潜力意味着将它交到每个开发人员、企业和政府的手中。 迄今为止，组织很难访问生成式人工智能，更不用说定制它了，而且有时该技术容易产生可能破坏信任的不准确信息。 十年前，随着企业和开发人员获得适用于新外形、界面和交互的安全、可靠和强大的工具，移动生态系统蓬勃发展——同样，为了使生成人工智能蓬勃发展，组织需要新一代工具来简化它 构建生成式 AI 应用程序或生成应用程序。

To address these needs, Google Cloud will launch a range of products that infuse generative AI into our offerings, empowering developers to responsibly build with enterprise-level safety, security, and privacy. This journey starts today with the introduction of two new technologies:

为了满足这些需求，谷歌云将推出一系列产品，将生成式人工智能融入我们的产品中，使开发人员能够负责任地构建具有企业级安全、保障和隐私的产品。 这一旅程从今天开始引入两项新技术：

* Generative AI support in Vertex AI gives data science teams access to foundation models from Google and others, letting them build and customize atop these models on the same platform they use for homegrown ML models and MLOps. 
* Vertex AI 中的生成式 AI 支持使数据科学团队能够访问来自 Google 和其他公司的基础模型，让他们在用于本地 ML 模型和 MLOps 的同一平台上构建和自定义这些模型。
* Generative AI App Builder allows developers to quickly ship new experiences including bots, chat interfaces, custom search engines, digital assistants, and more. Developers have API access to Google’s foundation models and can use out-of-the-box templates to jumpstart the creation of gen apps in minutes or hours.
* Generative AI App Builder 允许开发人员快速交付新体验，包括机器人、聊天界面、自定义搜索引擎、数字助理等。 开发人员可以通过 API 访问 Google 的基础模型，并可以使用开箱即用的模板在几分钟或几小时内快速启动生成应用程序的创建。

Across these, Google Cloud ensures organizations have complete control over if, how, and for what their data is used. 

在这些方面，谷歌云确保组织可以完全控制他们的数据是否、如何以及用于什么目的。

“Google Cloud is bringing decades of AI research, innovation, and investment to the world with the launch of Generative AI support in Vertex AI and Generative AI App Builder,” said Ritu Jyoti, Group Vice President, Worldwide Artificial Intelligence (AI) and Automation Research, IDC. “With this, Google Cloud is poised to enable a whole new generation of builders, innovators, developers and doers to harness the power of AI in novel ways. Google Cloud's vision to empower teams, transform industries and truly change the world for the better is inspiring, and what sets them apart is their thoughtful yet bold approach, grounded in their deep commitment to responsibility.” 

全球人工智能 (AI) 和自动化集团副总裁 Ritu Jyoti 表示：“通过在 Vertex AI 和 Generative AI App Builder 中推出 Generative AI 支持，Google Cloud 正在为世界带来数十年的 AI 研究、创新和投资。” 研究，国际数据中心。 “有了这个，谷歌云准备让全新一代的建设者、创新者、开发者和实干家以新颖的方式利用人工智能的力量。 Google Cloud 赋予团队权力、转变行业并真正让世界变得更美好的愿景令人鼓舞，而让他们与众不同的是他们周到而大胆的方法，这些方法基于他们对责任的坚定承诺。”

## Build, tune, and deploy foundation models with Vertex AI

[Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-launches-vertex-ai-unified-platform-for-mlops), Google Cloud’s machine learning platform for training and deploying ML models and AI applications, is getting its biggest upgrade ever.

Vertex AI 是 Google Cloud 用于训练和部署 ML 模型和 AI 应用程序的机器学习平台，正在进行有史以来最大的升级。

Generative AI support in Vertex AI offers the simplest way for data science teams to take advantage of foundation models like PaLM, in a way that provides them with the most choice and control, including the ability to: 

Vertex AI 中的生成式 AI 支持为数据科学团队提供了利用 PaLM 等基础模型的最简单方式，为他们提供最多的选择和控制，包括能够：

* Choose the use case you want to solve for. Developers can now easily access PaLM API on Vertex AI to immediately address use cases such as content generation, chat, summarization, classification, and more. 
* 选择您要解决的用例。 开发人员现在可以轻松访问 Vertex AI 上的 PaLM API，以立即处理内容生成、聊天、摘要、分类等用例。
* Choose from Google’s latest foundation models. Options will include models invented by Google Research and DeepMind, and support for a variety of data formats, including text, image, video, code, and audio.
* 从 Google 的最新基础模型中进行选择。 选项将包括由 Google Research 和 DeepMind 发明的模型，并支持多种数据格式，包括文本、图像、视频、代码和音频。
* Choose from a variety of models. Over time, Vertex AI will support open-source and third-party models. With the widest variety of model types and sizes available in one place, Vertex AI gives customers the flexibility to use the best resource for their business needs. 
* 从多种型号中进行选择。 随着时间的推移，Vertex AI 将支持开源和第三方模型。 Vertex AI 可在一处提供最广泛的模型类型和尺寸，使客户能够灵活地使用最佳资源来满足其业务需求。
* Choose how to tune, customize, and optimize prompts. Use business data to increase the relevance of foundation model output and maintain control over costs, while ensuring data sovereignty and privacy.
* 选择如何调整、自定义和优化提示。 使用业务数据来增加基础模型输出的相关性并保持对成本的控制，同时确保数据主权和隐私。
* Choose how to engage with models. Whether via notebooks, APIs, or interactive prompts, a variety of tools lets developers, data scientists, and data engineers all contribute to building gen apps and customized models. 
* 选择如何与模特互动。 无论是通过笔记本、API 还是交互式提示，各种工具都可以让开发人员、数据科学家和数据工程师都为构建生成应用程序和定制模型做出贡献。

“Since its launch, Vertex AI has helped transform the way CNA scales AI, better managing machine learning models in production,” said Santosh Bardwaj, SVP, Global Chief Data & Analytics Officer at CNA Insurance. “With Generative AI support in Vertex AI, CNA can now tailor its insights to best suit the unique business needs of customers and colleagues.”

“自推出以来，Vertex AI 帮助改变了 CNA 扩展 AI 的方式，更好地管理生产中的机器学习模型，”CNA Insurance 高级副总裁兼全球首席数据与分析官 Santosh Bardwaj 说。 “借助 Vertex AI 对生成式 AI 的支持，CNA 现在可以定制其见解，以最好地满足客户和同事的独特业务需求。”

![](https://storage.googleapis.com/gweb-cloudblog-publish/images/SummitAI_YT_thumb_1280x720_Vertex_v3.max-1300x1300.jpg)

Starting today, trusted testers are accessing Generative AI support in Vertex AI. If you are interested in updates on our early access opportunities, please join our technical community, [Google Cloud Innovators](https://cloud.google.com/ai).

从今天开始，受信任的测试人员可以访问 Vertex AI 中的生成式 AI 支持。 如果您对我们的早期访问机会的更新感兴趣，请加入我们的技术社区 Google Cloud Innovators。

## Build applications in minutes or hours with Generative AI App Builder
## 使用 Generative AI App Builder 在几分钟或几小时内构建应用程序

Businesses and governments also want to make customer, partner, and employee interactions more effective and helpful with this new AI technology. To enable this, we are announcing our new Generative AI App Builder, the fastest way for developers to jumpstart the creation of gen apps such as bots, chat apps, digital assistants, custom search engines, and more, with limited technical expertise required. It lets developers:

企业和政府还希望通过这种新的 AI 技术使客户、合作伙伴和员工的互动更有效、更有帮助。 为实现这一点，我们宣布推出新的 Generative AI App Builder，这是开发人员快速开始创建 gen 应用程序（例如机器人、聊天应用程序、数字助理、自定义搜索引擎等）的最快方式，只需要有限的技术专业知识。 它让开发人员：

* Build in minutes or hours — not weeks or years. Developers can get going quickly with direct API access to foundation models and out-of-the-box templates for major use cases, including search, support, product recommendations and discovery, and media creation. Additionally, pre-built connectors let developers integrate their data with the intelligence of foundation models, all while keeping data private.
* 在数分钟或数小时内完成构建，而不是数周或数年。 开发人员可以通过直接 API 访问基础模型和适用于主要用例（包括搜索、支持、产品推荐和发现以及媒体创建）的开箱即用模板来快速上手。 此外，预建连接器让开发人员能够将他们的数据与基础模型的智能集成，同时保持数据的私密性。
* Combine organizational data and information retrieval techniques to provide relevant answers. Organizations can now build apps that infer the intent of a user’s question, surface proprietary data alongside relevant information from the foundation model, and serve responses with the required citations and attributions, all while ensuring data isolation and sovereignty.
* 结合组织数据和信息检索技术以提供相关答案。组织现在可以构建应用程序来推断用户问题的意图，显示专有数据以及来自基础模型的相关信息，并提供具有所需引用和属性的响应，同时确保数据隔离和主权。
* Search and respond with more than just text. Customers can type, talk, tap, and submit images when they interact — and bots, assistants, and other gen apps can reply with text, voice, and media. 
* 搜索和回复不仅仅是文本。 客户可以在互动时打字、交谈、点击和提交图像——而机器人、助手和其他生成应用程序可以用文本、语音和媒体进行回复。
* Combine natural conversations with structured flows. With granular control, developers can now blend the output of foundation models with step-by-step conversation orchestration to guide customers to the right answers, no matter the duration of engagement.
* 将自然对话与结构化流程相结合。 通过精细控制，开发人员现在可以将基础模型的输出与逐步对话编排相结合，以引导客户找到正确的答案，无论参与时间长短。
* Don’t just inform — transact. Beyond just serving content, digital assistants and bots can connect to purchasing and provisioning systems, and escalate customer conversations to a human agent when the context demands. 
* 不要只是通知——交易。 除了提供内容之外，数字助理和机器人还可以连接到采购和供应系统，并在上下文需要时将客户对话升级为人工代理。

“With the growing popularity of voice assistants in everyday life, consumers increasingly expect accurate and consistent voice interactions. The integration of Google Cloud’s industry-leading, AI-based speech services are already providing high-quality voice services to Toyota and Lexus customers," said Steve Basra, CEO and president, Toyota Connected North America. "With these latest generative AI announcements, we’re excited to expand our partnership and explore how foundation language models can further our vision to bring the best and most innovative in-car experiences to drivers.”

“随着语音助手在日常生活中的日益普及，消费者越来越期望准确和一致的语音交互。 丰田 Connected North America 首席执行官兼总裁 Steve Basra 表示：“Google Cloud 行业领先的基于人工智能的语音服务的集成已经在为丰田和雷克萨斯的客户提供高质量的语音服务。通过这些最新的生成人工智能公告， 我们很高兴扩大我们的合作伙伴关系，并探索基础语言模型如何进一步推进我们的愿景，为驾驶员带来最好和最具创新性的车内体验。”

“Thanks to their AI capabilities, Google Cloud has been a key strategic partner for Mayo Clinic in advancing the diagnosis and treatment of disease to improve the health of people and communities,” said Christopher Ross, CIO, Mayo Clinic. “Generative AI-powered search has the potential to help clinicians find, understand, and interpret information while keeping the data private and secure. We look forward to our continued co-innovation to deliver solutions for clinicians and employees to help reduce cognitive information overload and improve operational efficiencies.”

梅奥诊所首席信息官 Christopher Ross 表示：“凭借其 AI 功能，谷歌云已成为梅奥诊所在推进疾病诊断和治疗以改善人们和社区健康方面的重要战略合作伙伴。” “生成式人工智能搜索有可能帮助临床医生查找、理解和解释信息，同时保持数据的私密性和安全性。 我们期待着我们继续共同创新，为临床医生和员工提供解决方案，以帮助减少认知信息过载并提高运营效率。”

“Google Cloud’s leading AI technology enables STARZ customers to discover more relevant content, increasing engagement with, and the likelihood of completing the content served to them,” said Robin Chacko, EVP Direct-to-Consumer, STARZ. “We’re excited about how generative AI-powered search will help users find the most relevant content even easier and faster."

STARZ 直接面向消费者的执行副总裁 Robin Chacko 表示：“谷歌云领先的 AI 技术使 STARZ 客户能够发现更多相关内容，增加参与度，并更有可能完成向他们提供的内容。” “我们对生成式人工智能搜索将如何帮助用户更轻松、更快速地找到最相关的内容感到兴奋。”

Starting today, trusted testers are accessing Generative AI App Builder. If you are interested in updates on our early access opportunities, please join our technical community, [Google Cloud Innovators](https://cloud.google.com/ai).

从今天开始，受信任的测试人员可以访问 Generative AI App Builder。 如果您对我们的早期访问机会的更新感兴趣，请加入我们的技术社区 Google Cloud Innovators。

## Turning generative AI building blocks into enterprise value
## 将生成式 AI 构建模块转化为企业价值

Let’s look at a few examples of how organizations are already looking to unlock the power of generative AI with Google Cloud:

让我们看一些组织已经在寻求如何使用 Google Cloud 释放生成式 AI 的力量的例子：

* Automated content generation. Generative AI can facilitate brainstorming, perform copywriting, and generate media assets — meaning that emails, marketing messages, and creative assets can be prototyped in seconds, and ready for review within minutes or hours, not weeks or months. Marketing and creative teams across organizations are looking to augment current workflows with this technology to instantly bring more choices, more flavors, and greater ingenuity to campaigns, programs, ads, and more.
* 自动化内容生成。 生成式 AI 可以促进头脑风暴、执行文案和生成媒体资产——这意味着电子邮件、营销信息和创意资产可以在几秒钟内制作原型，并在几分钟或几小时内准备好进行审查，而不是几周或几个月。 跨组织的营销和创意团队正在寻求利用这项技术来增强当前的工作流程，以立即为活动、计划、广告等带来更多选择、更多口味和更大的独创性。
* AI experiences and assistants for virtually any task. Because generative AI lets businesses and governments turn large, complex volumes of data into summaries, interactive multimedia experiences, and human-like conversations, we see many customers interested in leveraging this technology for not only customer-facing experiences, like brand or product Q&As, but also more complex data science scenarios. For example, gen apps, like digital assistants, can help data analysts and business users up-level their skills, by generating SQL queries, enabling exploration of data through natural language queries, and more. 
* 几乎任何任务的 AI 体验和助手。由于生成人工智能让企业和政府能够将大量复杂的数据转化为摘要、交互式多媒体体验和类似人类的对话，我们看到许多客户对利用这项技术感兴趣，不仅是面向客户的体验，如品牌或产品问答， 还有更复杂的数据科学场景。 例如，生成应用程序，如数字助理，可以通过生成 SQL 查询、通过自然语言查询探索数据等，帮助数据分析师和业务用户提升技能。
* Searching and understanding large, internal datasets that span many sources. Many of our banking customers analyze various internal and external data sources to get a comprehensive view of the market. They’re exploring how to use this technology to ensure that when employees search for information across these sources, they get relevant results, accurate summaries of large documents, tools to refine queries’ sources, and citations and attributions so employees can trust outputs and dig deeper as needed. 
* 搜索和理解跨多个来源的大型内部数据集。 我们的许多银行客户分析各种内部和外部数据源以全面了解市场。 他们正在探索如何使用这项技术来确保当员工在这些来源中搜索信息时，他们会得到相关的结果、大型文档的准确摘要、优化查询来源的工具以及引用和归属，以便员工可以信任输出并挖掘 根据需要更深。
* Providing a foundation to jumpstart the wave of gen app startups. Vertex AI levels the playing field for development with generative AI. By providing API access to foundation models while reducing the massive and prohibitive data requirements these technologies usually entail, Vertex AI will empower builders and innovators of all kinds, from data scientists to self-taught developers, to create the next generation of startups. 
* 为启动新一代应用程序创业浪潮奠定基础。 Vertex AI 为生成式 AI 的开发创造了公平的竞争环境。 通过提供对基础模型的 API 访问，同时减少这些技术通常需要的大量和禁止的数据要求，Vertex AI 将使从数据科学家到自学成才的开发人员的各种构建者和创新者能够创建下一代初创公司。

## Protecting data and shaping conversation flows with Responsible AI
## 使用 Responsible AI 保护数据和塑造对话流

When Google Cloud brings new AI advances to our products, our commitment is two-fold: to deliver transformative capabilities, while ensuring our technologies include proper protections for our organizations, their users, and society. To this end, our AI Principles, established in 2017, form a living constitution that guides our approach to building advanced technologies, conducting research, and drafting our product development policies. For more information about how we put our AI Principles into practice, read our most comprehensive report to date.

当 Google Cloud 为我们的产品带来新的 AI 进步时，我们的承诺有两个：提供变革性功能，同时确保我们的技术包括对我们的组织、他们的用户和社会的适当保护。 为此，我们于 2017 年制定的 AI 原则构成了一部活生生的宪法，指导我们构建先进技术、开展研究和起草产品开发政策的方法。 有关我们如何将 AI 原则付诸实践的更多信息，请阅读我们迄今为止最全面的报告。

Our new announcements are no exception, providing:

我们的新公告也不例外，提供：

* Transparency and explainability: Both Vertex AI and Generative AI App Builder include tools to inspect, understand, and modify model behavior.
* 透明度和可解释性：Vertex AI 和 Generative AI App Builder 都包含用于检查、理解和修改模型行为的工具。
* Data privacy and sovereignty: Whether a company is training a model in Vertex AI or building a customer service experience on Generative AI App Builder, private data is kept private, and not used in the broader foundation model training corpus. Organizations always maintain control over where data is stored and how or if it is used, letting them safely pursue data-rich use cases while complying with various regulations and data sovereignty laws.
* 数据隐私和主权：无论公司是在 Vertex AI 中训练模型，还是在 Generative AI App Builder 上构建客户服务体验，私有数据都是保密的，不会用于更广泛的基础模型训练语料库。 组织始终保持对数据存储位置以及使用方式或是否使用的控制，让他们在遵守各种法规和数据主权法律的同时安全地追求数据丰富的用例。
* Factuality and freshness: Generative AI App Builder uses information retrieval algorithms to provide the right sourcing and attribution while serving the most relevant information.
* 真实性和新鲜度：Generative AI App Builder 使用信息检索算法来提供正确的来源和归因，同时提供最相关的信息。
* Probabilistic models with deterministic workflow controls: Organizations want to blend the interactive and probabilistic nature of generative AI with results that are controlled, deterministic, and reliable.
* 具有确定性工作流控制的概率模型：组织希望将生成 AI 的交互性和概率性与受控、确定性和可靠的结果相结合。
* Choices to fit different requirements: To meet organizations' differing needs, our platforms are designed to be flexible, including data and model lineage capabilities, integrated security and identity management services, support for third-party models, choice and transparency on models and costs, integrated billing and entitlement support, and support across many languages.
* 满足不同需求的选择：为了满足组织的不同需求，我们的平台设计灵活，包括数据和模型沿袭功能、集成安全和身份管理服务、对第三方模型的支持、模型和成本的选择和透明度， 集成的计费和授权支持，以及跨多种语言的支持。

“Google Cloud has been a strategic partner for Deutsche Bank, working with us to improve operational efficiency and reshape how we design and deliver products for our customers,” said Gil Perez, Chief Innovation Officer, Deutsche Bank. “We appreciate their approach to Responsible AI and look forward to co-innovating with their advancements in generative AI, building on our success to date in enhancing developer productivity, boosting innovation, and increasing employee retention.”

“谷歌云一直是德意志银行的战略合作伙伴，与我们合作提高运营效率并重塑我们为客户设计和交付产品的方式，”德意志银行首席创新官 Gil Perez 说。 “我们赞赏他们对 Responsible AI 的方法，并期待与他们在生成 AI 方面的进步共同创新，在我们迄今为止在提高开发人员生产力、促进创新和提高员工保留率方面取得的成功的基础上再接再厉。”

Additionally, as part of our commitment to an open approach to AI development, we’re also announcing [new AI partnerships and programs](https://cloud.google.com/blog/products/ai-machine-learning/building-an-open-generative-ai-partner-ecosystem) that make it easier for startups, developers, and enterprises to accelerate their AI projects. 

此外，作为我们对 AI 开发开放方法的承诺的一部分，我们还宣布了新的 AI 合作伙伴关系和计划，使初创公司、开发人员和企业更容易加速他们的 AI 项目。

Visit our [AI on Google Cloud](https://cloud.google.com/ai) webpage or join us at [Google Data Cloud & AI Summit](https://cloudonair.withgoogle.com/events/summit-data-cloud-2023?_gl=1*82x527*_ga*NzM2NDc2ODAzLjE2Nzg4MDQ1OTI.*_ga_WH2QY8WWF5*MTY3ODgwNDU5MS4xLjEuMTY3ODgwNDU5MS4wLjAuMA..&_ga=2.254159438.-736476803.1678804592), live online March 29, to learn more about our new announcements.

请访问我们的 AI on Google Cloud 网页或加入我们的 Google 数据云和 AI 峰会（3 月 29 日在线直播），以详细了解我们的新公告。