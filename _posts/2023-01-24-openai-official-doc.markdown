---
layout: post
title: 麦克船长的 OpenAI 模型 API 官方文档入门解读
date:   2023-01-24 06:24:58 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, 神经网络, LLM, 大型语言模型, 语言模型, 大模型, GPT-3, OpenAI, davinci, curie, ada, babbage, fine-tune, fine-tuning, 精调, ChatGPT, completion, prompting, 训练, 数据集]
description: 在 OpenAI 的 API 官方首页的大标题写着的是「用 OpenAI 的强力模型构建下一代应用（Build next-gen apps with OpenAI’s powerful models）」，并直接点出了自己最拿得出手的三个商业化产品 GPT-3、Codex、DALL·E。本文麦克船长重点为你解读基于 GPT-3 的几款模型的入门使用。
excerpt: 在 OpenAI 的 API 官方首页的大标题写着的是「用 OpenAI 的强力模型构建下一代应用（Build next-gen apps with OpenAI’s powerful models）」，并直接点出了自己最拿得出手的三个商业化产品 GPT-3、Codex、DALL·E。本文麦克船长重点为你解读基于 GPT-3 的几款模型的入门使用。
katex: True
location: 香港
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

## 一、先大概看看 OpenAI 目前最拿得出手的三个商业化产品用起来啥样

在 OpenAI 的 API 官方首页的大标题写着的是「用 OpenAI 的强力模型构建下一代应用（Build next-gen apps with OpenAI’s powerful models）」，并直接点出了自己最拿得出手的三个商业化产品 GPT-3、Codex、DALL·E。

并提到自己的 API 调用非常简单：

```python
import openai

openai.Completion.create(
  engine="davinci",
  prompt="Make a list of astronomical observatories:"
)
```

引入 OpenAI 的 python 依赖库后，调用时指定用哪个引擎、提示词是什么即可。轻松一些，我们先来看一些示例吧。

### 1、执行各种自然语言任务的 GPT-3

#### 示例 1：Copywriting

人类输入：

```
Create promo copy for the FamilyTime mobile application. It allows unlimited uploading, special filters and makes it easy to create albums of photos and videos. It runs on iOS and Android:
```

GPT-3 输出：

```
FamilyTime is the perfect app to share and create memories with your family. FamilyTime is a free, unlimited photo and video editor that allows you to create albums of your memories. It has a special filter for the perfect summer look and is so simple to use.
```

#### 示例 2：Summarization

人类输入：

```
Summarize this email into a single sentence:

Dear Olivia,

The solar energy conference went great. New Horizon Manufacturing wants to meet with us to talk about our photovoltaic window system we’re about to launch.
I think it would be great to talk this Tuesday.

Best,
Allison

```

GPT-3 输出：

```
Allison is excited to meet with New Horizon Manufacturing to discuss their photovoltaic window system.
```

#### 示例 3：Parsing Unstructured Text

人类输入：

```
Answer the following questions about this customer email:

I’m wondering if you could provide me with information about your cloud services for companies. I’m CTO for Moon Door, a movie production company and want to know if it can help us manage our different projects that are in production.

Questions:
1. What industry is the company
2. What is the name of the company
3. What are they interested in
4. What is the customer’s position

Answers:
```

GPT-3 输出：

```
1. The company is in the movie production industry.
2. The company is Moon Door.
3. The customer is interested in cloud services for companies.
4. The customer is the CTO.
```

#### 示例 4：Classification

人类输入：

```
Choose a genre category for each book 1. The Hunger Games, 2. The Kite Runner 3. A Wrinkle in Time (“fiction”, “young adult”, “science fiction”, “fantasy”, “other”) and make a list of the book and its genre:
```

GPT-3 输出：

```
1. The Hunger Games: young adult, fiction
2. The Kite Runner: fiction, young adult
3. A Wrinkle in Time: science fiction, fantasy, other
```

#### 示例 5：Translation

人类输入：

```
Translate this into French: Where can I find a bookstore?
```

GPT-3 输出：

```
Où puis-je trouver un magasin de livres?
```

### 2、将自然语言翻译成代码的 Codex

#### 示例 6：用自然语言写 SQL

人类输入：

```
Table customers, columns = [CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId, TotalSpend]

Create a MySQL query for all customers in Texas who have spent over five thousand dollars.
```

Codex 输出：

```
query = "SELECT * FROM customers WHERE State = 'TX' AND TotalSpend > 5000"
```

#### 示例 7：用自然语言调用一个 API

人类输入：

```
Python 3
Get the current value of a Bitcoin in US dollars using the bitcoincharts api
```

Codex 输出：

```python
import requests
import json

def get_bitcoin_price():
    url = 'http://api.bitcoincharts.com/v1/weighted_prices.json'
    response = requests.get(url)
    data = json.loads(response.text)
    return data['USD']['7d']

if __name__ == '__main__':
    print(get_bitcoin_price())
```

#### 示例 8：用自然语言续写代码

人类输入：

```
Extract all the html from the string and replace the tags with ''
```

Codex 输出：

```
def extract_html(text):
    return re.sub('&lt;[^&lt;]+?&gt;', '', text)
```

### 3、创建和编辑原始图像的 DALL·E

超过 300 万人已经在使用 DALL·E 来放大他们的创造力并加快他们的工作流程，每天生成超过 400 万张图像。

## 二、OpenAI API 总览性介绍

### 1、一些关键概念

关于 prompt 和 completion：OpenAI 提到一个理念：「设计提示语，就相当于在用一些指令和少量例子给模型编程」。另外 OpenAI 还强调了在目标任务上的区别，就是 OpenAI 的 NLP 模型与其他 NLP 模型很大的一个区别是，它不是设计用来解决单一类型任务的，而是可以解决几乎各种类型的 NLP 任务，包括但不限于文本生成（content generation）、代码生成（code generation）、总结（summarization）、扩写（expansion）、对话（conversation）、创意写作（creative wrting）、风格转换（style transfer）等。

关于 token：我们理解和处理文本，是把文本先打碎成 token。以英文文本为例，token 可以是单词，也可以词根（一些字母组合），比如单词「hamburger」可能会被打碎成「ham」、「bur」、「ger」这几个 tokens。再比如「pear」这个单词，可能就会单独作为一个 token 不再打碎了。还有些 token 可能会以「空格」开头，比如「 hello」、「 bye」。一个大概的经验是，通常英文文本里 1 token 有 4 个字母或者 0.75 个单词。使用时的一个限制是，最好你的提示（prompt）或生成内容，不要超过 2048 个 tokens，大概相当于 1500 个单词。

关于 model：目前 OpenAI 有基于 GPT-3.5 的基础模型 Turbo 和这些基于 GPT-3 的基础模型 Davinci、Curie、Babbage、Ada 开放 API，另外 Codex 系列是 GPT-3 的后代，是用「自然语言 + 代码」训练的。

### 2、模型

* `text-davinci-003`：最大请求 4000 tokens，训练数据 up to 2021 年 6 月，能做几乎所有 NLP 任务。
* `text-curie-001`：最大请求 2048 tokens，训练数据 up to 2019 年 10 月，比 davinci 要弱一点，但是速度更快、更便宜。
* `text-babbage-001`：最大请求和训练数据和 `text-curie-001` 一样，一些比较直接的任务（straightforward tasks），比 `text-curie-001` 更快、更便宜。
* `text-ada-001`：最大请求和训练数据和 `text-curie-001` 一样，一些非常简单的任务，这些模型里最快、最便宜的。

这四个模型根据输入的 token 数量做的如下定价：

* 基础模型使用 0.000**4** USD/1K tokens，Ada
* 基础模型使用 0.000**5** USD/1K tokens，Babbage
* 基础模型使用 0.00**20** USD/1K tokens，Curie
* 基础模型使用 0.0**200** USD/1K tokens，Davinci
* 呼出模型使用 0.00**20** USD/1K tokens，Turbo

从定价上看，Ada 和 Babbage 基本没有差多少。另外命名上，可以看出 OpenAI 有意地给他们取了 ABCD 开头的名字。另外你也可以 finetune 你自己的模型，对于 fine-tuned models 如下收费：

* Finetune 训练费 0.000**4** USD/1K tokens，使用费 0.00**16** USD/1K tokens，Ada
* Finetune 训练费 0.000**6** USD/1K tokens，使用费 0.00**24** USD/1K tokens，Babbage
* Finetune 训练费 0.00**30** USD/1K tokens，使用费 0.0**120** USD/1K tokens，Curie
* Finetune 训练费 0.0**300** USD/1K tokens，使用费 0.**1200** USD/1K tokens，Davinci
* 暂未提供 Turbo 的 finetune。

在 OpenAI 的 PlayGround 你可以试试：[https://platform.openai.com/playground/p/default-chat](https://platform.openai.com/playground/p/default-chat) 。

![image](/img/src/2023/2023-01-24-openai-api.jpg)

## 三、主要 API 介绍及代码示例

安装 OpenAI 的 python 库，参考 `https://anaconda.org/conda-forge/openai`：

```shell
mikecaptain@local $ conda install -c conda-forge openai
```

在 `https://platform.openai.com/account/api-keys` 创建自己的 API。完成这两步后就可以编写代码尝试一下：

```python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.Model.list())
```

会打印出 OpenAI 的各个 models 的一些信息、权限等等。

### 1、Text Completion 任务

下面这个例子会简单调用一下 completion，并打印出结果，用了一句需要你自己编写的 prompt：

```python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
text_prompt = "In a shocking turn of events, scientists have discovered that "
completion = openai.Completion.create(
    model="text-davinci-002",
    prompt=text_prompt,
    max_tokens=100,
    n=1,
    stop=None,
    temperature=0.5,
)

generated_text = completion.choices[0].text
print(generated_text)
```

这里用到了最重要的 `openai.Completion`，其 `create` 函数的参数解释如下：

* `model`：之前 OpenAI 把它叫「engine」，后来给 deprecated 了，现在都是用 `model`，所有的可用 models 可以通过 `open.Model.list()` 来查看。
* `prompt`：`string` 类型，就是输入数据。
* `suffix`：`string` 类型，生成文本的结束符。
* `max_tokens`：`integer` 类型，生成文本的最大 tokens 数。
* `n`：`integer` 类型，表示你要产生几个不同的输出结果。比如设置 3 就会得到 3 个不同的结果，以便您可以从中选择最合适的一个。
* `stop`：`string` 类型，用于指定模型何时应该停止生成文本。当模型在生成的文本中遇到 stop 字符串时，它将停止生成文本。ChatGPT 推出后迭代过一版增加了「stop generating」就是用的这个参数。
* `temperature`：`number` 类型，这是 NLP 模型里常见的一个超参数。这个参数，来自于统计热力学的概念，温度越高表示系统的熵越高、混乱度越高、随机性越强，这里的 temperature 也是值越高输出结果的随机性也越高。这样如果 temperature 设置得很低，生成的结果可能更正确，但没有多少创造性和随机性。

### 2、Text Edit 任务

Completion 类任务，通俗点理解的话，完形填空、句子补齐、写作文、翻译 …… 都算 Completion，就是无中生有。而对于已经有的内容，做修改，就是 OpenAI 的 API 里的「Edit」类的任务了。

```python
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Edit.create(
  model="text-davinci-edit-001",
  input="The qick brown fox jumps over the layz dog.",
  instruction="Fix the spelling mistakes"
)
```

调用 `openai.Edit.create`，用 `text-davinci-edit-001` 模型，输入一句有拼写错误的英文「The qick brown fox jumps over the layz dog.」，并提供一句指令 instruction「Fix the spelling mistakes」。

* `instruction`：要告诉模型如何修改，**其实这句话就是新时代的「programming」了**。
* `temperature`：默认是 0，对于纠正拼写类的任务，我们用默认 0 就可以了，不需要什么创造性和随机性。

### 3、Image Create 任务（Beta）

截止 2023 年年初 1 月份，这个 API 还是 beta，我们看个例子：

```python
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Image.create(
  prompt="A cute baby sea otter",
  n=2,
  size="1024x1024"
)
```

这也是一个 OpenAI 官网的例子。大家可能看到这里，船长没有指定 model，但是可以想到一定用的是 DALL·E，因为它没有像 GPT-3 一样提供很多版本的选择，所以就不需要传参数了。这个程序就是生成一个 1024x1024 的图片。

* `prompt`：就是输入的提示语，返回的数据里，会告诉你生成的图片的 URL. 
* `n`：是图片结果数量，最多 10，默认 1. 

### 4、Image Edit 任务

给定一个图片，OpenAI 也可以来修改指定区域：

```python
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Image.create_edit(
  image=open("otter.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="A cute baby sea otter wearing a beret",
  n=2,
  size="1024x1024"
)
```

* `image`：这里对输入图片有要求，必须是正方形的！另外不能超过 4MB，还得是 PNG。
* `mask`：还可以提供掩码图片（叫什么比较合适，掩图？哈哈）。如果不提供的话，`image` 里就必须有透明的部分（必须全透明，即 alpha = 0），那个透明部分就是被用来 Edit 的。如果有 `mask` 则透明部分用来做「掩图」来改 `image`。
* 同样地，结果图片的 URL 会返回给你。

### 5、审查（Moderation）

Moderation 用来审查内容是否符合 OpenAI 的内容政策，快速使用的方式如下：

```python
response = openai.Moderation.create(
    input="Sample text goes here"
)
output = response["results"][0]
```

API 官网给出我们如下的返回结果示例：

```python
{
  "id": "modr-XXXXX",
  "model": "text-moderation-001",
  "results": [
    {
      "categories": {
        "hate": false,
        "hate/threatening": false,
        "self-harm": false,
        "sexual": false,
        "sexual/minors": false,
        "violence": false,
        "violence/graphic": false
      },
      "category_scores": {
        "hate": 0.18805529177188873,
        "hate/threatening": 0.0001250059431185946,
        "self-harm": 0.0003706029092427343,
        "sexual": 0.0008735615410842001,
        "sexual/minors": 0.0007470346172340214,
        "violence": 0.0041268812492489815,
        "violence/graphic": 0.00023186142789199948
      },
      "flagged": false
    }
  ]
}
```

输入参数很简单，关键看返回的输出结果。OpenAI 对于包含哪类不适内容，做了比较详尽的分类，比如对于色情内容，也分成了未成年色情和易引起性兴奋的内容。

* `hate`：是否包含基于种族、性别、民族、宗教、国籍、性取向、残疾状况或种姓表达、煽动或促进仇恨的内容，如果没有则是 `false`，否则为 `true`。
* `hate/threatening`：是否包含仇恨内容还包括对目标群体的暴力或严重伤害，没有则 `false`，包含则值为 `true`。
* `self-harm`：是否包含提倡、鼓励或描述自残行为（例如自杀、割伤和饮食失调）的内容，没有则 `false`，否则 `true`。
* `sexual`：是否包含意在引起性兴奋的内容，例如对性活动的描述，或宣传性服务（不包括性教育和健康）的内容，没有则 `false`，否则 `true`。
* `sexual/minors`：是否包含包含 18 岁以下个人的色情内容，没有则 `false`，否则 `true`。
* `violence`：是否包含宣扬或美化暴力或颂扬他人的痛苦或屈辱的内容，没有为 `false`，否则 `true`。
* `violence/graphic`：是否包含以极端的画面细节描绘死亡、暴力或严重身体伤害的暴力内容，没有 `false`，否则 `true`。

显然，对于使用 OpenAI 生成内容的场景下如果需要用到 Moderation，则是免费调用的。如果你不是对 OpenAI 的输入 & 生成场景，而是自己的其他内容想白嫖 Moderation API 是不可能的。但是我们也注意到，这里其实没有整治敏感的分类，因为 OpenAI 没有考虑具体的使用者所处的政体或政治环境，而且这些尺度是比较容易变化的，并且有一些可能并不是普适性的理念，因此某些国家的使用者要额外配套自己的内容审查能力。

## 四、微调（Fine-tuning）

**Few-shot learning 是什么？**：GPT-3 用了互联网上的海量文本数据训练，所以当你给少量示例（a promopt with just a few examples）时，GPT-3 会从「直觉上」知道你大概是想要解决什么任务，然后给出一些大概齐的反馈内容作为 completion，这通常就被叫做「few-shot learning」或者「few-shot prompting」。

而如果你提供一些针对目标任务的训练数据，很可能可以实现没有 examples 也可以执行任务，也就是使用时连「few-shot learning」都免了。OpenAI 也提供了让用户自己 fine-tune 模型的接口，自己 fine-tune 的好处是：

* **高质量**：这是显然的，比「设计提示（prompt design）」得到的结果质量更高。
* **相当于批量 prompt**：可以比 prompt 给模型更多的 examples，比如用一个文件，里面包含大量用于 fine-tuning 的输入数据。
* **更省**：可以更省 tokens，也就更省钱。
* **更快**：更低的延迟的请求响应。

**步骤和价格**方面，Fine-tune 一共三步：上传用于 fine-tune 的数据、用数据 fine-tune 模型、使用属于你自己的 fine-tune 过的模型。从定价上我们看到 Fine-tune 后的模型使用费用基本翻了 4~6 倍，可以说相比基本模型的使用，是非常贵了。

另外 OpenAI 也支持你对一个 fine-tune 过的模型继续 fine-tune，而不用从头开始。目前 davinci、curie、babbage、ada 都支持 fine-tuning。训练数据的格式也很简单，就是一组 prompt-completion 的 JSONL 文件，just like this：

```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```

Fine-tune 的 example 与 few-shot learning 的最大区别：

* few-shot learning 要给出详尽的 instruction 来描述任务
* few-shot learning 的一个 prompt 是在使用时给出的，所以一个 prompt 大概率会带多个 examples（相对详细）；而 fine-tune 的 example 都是一些简单直接的 prompt 以及直接对应的 completion。

OpenAI 建议 fine-tune 的 examples 数量至少几百（a couple hundred）。另外 fine-tune 也符合 scaling law，基本上 fine-tune 的数据集成倍上翻的话，效果是线性增长的。

### 1、创建一个 fine-tune 模型

CLI 下运行如下命令，其中 `<TRAIN_FILE_ID_OR_PATH>` 是你的训练数据文件，`<BASE_MODEL>` 是你要用的模型，具体的参数可以用 `ada`、`babbage`、`curie` 和 `davinci`。

```shell
mikecaptain@local $ openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>
```

这句命令让 OpenAI 不仅基于 base model 创建了一个模型，而且开始运行训练任务。训练任务可能会花费几分钟、几小时甚至根据，取决于你的训练集和模型选择。训练任务可能会被 OpenAI 排队，不一定马上开始运行。如果过程中被打断了，可以如下继续：

```shell
mikecaptain@local $ openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>
```

保存一个 fine-tune job 的命令如下：

```shell
mikecaptain@local $ openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>
```

取消一个 fine-tune job 的命令如下：

```shell
mikecaptain@local $ openai api fine_tunes.cancel -i <YOUR_FINE_TUNE_JOB_ID>
```

### 2、使用 fine-tuned 模型

```python
import openai
openai.Completion.create(model=FINE_TUNED_MODEL, prompt=YOUR_PROMPT)
```

### 3、删掉一个 fine-tuned 模型

```python
import openai
openai.Model.delete(FINE_TUNED_MODEL)
```

### 4、一个 fine-tuned 模型之上继续 fine-tune

如果你微调了一个模型，现在又有为的训练数据想要合并进来，可以基于已 fine-tuned 模型继续微调，无需从头再全部训练一遍。唯一要做的，就是在创建新的 fine-tune job 时传入已 fine-tune 的模型名称，替代`<BASE_MODEL>`（例如 `-m curie:ft-<org>-<date>`），不必更改其他训练参数。

有一个要注意的，如果新增的训练数据比以前的训练数据规模小得多，那最好把 `learning_rate_multiplier` 减少 2 到 4 倍，否则很可能跳过了最优解。

## 参考

* https://openai.com/api/
* https://developer.aliyun.com/article/933516