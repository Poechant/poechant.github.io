---
layout: post
title:  OpenAI GPT-3 的快速指南
date:   2022-05-01 23:54:49 +0800
categories: ai
tags: [AI, 人工智能, OpenAI, Quickstart, GPT-3]
description: 
excerpt: 
katex: True
location: 杭州
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

OpenAI has trained cutting-edge language models that are very good at understanding and generating text. Our API provides access to these models and can be used to solve virtually any task that involves processing language.

In this quickstart tutorial, you’ll build a simple sample application. Along the way, you’ll learn key concepts and techniques that are fundamental to using the API for any task, including:

* 内容生成
* 总结概括
* Classification，categorization 和情感分析
* Data extraction
* 翻译
* Many more!

## 0、Introduction

The completions endpoint is the core of our API and provides a simple interface that’s extremely flexible and powerful. You input some text as a prompt, and the API will return a text completion that attempts to match whatever instructions or context you gave it.

> Prompt: Write a tagline for an ice cream shop.
> Completion: We serve up smiles with every scoop!

你可以把这个认为是一个高级版本的自动补全 —— 模型处理你的文字提示（prompt），并试着预测接下来最可能的内容是什么。

## 1、从一个指令（Instruction）开始123

想象你要创造一个宠物名字生成器。Coming up with names from scratch is hard!

首先，你需要一个能清楚表述你要什么的提示（prompt）。Let’s start with an instruction. Submit this prompt to generate your first completion.

> Suggest one name for a horse.
> Midnight

Not bad！现在试试更具体一点的指令（Instruction）

> Suggest one name for a black horse.
> Midnight

As you can see, adding a simple adjective to our prompt changes the resulting completion. Designing your prompt is essentially how you “program” the model.

## 2、Add some examples

Crafting good instructions is important for getting good results, but sometimes they aren’t enough. Let’s try making your instruction even more complex.

> Suggest three names for a horse that is a superhero.

结果：

> 1. Super Stallion
> 2. Captain Colt
> 3. Mighty Mustang

This completion isn't quite what we want. These names are pretty generic, and it seems like the model didn't pick up on the horse part of our instruction. Let’s see if we can get it to come up with some more relevant suggestions.

In many cases, it’s helpful to both show and tell the model what you want. Adding examples to your prompt can help communicate patterns or nuances. Try submitting this prompt which includes a couple examples.

> Suggest three names for an animal that is a superhero.
>
> Animal: Cat
> Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
> Animal: Dog
> Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
> Animal: Horse
> Names:

结果：

> Super Stallion, Mighty Mare, The Magnificent Equine

Nice! Adding examples of the output we’d expect for a given input helped the model provide the types of names we were looking for.

## 3、Adjust your settings
Prompt design isn’t the only tool you have at your disposal. You can also control completions by adjusting your settings. One of the most important settings is called temperature.

You may have noticed that if you submitted the same prompt multiple times in the examples above, the model would always return identical or very similar completions. This is because your temperature was set to 0.

Try re-submitting the same prompt a few times with temperature set to 1.

> Suggest three names for an animal that is a superhero.
> 
> Animal: Cat
> Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
> Animal: Dog
> Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
> Animal: Horse
> Names:

结果：

> Super Stallion, Mighty Mare, The Magnificent Equine

See what happened? When temperature is above 0, submitting the same prompt results in different completions each time.

Remember that the model predicts which text is most likely to follow the text preceding it. Temperature is a value between 0 and 1 that essentially lets you control how confident the model should be when making these predictions. Lowering temperature means it will take fewer risks, and completions will be more accurate and deterministic. Increasing temperature will result in more diverse completions.

For your pet name generator, you probably want to be able to generate a lot of name ideas. A moderate temperature of 0.6 should work well.

## 4、Build your application (`PYTHON(FLASK)`)

Now that you’ve found a good prompt and settings, you’re ready to build your pet name generator! We’ve written some code to get you started — follow the instructions below to download the code and run the app.

### Setup

If you don’t have Node.js installed, install it from here. Then download the code by cloning this repository.

```shell
git clone https://github.com/openai/openai-quickstart-python.git
```

If you prefer not to use git, you can alternatively download the code using this zip file.

### Add your API key

Navigate into the project directory and make a copy of the example environment variables file.

```shell
cd openai-quickstart-python
cp .env.example .env
```

Copy your secret API key and set it as the OPENAI_API_KEY in your newly created .env file. If you haven't created a secret key yet, you can do so below.

|SECRET KEY | CREATED     | LAST USED|
|-----------|-------------|----------|
|sk-...riXH |2022年12月10日|	Never	 |

Important note: When using Javascript, all API calls should be made on the server-side only, since making calls in client-side browser code will expose your API key. See here for more details.

### Run the app

Run the following commands in the project directory to install the dependencies and run the app.

```shell
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
flask run
```

### Understand the code

Understand the code
Open up app.py in the openai-quickstart-python folder. At the bottom, you’ll see the function that generates the prompt that we were using above. Since users will be entering the type of animal their pet is, it dynamically swaps out the part of the prompt that specifies the animal.

```python
def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(animal.capitalize())
```

On line 14 in app.py, you’ll see the code that sends the actual API request. As mentioned above, it uses the completions endpoint with a temperature of 0.6.

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=generate_prompt(animal),
  temperature=0.6
)
```

And that’s it! You should now have a full understanding of how your (superhero) pet name generator uses the OpenAI API!

## 5、Closing
These concepts and techniques will go a long way in helping you build your own application. That said, this simple example demonstrates just a sliver of what’s possible! The completions endpoint is flexible enough to solve virtually any language processing task, including content generation, summarization, semantic search, topic tagging, sentiment analysis, and so much more.

One limitation to keep in mind is that, for most models, a single API request can only process up to 2,048 tokens (roughly 1,500 words) between your prompt and completion.

> Models and pricing
>
> We offer a spectrum of models with different capabilities and price points. In this tutorial, we used text-davinci-003, our most capable natural language model. We recommend using this model while experimenting since it will yield the best results. Once you’ve got things working, you can see if the other models can produce the same results with lower latency and costs.
> The total number of tokens processed in a single request (both prompt and completion) can’t exceed the model's maximum context length. For most models, this is 2,048 tokens or about 1,500 words. As a rough rule of thumb, 1 token is approximately 4 characters or 0.75 words for English text.
> Pricing is pay-as-you-go per 1,000 tokens, with $18 in free credit that can be used during your first 3 months. Learn more.

For more advanced tasks, you might find yourself wishing you could provide more examples or context than you can fit in a single prompt. The fine-tuning API is a great option for more advanced tasks like this. Fine-tuning allows you to provide hundreds or even thousands of examples to customize a model for your specific use case.

## 6、Next steps

To get inspired and learn more about designing prompts for different tasks:

* Read our completion guide.
* Explore our library of example prompts.
* Start experimenting in the Playground.
* Keep our usage policies in mind as you start building.