---
layout: post
title:  让你和你的朋友们，在微信上跟 ChatGPT 聊聊天 / 2022#1
date:   2022-12-11 23:59:57 +0800
categories: 自然语言处理
tags: [人工智能, AI, ChatGPT, OpenAI, 微信]
description: 最近 OpenAI 的 ChatGPT 非常地出圈，ChatGPT 是一个由 OpenAI 训练的大型语言模型，被设计用来回答用户的问题并提供信息。官方的 Slogan 是「Optimizing Language Models for Dialogue」，所以非常适合做到 IM 里聊天。那么我在想如果用一个微信号，背后是 ChatGPT，是不是很有趣？正当我准备利用 WeChaty 开发一个服务端程序来连接 ChatGPT 时，发现目前 Github 上已经有人做了，刚好可以省去很多工程的工作。
excerpt: 最近 OpenAI 的 ChatGPT 非常地出圈，ChatGPT 是一个由 OpenAI 训练的大型语言模型，被设计用来回答用户的问题并提供信息。官方的 Slogan 是「Optimizing Language Models for Dialogue」，所以非常适合做到 IM 里聊天。那么我在想如果用一个微信号，背后是 ChatGPT，是不是很有趣？正当我准备利用 WeChaty 开发一个服务端程序来连接 ChatGPT 时，发现目前 Github 上已经有人做了，刚好可以省去很多工程的工作 ……
---
![image](/img/src/2022-12-11-wechat-chatgpt-3.png)

### 写在前面
最近 OpenAI 的 ChatGPT 非常地出圈，ChatGPT 是一个由 OpenAI 训练的大型语言模型，被设计用来回答用户的问题并提供信息。官方的 Slogan 是 **「Optimizing Language Models for Dialogue」**，所以非常适合做到 IM 里聊天。那么我在想如果用一个微信号，背后是 ChatGPT，是不是很有趣？正当我准备利用 WeChaty 开发一个服务端程序来连接 ChatGPT 时，发现目前 Github 上已经有人做了，刚好可以省去很多工程的工作。

### Step by step

本实践依赖：CLI、Docker、npm、Github、fuergaosi233/wechat-chatgpt、git、YAML、Chrome 的使用。以下将简洁地 Step by step 列出步骤。

第一步，你要现有一个 OpenAI 的账号，注意注册时手机号不能是中国大陆或香港的，IP 地址和 GPS 也不能暴露你是中国大陆或者香港的。

第二步，准备一台服务器（否则个人电脑要一直处于开机运行状态），由于后面将用到 Session Token 来登录，因此 IP 地址是香港也没关系，于是我是在我的香港服务器上部署 wechat-chatgpt

第三步，在服务器上安装 Docker，不赘述。

第四步，从 Github 上拉取项目项目到服务器上。

第五步，任何设备上登录 ChatGPT，用 Chrome 的 Inspect 来查看并复制 session token 到剪贴板。

第六步，编辑 wechat-chatgpt 的 config.yaml，填写 session token；设置 private trigger keywords（可选）。

第七步，用 docker 来拉取 wechat-chatgpt

    docker pull holegots/wechat-chatgpt:latest。

第八步，启动 wechat-chatgpt：

    docker run -d --name wechat-chatgpt -v $(pwd)/config.yaml:/app/config.yaml holegots/wechat-chatgpt:latest

注意，如果手动模式下也可以用npm run dev启动。如果提示系统不认识 npm 则可以运行 `npm install && poetry install` 来解决。到此你就可以在微信上跟这个打通了 ChatGPT 的账号聊天了。

|![image](/img/src/2022-12-11-wechat-chatgpt-1.png){: style="width:100%"} | ![image](/img/src/2022-12-11-wechat-chatgpt-2.png){: style="width:100%"}|
|---------------------------------|---------------------------------|
|||

其实可以看到这个 AI 船长不管是专业性问题（计算机相关）还是非专业问题，都回答的很不错。

如何停止、重启、查看日志呢？首先停止的命令是docker stop wechat-chatgpt，登录时需要扫码登录微信并追踪 logs，因为这其实是用了微信在桌面端的接口。

    docker logs -f wechat-chatgpt

会在 Terminal 里显示一个文字阵列组成的桌面端微信登录二维码，用你打算做成微信 AI 机器人那个微信号扫一下，相关信息都填完。另外，这样最好别用自己的微信大号，而是用一个小号。微信不让聊这些，小号注意要完成实名认证。

如果要停止运行，用如下命令：

    docker stop wechat-chatgpt

### 参考

1、[https://github.com/fuergaosi233/wechat-chatgpt/tree/main](https://github.com/fuergaosi233/wechat-chatgpt/tree/main)