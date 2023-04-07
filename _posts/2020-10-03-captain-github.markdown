---
layout: post
title:  
date:   2020-10-03 04:59:21 +0800
categories: thinking
tags: [思考]
description: 
excerpt: 
location: 杭州
author: 麦克船长
---

### Branch 分支

将一个分支 src_branch，移动成另一个分支 dest_branch，不保留原来的分支：

```shell
git checkout dest_branch
git merge --strategy=ours src_branch
git branch -D src_branch
```

其中我们使用了 ours 策略，就是对于所有冲突都采用当前的分支，即上述示例中的 dest_branch。

### Action 和 Workflow 工作流的使用

可以在工作流程文件中使用 `on` 关键字来指定在**什么动作时**、**哪个分支上**触发工作流程，例如在 `push` 到 `master` 分支时触发：

```yaml
on:
  push:
    branches:
      - master
```

也可以使用 deployment 关键字来指定将工作流程部署到哪个分支，例如部署到生产环境的 `master` 分支：

```yaml
deployment:
  production:
    branch: master
```

#### 通过 GitHub Action 发布 Jekyll 网站

为了通过 GitHub Action 发布 Jekyll 网站，您需要以下步骤：

### 关于 SSH key 与 Github

1. 首先检查本地是否已经有 ssh key，可以使用命令 `ls -al ~/.ssh` 检查。如果没有 ssh key，需要先生成一个。
2. 生成 ssh key:
* 打开终端，输入 `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`，其中 `your_email@example.com` 是你在 GitHub 上注册的邮箱。
* 按提示输入文件名和密码，一般默认即可。
3. 将 ssh key 添加到 ssh-agent 中：输入 `ssh-add ~/.ssh/id_rsa`。
4. 将 ssh key 添加到 GitHub 上：输入 `cat ~/.ssh/id_rsa.pub`，将输出的 ssh key 添加到 GitHub 上的 ssh keys 中。
5. 设置 git 使用 ssh key：输入 `git remote set-url origin git@github.com:yourusername/yourrepo.git`，替换 `yourusername` 和 `yourrepo` 为你自己的用户名和仓库名。
6. 测试 ssh 连接：输入 `ssh -T git@github.com`

