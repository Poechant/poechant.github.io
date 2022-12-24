---
layout: post
title:  一名出色软件工程师的技术基本功：编程与工具 / 2012#1
date:   2012-05-16 01:06:59 +0800
categories: thinking
tags: [思考]
description: 再过一个多月，我就毕业工作一年了。目前在广州的 YY 语音，是 Web YY 音视频媒体技术负责人，公司预计在下半年上市，我希望通过 Web 版 YY 能为用户更容易访问（免注册、免登陆）来拉动 YY 的 DAU（活跃用户人数）助力 YY 上市。夜深人静，写一些自己对于出色软件工程师技术基本功的理解。
excerpt: 再过一个多月，我就毕业工作一年了。目前在广州的 YY 语音，是 Web YY 音视频媒体技术负责人，公司预计在下半年上市，我希望通过 Web 版 YY 能为用户更容易访问（免注册、免登陆）来拉动 YY 的 DAU（活跃用户人数）助力 YY 上市。夜深人静，写一些自己对于出色软件工程师技术基本功的理解 ……
---

### 0、写在前面

再过一个多月，我就毕业工作一年了。目前在广州的 YY 语音，是 Web YY 音视频媒体技术负责人，公司预计在下半年上市，我希望通过 Web 版 YY 能为用户更容易访问（免注册、免登陆）来拉动 YY 的 DAU（活跃用户人数）助力 YY 上市。夜深人静，写一些自己对于出色软件工程师技术基本功的理解。

### 1、编程

首先至少精通一门高级语言（注意是精通），然后要熟悉额外的几门语言。举例来说：

#### 如果你精通 C 语言

那么除了其语言标准之外，还要精通 Linux 平台的系统 API，以及一些常用的库，还有单元测试工具。当然，如果你需要精通 C 语言的话，应该是需要你经常做与操作系统直接接触的应用底层开发，或者编写一些基础库。

#### 如果你精通 C++ 语言

那么除了 C++ 语言标准外，你应该还要精通 STL（虽然这已经纳入 C++ 标准，但是我还是要提两句），以及一些常用的库，比如 Boost、ACE、POCO 等。

另外，精通 C/C++ 要求你必须要会用 GCC/G++、GDB、Makefile（整合 Makefile 的 CMake 等）/Scons 等等。

#### 精通的关键，还是针对语言核心来说的。

第一，你要对这个语言的语法特性熟稔；

第二，你要对这个语言的标准库的每个 API 熟稔；

第三，你要能够熟练运用这门语言编写各种设计模式；

第四，你能够运用你对这门语言的掌握，完成任意给定的编程任务。

那么，其他额外要熟悉的语言，你要做到有的放矢，就是当你要进行某种开发的时候，你在这方面能够熟练使用这门语言。比如你可以用 PHP 熟练地进行 Web 开发，你可以用 Perl 熟练地处理文本，你可以用 Bash 熟练地编写脚本小工具。

#### 与计算机、网络的基础结构相关联的技术实现

除了这些呢，设计模式、异步 IO、进程与线程、网络编程也是你必须精通的。当然，你只要精通你所使用的语言的这些方面的就可以了。

### 2、工具

对于工具有三个层面：

第一，是熟练的使用一些工具。

第二，是能够发现提高生产力的工具。

第三，是能够在无可用工具时自己编写工具。

那么都有哪些最最最基本的工具呢？

#### IDE（Integrated Development Environment）

第一自然是 IDE，这是程序员的武器。如果你是 Windows 下的 C/C++ 开发者，建议你使用 Visual Studio，不要小看它，如果你能够精通它，你也算是一个高手。如果你是 Mac 下的C/C++/Objective-C 开发者，可以选择 XCode、Eclipse，并配合 Vim/Emacs 使用。如果你是 Linux 下的开发者，可以使用 Vim/Emacs。

#### VCS（Version Control System）

VCS 可以分为两类，一类是 CVCS（Central VCS），另一类是 DVCS（Distributed VCS）。现在 CVCS 一般使用 SVN、CVS，DVCS 一般使用 Git、Mercurial（Hg）。至于 CVCS 和 DVCS 的区别，道地谁先进，我喜欢下面这段比喻：

> Once you understand the conceptual differences between CVS/SVN and Git, and then subsequently start to use Git, you may find it very difficult to go back. You should really start to experiment only if you think you're going to migrate in the near future, because using Git is like watching TV in colour: once you've discovered it, it's really difficult to go back to black & white.

一旦你使用了 VCS，你就会接触到 Google Code、Github、BitBucket 等等。它们其实可以算是一种在线工具。

#### CLI（Command Line Interface）

我们一般都说命令行（Command Line），为什么还带一个「I」呢？类比 API（Application Program Interface）、GUI（Graphical User Interface）就能明白了，这都是与某个系统的交互接口，API 是通过一些 Library 调用实现交互，GUI 是通过在图形界面上的点击/拖动/滑动等实现交互。熟练地运用操作系统的 CLI。无论你是使用 Linux、Mac、Solaris、FreeBSD，甚至是 Windows，你都要熟练使用 CLI。

### 3、结语

还能想到什么？由于现在夜深人静，头脑不够清醒，只能想到这些。况且在这些方面，我也达不到「精通」，甚至想去甚远。那姑且先这样吧，如果哪位朋友有什么想说的，可以在下面给我留言，我会补充到文中。