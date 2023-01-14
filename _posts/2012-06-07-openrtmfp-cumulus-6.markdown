---
layout: post
title:  麦克船长的 OpenRTMFP/Cumulus 原理、源码及实践 6：独立使用 CumulusLib 的线程安全 Bug 修复方法
date:   2012-06-07 23:34:18 +0800
categories: rt_tech
tags: [直播技术]
description: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。对于使用 Cumulus 来做二次开发的技术人员，CumulusLib 是一定会使用到的，但是 CumulusLib 的源码在被单独使用时是存在严重的线程安全 Bug 的，这就是本文诞生的原因。YY 的网页版流媒体技术服务端使用到 CumulusLib 时遇到了这个问题，因此修复了这个 Bug。最终的 Bug 修复很简单，但是要先理解 CumulusLib 整体线程安全问题才能确定解决方案。
excerpt: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。对于使用 Cumulus 来做二次开发的技术人员，CumulusLib 是一定会使用到的，但是 CumulusLib 的源码在被单独使用时是存在严重的线程安全 Bug 的，这就是本文诞生的原因。YY 的网页版流媒体技术服务端使用到 CumulusLib 时遇到了这个问题，因此修复了这个 Bug。最终的 Bug 修复很简单，但是要先理解 CumulusLib 整体线程安全问题才能确定解决方案。
location: 广州
author: 麦克船长
---

OpenRTMFP/Cumulus 提供了 `CumulusLib` 可以供其他 RTMFP 应用使用，而不局限于 `CumulusServer`。

一般来说，Thread A 会准备好要 `push` 的消息，然后 Thread A 向消息队列 `push` 消息。

但是 `CumulusLib` 中实现的，是 Thread A 向消息队列 `push` 消息，然后根据这个消息在队列中的指针，再向消息内填写字段。并期望如下：

![image](/img/src/2012-06-07-openrtmfp-cumulus-6-1.png)

由于在 `CumulusServer` 中，一个 Client 只在一个线程内被操作，相应的 `FlowWriter` 也不会出现跨线程的问题。但是如果单独使用 `CumulusLib`，如果出现线程通信，并且共享 `FlowWriter` 的话，就会共享消息队列，此时可能出现这种情况。

![image](/img/src/2012-06-07-openrtmfp-cumulus-6-2.png)

这就导致了很严重的错误，会使得进程崩溃。修正的方式，可以是将消息完全准备好之后，再放入队列，如下：

```c++
/*
 * author:  michael
 * date:    June 6th, 2012
 * type:    add
 */
MessageBuffered* FlowWriter::createAMFMessage(const std::string& name)
 
    // signature.empty() means that we are on the flowWriter of FlowNull
    if (!(_closed || signature.empty() || _band.failed())) {
        MessageBuffered* pMessage = new MessageBuffered();
        MessageBuffered& message(*pMessage);
        writeResponseHeader(message.rawWriter,name,0);
        return pMessage;
    }
 
    MessageBuffered& message(_MessageNull);
    writeResponseHeader(message.rawWriter,name,0);
    return NULL;
}
```

然后再调用时最后再增加 `push` 操作：

```c++
/*
 * author:  michael
 * date:    June 6th, 2012
 * type:    add
 */
void FlowWriter::pushAMFMessage(MessageBuffered* pMessage) {
    if (pMessage != NULL) {
        _messages.push_back(pMessage);
    }
}
```

这样就使得消息的数据被写完了，才被放入队列中，如下：

![image](/img/src/2012-06-07-openrtmfp-cumulus-6-3.png)

不过如果考虑线程安全，多个线程对同一个消息队列进行操作时，就要加锁：

```c++
/*
 * author:  michael
 * date:    June 6th, 2012
 * type:    add
 */
void FlowWriter::pushAMFMessage(MessageBuffered* pMessage) {
    if (pMessage != NULL) {
        Poco::Mutex::ScopedLock lock(msgQueueMutex);
        _messages.push_back(pMessage);
    }
}
```

这样就基本解决了这个线程安全问题。

另外，使用 `CumulusLib` 要遵循 GPL 协议，一定不要忘记。