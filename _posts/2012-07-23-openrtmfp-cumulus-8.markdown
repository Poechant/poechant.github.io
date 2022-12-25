---
layout: post
title:  OpenRTMFP/Cumulus 原理及源码解读 8：经由服务器的 Pub/Sub 流程的关键点
date:   2012-07-23 11:07:43 +0800
categories: rt_tech
tags: [直播技术]
description: 
excerpt: 
---

**本文目录**
* TOC
{:toc}

整个流程概括如下：

Flash 客户端通过 `NetConnection` 与 `Cumulus` 建立连接，然后通过 `NetStream` 使用 RTMFP 发布 Audio/Video/Data（下面简称为 A/V/D） 给服务器，这个 Flash Player 就作为一个发布者（Publisher）。RTMFP 服务器接收到后给所有的订阅者（Subscribers）发送 Audio/Video/Data。

### 1、客户端发布（Publishing on client side）

通过 `NetConnection` 连接 RTMFP 服务器 Cumulus，可以参考[《OpenRTMFP/Cumulus 原理及源码解读 1：入门介绍、部署与 Hello World》](/2012/04/10/openrtmfp-cumulus-1/)一文。关键的一个语句如下，其中 `nc` 是一个 `NetConnection` 对象。

```actionscript
nc.connect("rtmfp://localhost:1935");
```

在连接成功后通过 NetStream 发布 Audio/Video，如下所示，其中 `ns1` 是一个 `NetStream` 对象。

```actionscript
ns1.publish("poechant_media_flow", "live");
```

根据音视频不同的需求，播放相应内容。如果是发布 Data，则使用NetStream.send()来实现。这样就完成了客户端的 A/V/D 发布

### 2、服务器端（Server-side）

Cumulus 通过 `RTMFPReceiving` 这个 RTMFP 协议数据接收引擎完成一些连接建立的相关动作，以及接收数据包：

```c++
void RTMFPServer::receive(RTMFPReceiving& rtmfpReceiving);
```

该函数会在收到客户端发来请求时响应，如果是仍未建立连接的请求，则由此创建 Session（RTMFP 的核心概念之一），并取出其中的数据包。这其中有多个过程，我这里就不详述，以后会发布文章来解释。

继续我们的话题，在RTMFPServer::receive 函数中如果是建立连接阶段，则会调用 `Handshake` 类的 `receive` 来做接下来的处理，这个我就不去详细分析了，因为与本文主题无关。与本文有关的是，如果是已经创建了 Session 的，则会调用：

```c++
void ServerSession::packetHandler(PacketReader& packet);
```

这是一个相对复杂的函数，会从 packet 中取出很多有用的信息。此外，比较重要的是，在我们上述情况下，会调用 Flow 类的：

```c++
void Flow::fragmentSortedHandler(UInt64 stage,PacketReader& fragment,UInt8 flags);
```

该函数中会对 Audio/Video/Data 分别响应不同的处理机制：

```c++
switch(type) {
    case Message::AMF_WITH_HANDLER:
    case Message::AMF:
        messageHandler(name,amf);
        break;
    case Message::AUDIO:
        audioHandler(*pMessage);
        break;
    case Message::VIDEO:
        videoHandler(*pMessage);
        break;
    default:
        rawHandler(type,*pMessage);
}
```

接下来在 `Publication` 中完成对所有订阅了该发布者的 Flash Players 发送信息，核心的代码为：

```c++
for (it = _listeners.begin(); it != _listeners.end(); ++it) {
    it->second->pushAudioPacket(time,packet);
    packet.reset(pos);
}
 
for(it=_listeners.begin();it!=_listeners.end();++it) {
    it->second->pushVideoPacket(time,packet);
    packet.reset(pos);
}
 
for(it=_listeners.begin();it!=_listeners.end();++it) {
    it->second->pushDataPacket(name,packet);
    packet.reset(pos);
}
```

其中的 `_listeners` 就是该 `Publication` 中的所有订阅者。订阅者的添加/删除是通过：

```c++
Listener& addListener(
    Peer& peer,
    Poco::UInt32 id,
    FlowWriter& writer,
    bool unbuffered);
 
void removeListener(
    Peer& peer,
    Poco::UInt32 id);
```

这两个函数来实现的。

要注意的是，在 Publication 中已经完成了向订阅者发布信息，之后虽然会响应到 Peer 及 RTMFPServer 的onAudioPacket、onVideoPacket、onDataPacket，但此时都与订阅者接收信息无关了。Cumulus 正是在RTMFPServer::onAudioPacket、RTMFPServer::onVideoPacket、RTMFPServer::onDataPacket中调用用户定制的服务（Lua 脚本实现），完成一些自定义的需求。我是在此通过直接的 C++ 功能扩展，来添加业务需求的，没有使用 Lua 脚本及 Cumulus 中的 Lua 脚本引擎，主要原因是为了提高效率。

### 3、客户端订阅（Subscribing on client side）

订阅很简单，在 play 的时候传入正确的发布者名称即可。

```
ns2.play("poechant_media_flow");
```

测试代码可以参考 Reference-1，其中的例子是关于 `NetStream::send(…)` 的，用于发送 `Data`，`Audio` 和 `Video` 的程序可以参考该例修改。

客户端订阅后，这些信息并不会直接从发布者那里通过 P2P 的方式接收。如果想使用发布者与接受者直接连接的方式，则需要在 `NetStream` 初始化的时候，传入 `NetStream.DIRECT_CONNECTIONS` 参数，默认的 `NetStream.CONNECT_TO_FMS` 是将数据上行到服务器再下行给所有订阅者（Subscribers）的。根据不同的应用场景，可以使用不同的方式。

### 4、Reference

* http://help.adobe.com/en_US/FlashPlatform/reference/actionscript/3/flash/net/NetStream.html#send()