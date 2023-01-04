---
layout: post
title:  麦克船长的 OpenRTMFP/Cumulus 原理、源码及实践 3：CumulusServer 源码主进程主循环分析
date:   2012-04-15 22:26:58 +0800
categories: rt_tech
tags: [直播技术]
description: CumulusServer 主进程的主循环分析，看本文一篇就够了。从绑定地址开始，本文介绍了如何接收数据，如何在 CumulusEdge 和 CumulusServer 的 socket 不同情况下的处理逻辑，如何处理发送方 IP 被禁、数据包大小异常等问题。通过本文让你了解 CumulusServer 的主循环，需要你对 POCO 库有一点了解，还要稍微熟悉 C++ 的基本语法。
excerpt: CumulusServer 主进程的主循环分析，看本文一篇就够了。从绑定地址开始，本文介绍了如何接收数据，如何在 CumulusEdge 和 CumulusServer 的 socket 不同情况下的处理逻辑，如何处理发送方 IP 被禁、数据包大小异常等问题。通过本文让你了解 CumulusServer 的主循环，需要你对 POCO 库有一点了解，还要稍微熟悉 C++ 的基本语法。
location: 广州
---

**本文目录**
* TOC
{:toc}

`CumulusServer` 主进程的主循环分析，看本文一篇就够了。从绑定地址开始，本文介绍了如何接收数据，如何在 `CumulusEdge` 和 `CumulusServer` 的 socket 不同情况下的处理逻辑，如何处理发送方 IP 被禁、数据包大小异常等问题。通过本文让你了解 `CumulusServer` 的主循环，需要你对 POCO 库有一点了解，还要稍微熟悉 C++ 的基本语法。

本所要介绍的这个主循环在 `RTMFPServer::run(const volatile bool& terminate)` 函数中。RTMFPServer覆盖 `Startable` 的 `run(const volatile bool &terminate)` 方法。

```c++
void RTMFPServer::run(const volatile bool& terminate) {
```

### 1、绑定地址

`CumulusServer` 的 IP 地址和端口：

```c++
    SocketAddress address("0.0.0.0",_port);
    _socket.bind(address,true);
绑定CumulusEdge的 IP 地址和端口：

```

```c++
    SocketAddress edgesAddress("0.0.0.0",_edgesPort);
    if (_edgesPort>0)
        _edgesSocket.bind(edgesAddress,true);
```

发送者（Client）的 IP 地址和端口：

```c++
    SocketAddress sender;
    UInt8 buff[PACKETRECV_SIZE];
    int size = 0;
 
    while (!terminate) {
 
        bool stop=false;
        bool idle = realTime(stop);
        if(stop)
            break;
 
        _handshake.isEdges=false;
```

### 2、`CumulusServer` 接收数据

`CumulusServer` 的 `socket` 有数据可读：

```c++
        if (_socket.available() > 0) {
            try {
```

从 `socket` 读取：把数据存到 `buff`，把发送者地址赋给 `sender`，把所读长度返回给 `size`。

```c++
             size = _socket.receiveFrom(buff,sizeof(buff),sender);
```

处理 `CumulusServer` 的 `socket` 产生的异常：

```c++
            } catch(Exception& ex) {
                DEBUG("Main socket reception : %s",ex.displayText().c_str());
                _socket.close();
                _socket.bind(address,true);
                continue;
            }
```

### 3、如果 `CumulusEdge` 端口存在且 edge socket 可用。

`CumulusEdge` 的 `socket` 有数据可读：

```c++
        } else if (_edgesPort > 0 && _edgesSocket.available() > 0) {
            try {
                size = _edgesSocket.receiveFrom(buff, sizeof(buff), sender);
                _handshake.isEdges = true;
            } catch(Exception& ex) {
                DEBUG("Main socket reception : %s", ex.displayText().c_str());
                _edgesSocket.close();
                _edgesSocket.bind(edgesAddress, true);
                continue;
            }
            Edge* pEdge = edges(sender);
            if (pEdge)
                pEdge->update();
```

### 4、`CumulusServer` 和 `CumulusEdge` 的 `socket` 都没有数据：

```c++
        } else {
```

`CumulusServer` 空闲：

```c++
            if (idle) {
```

主线程等待一秒。

```c++
                Thread::sleep(1);
                if (!_timeLastManage.isElapsed(_freqManage)) {
```

Just middle session

```c++
                    if (_middle) {
                        Sessions::Iterator it;
                        for (it = _sessions.begin(); it != _sessions.end(); ++it) {
                            Middle* pMiddle = dynamic_cast<Middle*>(it->second);
                            if (pMiddle)
                                pMiddle->manage();
                        }
                    }
                } else {
                    _timeLastManage.update();
                    manage();
                }
            }
            continue;
        }
```

### 5、发送方的 ip 被禁：

```c++
        if (isBanned(sender.host())) {
            INFO("Data rejected because client %s is banned",
                sender.host().toString().c_str());
            continue;
        }
```

### 6、数据包长度小于可能的最小值（12）

```c++
        if (size < RTMFP_MIN_PACKET_SIZE) {
            ERROR("Invalid packet");
            continue;
        }
 
        PacketReader packet(buff,size);
        Session* pSession = findSession(RTMFP::Unpack(packet));
 
        if (!pSession)
            continue;
 
        if (!pSession->checked)
            _handshake.commitCookie(*pSession);
```

给 `CumulusEdge` 或者自己（`CumulusServer`）的 `socket`:

```c++
        pSession->setEndPoint(_handshake.isEdges ? _edgesSocket : _socket,sender);
        pSession->receive(packet);
    }
    _handshake.clear();
    _sessions.clear();
    _socket.close();
    if (_edgesPort>0)
        _edgesSocket.close();
    if(_pCirrus) {
        delete _pCirrus;
        _pCirrus = NULL;
    }
}
```