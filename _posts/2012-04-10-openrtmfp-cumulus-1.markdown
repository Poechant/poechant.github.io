---
layout: post
title:  OpenRTMFP/Cumulus原理及源码解读-1：入门介绍、部署与 Hello World
date:   2012-04-10 02:57:19 +0800
categories: rt_tech
tags: [直播技术]
description: RTMFP 是 Adobe 开发的基于 UDP 协议的实时传输媒体流协议，支持 P2P 传输，具有较高的实时性和安全性。它的主要应用场景是视频通信、语音通信和网络游戏。OpenRTMFP 是一个开源的 RTMFP 实现，可以用于构建基于 RTMFP 的应用程序。Cumulus 是一个基于 OpenRTMFP 的服务器，提供 RTMFP 服务。它具有轻量级、跨平台和可扩展的特点，并且还提供了负载均衡和可扩展性解决方案。YY 语音的 Web 端音视频流媒体能力，正是基于 RTMFP 协议做的迭代优化实现的。本文是船长关于这个系列文章的第一篇。
excerpt: RTMFP 是 Adobe 开发的基于 UDP 协议的实时传输媒体流协议，支持 P2P 传输，具有较高的实时性和安全性。它的主要应用场景是视频通信、语音通信和网络游戏。OpenRTMFP 是一个开源的 RTMFP 实现，可以用于构建基于 RTMFP 的应用程序。Cumulus 是一个基于 OpenRTMFP 的服务器，提供 RTMFP 服务。它具有轻量级、跨平台和可扩展的特点，并且还提供了负载均衡和可扩展性解决方案。YY 语音的 Web 端音视频流媒体能力，正是基于 RTMFP 协议做的迭代优化实现的。本文是船长关于这个系列文章的第一篇 ……
---

**本文目录**
* TOC
{:toc}

### 一、RTMFP 是什么？

Real-Time Media Flow Protocol（RTMFP）是 Adobe 开发的一种基于 UDP 并支持 P2P 的实时传输媒体流。主要特点是：高传输效率（可以使用压缩和算法来优化流量从而提高传输效率）、高实时性（可以保证媒体流的实时性使得视频通信和其他实时通信更加流畅）、支持 P2P 传输（减少对服务器的依赖从而减少带宽和服务器资源消耗）、高安全性（加密媒体流从而保证其安全性）。

RTMFP 的主要应用场景包括：视频通信（视频聊天和视频会议）、语音通信（语音聊天、电话）、网络游戏。不过 RTMFP 目前仅有 Adobe 开发的版本，所以它并不是个开源项目，而是个商业化服务。那么有没有开源版本呢？

#### 文件分享 P2P 和实时流媒体 P2P 的区别是什么？

RTMFP 是一个 P2P 系统，但它仅针对实时通信时直接用户到用户之间的通信而设计，不能用于多个对等方之间进行文件共享（使用分段下载）。Facebook 在其 Pipe 应用中使用此协议将大文件直接在两个用户之间传输。

#### RTMFP 和 RTMP 之间的区别是什么？

RTMP 是实时消息协议，RTMFP 代表实时媒体流协议。RTMFP 基于用户数据报协议（UDP），而 RTMP 基于传输控制协议（TCP）。
与 RTMP 不同，RTMFP 还支持直接从一个 Adobe Flash Player 传输数据到另一个，而无需经过服务器。

#### Flash Player 支持 RTMFP 吗？

RTMFP 是基于用户数据报协议（UDP）的，而 RTMP 是基于传输控制协议（TCP）的。与 RTMP 不同，RTMFP 还支持直接在两个 Adobe Flash Player 之间发送数据，而不经过服务器。Flash Player 10.0 仅允许一对一通信进行 P2P，但从 10.1 开始允许应用程序级别的多播。Flash Player 查找适当的分发路由（覆盖网络），并可以将其分发到通过 P2P 连接的组。

#### Cumulus 使用 Adobe 的 Cirrus Key 吗？

不！当然，这是Cumulus的主要目标：成为Cirrus GPL的替代品。唯一的限制是：你的CPU，内存和单台机器的端口数。

#### 这个开源项目合法吗？

在美国，数字千年版权法（Digital Millennium Copyright Act）规定，逆向工程用于协议互操作性是合法的。你可以在 WikiPedia 上查看相关讨论：

* http://en.wikipedia.org/wiki/Real_Time_Media_Flow_Protocol
* http://en.wikipedia.org/wiki/Proprietary_protocol
* http://en.wikipedia.org/wiki/Digital_Millennium_Copyright_Act

当逆向工程的目的是协议互操作性时，有法律先例。在美国，数字千年版权法（Digital Millennium Copyright Act）为逆向工程软件以使其与其他软件互操作提供了安全保障。

### 二、OpenRTMFP 和 Cumulus

OpenRTMFP 是一个开源的 RTMFP 实现，可以用于构建基于 RTMFP 的应用程序。它包含了 RTMFP 协议的实现，以及一些额外的功能，如媒体流传输、P2P 通信、脚本引擎和数据存储。

Cumulus 是一个基于 OpenRTMFP 的服务器，是一个完整的开源且跨平台的 RTMFP 服务器，可通过脚本进行扩展。CumulusServer 根据 GPL 许可在考虑以下 4 个概念的情况下开发：速度、轻量、跨平台和可扩展。尽管尚未发布版本，但只有在 CumulusServer 经过测试和批准后才会将代码推送到 github。实际上，主要稳定的功能有：

*   P2P rendez-vous service
*   live streaming
*   RPC、pull、push exchange，实际上客户端和服务器之间的所有 AMF 可能交换
*   脚本引擎，用于创建自己的应用服务器或扩展 Cumulus 的功能
*   可扩展性和负载均衡解决方案

下面的内容是本篇 blog 的重点，包括两部分：先是 OpenRTMFP 应用的核心 CumulusServer 的入门介绍与部署，然后用 Lua 编写 HelloWorld 应用扩展 CumulusServer，我们开始吧！

### 三、入门介绍与部署 CumulusServer

#### 1、背景介绍

OpenRTMFP 可以帮助你实现 Flash 的实时应用的高并发扩展，OpenRTMFP/Cumulus 是基于 GNU General Public License 的。

POCO：POrtable COmponents，是一个强大的开源 C++ 库。其在 C++ 开发中的角色，相当于 Java Class Library、苹果的 Cocoa、.NET framework。

#### 2、准备工作

下载：

|  **External Dependencies**  |  **Official Site**  |  **Windows**  |  **Linux/OSX**  |
|---|---|---|---|
|  OpenSSL  |  [Official Site](http://www.slproweb.com/products/Win32OpenSSL.html)  |  [Download](http://www.slproweb.com/download/Win32OpenSSL_Light-1_0_1.exe)  |  [Download](http://www.openssl.org/source/openssl-1.0.1.tar.gz)  |
|  Lua  |  [Official Site](http://www.lua.org/)  |  [Download](http://luaforwindows.googlecode.com/files/LuaForWindows_v5.1.4-45.exe)  |  [Download](http://www.lua.org/ftp/lua-5.1.5.tar.gz)  |
|  POCO  |  [Official Site](http://pocoproject.org/)  |  [Download](http://downloads.sourceforge.net/project/poco/sources/poco-1.4.3/poco-1.4.3p1.zip)  |  [Download](https://sourceforge.net/projects/poco/files/sources/poco-1.4.3/poco-1.4.3p1.tar.gz/download)  |

注意：POCO for linux  版本必须是 1.4.0 或更高，否则会引起 TCP 相关的 bug。

#### 3、安装

##### 3.1、外部依赖的安装

Windows 下略，Linux 下基本就是：

~~~sh
$ ./configuremakesudo
$ make install
~~~

##### 3.2、安装 OpenRTMFP/Cumulus

```sh
$ cd OpenRTMFP-Cumulus/CumulusLib
$ make
$ cd ../CumulusServer
$ make
```

如果出现了 `.h` 文件、lib 库找不到的情况，请修改 OpenRTMFP-Cumulus/CumulusLib/Makefile 或 OpenRTMFP-Cumulus/CumulusServer/Makefile。

#### 4、配置

通过编写 `OpenRTMFP-Cumulus/CumulusServer/CumulusServer.ini` 文件来为 OpenRTMFP-Cumulus 进行个性化配置（默认是没有这个文件的），这个文件的内容形如：

```lua
;CumulusServer.ini
port = 1985
udpBufferSize = 114688
keepAlivePeer = 10
keepAliveServer = 15
[logs]
name=log
directory=C:/CumulusServer/logs
```

一些字段的设置含义如下，摘自：[地址](https://github.com/OpenRTMFP/Cumulus/wiki/Installation)。

* 公开给 Client 的端口号 `port`，默认值是 1935（RTMFP 服务器的默认端口），用于 CumulusServer 监听 RTMFP 请求。
* UDP 缓冲区字节数 `udpBufferSize`, allows to change the size in bytes of UDP reception and sending buffer. Increases this value if your operating system has a default value too lower for important loads.
* `keepAliveServer`, time in seconds for periodically sending packets keep-alive with server, 15s by default (valid value is from 5s to 255s).
* `keepAlivePeer`, time in seconds for periodically sending packets keep-alive between peers, 10s by default (valid value is from 5s to 255s).
* `edges.activated`, activate or not the edges server on the RTMFP server (see CumulusEdge, Scalability page for more details about CumulusEdge). By default, CumulusServer stays a RTMFP server without edges ability (default value is false).
* `edges.port`, port for the edges server, to accept incoming new CumulusEdge instances (see CumulusEdge, Scalability page for more details about CumulusEdge). By default, it’s the port 1936. 

> Warning: This port will receive plain text request from edges, for this purpose it should not be made public. It’s very important for security consideration. It must be available only for CumulusEdge instances, and anything else.

* `edges.attemptsBeforeFallback`, number of CumulusEdge attempt connections before falling back to CumulusServer (see CumulusEdge, Scalability page for more details about CumulusEdge). By default the value is 2 (in practical, 2 attempts happens after 5 sec approximately).
* SMTP IP 地址 `smtp.host`, configure a SMTP host to use mails feature provided by Cumulus in server application (see Server Application, Sockets page for more details about mails feature). By default the value is localhost.
* SMTP 端口 `smtp.port`, configure a SMTP port to use mails feature provided by Cumulus in server application (see Server Application, Sockets page for more details about mails feature). By default the value is 25.
* `smtp.timeout`, configure a SMTP timeout session in seconds to use mails feature provided by Cumulus in server application (see Server Application, Sockets page for more details about mails feature). By default the value is 60 seconds.
* 日志路径 `logs.directory`，默认是 `CumulusServer/logsby`。
* 日志文件名称 `logs.name`，默认是`log`。
    

#### 5、启动

Windows 下的启动方法为：

~~~dos
$ CumulusServer.exe /registerService [/displayName=CumulusServer /description="Open Source RTMFP Server" /startup=automatic]
~~~

Unix-like 下的启动方法为：

```sh
$ sudo ./CumulusServer --daemon [--pidfile=/var/run/CumulusServer.pid]
```

具体地，我的启动命令为：

```sh
$ sudo ./CumulusServer --daemon --pidfile=./CumulusServer.pid
```

#### 6、基本使用

本地 Flash client 可以通过如下语句连接：

```as
$ var nc:NetConnection = new NetConnection();nc.connect("rtmfp://localhost/");
```

RTMFP默认是采用1935端口，如果你特别指定了其他端口，比如12345，请使用如下方式：

    nc.connect("rtmfp://localhost:12345/");

#### 7、扩展 CumulusServer（Server Application）

启动CumulusServer后，会在可执行文件的目录下出现一个www目录，该目录的作用，就是作为 Server Application 的默认根目录。具体的对应关系如下：

> rtmfp://host:port/                            -> \[CumulusServer folder\]/www/main.lua (root application)

> rtmfp://host:port/myApplication     -> \[CumulusServer folder\]/www/myApplication/main.lua

> rtmfp://host:port/Games/myGame -> \[CumulusServer folder\]/www/Games/myGame/main.lua

另外要提醒的是，如果main.lua文件被修改，则不需要重启 CumulusServer，因为 Server Application 的创建是一种动态的方式。

### 三、用 Lua 编写 HelloWorld 应用扩展 CumulusServer

下面的这个实例是在本地（Client 与 Server 位于同一机器上）测试的。

#### 1、Server-side

##### 1.1、Server configuration

```lua
; CumulusServer.ini
port = 1935
udpBufferSize = 114688
keepAlivePeer = 10
keepAliveServer = 15
[logs]name = log
directory = logs
```

##### 1.2、Application file

```lua
    function onConnection(client,response,...)
      function client:test(...)
        name,firstname = unpack(arg)
        return "Hello "..firstname.." "..name
      end
    end
```

#### 2、Client-side

```java
// CumulusClient.as

package {
  import flash.display.Sprite;
  import flash.net.NetConnection;
  import flash.net.NetStream;
  import flash.net.Responder;

  public class CumulusClient extends Sprite {
    private var nc:NetConnection = null;
  	private var ns:NetStream = null;
  	
  	public function CumulusClient() {
      nc = new NetConnection();
      nc.connect("rtmfp://localhost");
      nc.client = this;
      nc.call("test",new Responder(onResult,onStatus), "OpenRTMFP/Cumulus", "World")
    }
  
  	public function close():void {            
			nc.close();
  	}
  
  	public function onStatus(status:Object):void {
    	trace(status.description)
	  }
  
  	public function onResult(response:Object):void {
    	trace(response) // expected to display "Hello World OpenRTMFP/Cumulus"        
	  }    
	}
}
```

#### 3、运行结果

```
Hello World OpenRTMFP/Cumulus
[SWF] CumulusClient.swf - 解压缩后为 1,776 个字节
[卸装 SWF] CumulusClient.swf
```

#### 4、远程测试：一个免费的测试服务器

获取 Developer Key 的地址：

```
http://108.59.252.39:8080/CumulusServer/index.jsp
```

服务器配置信息：

```
Server: amd64 OS: Linux 2.6.18-028stab095.1
Server IP: 108.59.252.39
OpenRTMFP as of: 22.Feb.2012
```

编写服务器段应用地址：

```
http://108.59.252.39:8080/CumulusServer/manage_ssls.jsp
```

快去试试吧 :)