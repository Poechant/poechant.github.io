---
layout: post
title:  OpenRTMFP/Cumulus 原理、源码及实践 7：Cumulus 源码的一个线程启动 Bug 及修复方法
date:   2012-06-25 10:56:26 +0800
categories: rt_tech
tags: [直播技术]
description: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。Cumulus 启动后，我们可以看到有多个线程被创建，但是有时其中的个别线程没有被成功启动，本文将告诉你如何修复并解决。
excerpt: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。Cumulus 启动后，我们可以看到有多个线程被创建，但是有时其中的个别线程没有被成功启动，本文将告诉你如何修复并解决。
location: 广州
---

`Cumulus` 中的线程都是继承自 `Startable`，在其中封装 `Poco::Thread` 成员，使得一些有关线程的操作更方便。`Startable` 中的 `start` 函数如下：

```c++
void Startable::start() {
    if(!_stop) // if running
        return;
    ScopedLock
  
    lock(_mutex);
    
   if(_haveToJoin) {
        _thread.join();
        _haveToJoin=
   false;
    }
    
   try {
        DEBUG(
   "Try to start up a new thread inherited from Startable");
        _thread.start(_process);
        _haveToJoin = 
   true;
        ScopedLock
   
     lock(_mutexStop);
        _stop=
    false;
    } 
    catch (Poco::Exception& ex) {
        ERROR(
    "Impossible to start the thread : %s",ex.displayText().c_str());
    }
}
``` 

这样一个类继承 `Startable` 的话，并启动时传入自己，则会调用到 `Startable::start()`，然后调用到该类自己的 `run()` 函数。一般来说这个函数会一个循环，以 `SocketManager` 为例：

```c++
void SocketManager::run() {
    … 
    while(running()) {
    …
    }
}
```

我们要看看这个 `running()` 是怎么回事，如下：

```c++ 
inline bool Startable::running() const {
    return !_stop;
}
```

很简单，就是通过 `Startable::_stop` 成员来判断是否还需要继续循环下去。那么这个 `_stop` 是什么时候被设置为 `false` 的呢？就是上面的 `start()`，这里存在的一个问题就是先 `start` 线程，再设置 `_stop` 为 `false`。

``` 
_thread.start(_process);
_stop=false;
```

而 `start()` 之后 `run()` 的时候就开始通过 `running()` 来判断 `_stop` 值了。所以你会在使用 `Cumulus` 时，发现有时候启动起来的线程个数不对。正常情况下应该有四个线程：

![image](/img/src/2012-06-25-openrtmfp-cumulus-7-1.png)


它们是：

* 主线程
* `RTMFPServer` 线程
* `MainSockets` 线程
* `RTMFPManager` 线程

而异常情况可能是 `MainSockets` 没有启动，甚至 `MainSockets` 和 `RTMFPManager` 都没有启动。

`MainSockets` 没有启动的情况，这时客户端是无法接入成功的。

![image](/img/src/2012-06-25-openrtmfp-cumulus-7-2.png)

`MainSockets` 和 `RTMFPManager` 都没有启动的情况 T.T

![image](/img/src/2012-06-25-openrtmfp-cumulus-7-3.png)

具体是哪个线程没有启动成功可以通过 GDB 查看。

解决办法就是将 `_stop` 的设置操作，在启动线程之前。不过要注意锁要同时移动，并且在产生异常时设置 `_stop` 值为 `true`。

```c++
void Startable::start() {
    if(!_stop) // if running
        return;
    ScopedLock
  
    lock(_mutex);
    
   if(_haveToJoin) {
        _thread.join();
        _haveToJoin=
   false;
    }
    
   try {
        DEBUG(
   "Try to start up a new thread inherited from Startable");
        {
            ScopedLock
   
     lock(_mutexStop);
            _stop=
    false;
        }
        _thread.start(_process);
        _haveToJoin = 
    true;
    } 
    catch (Poco::Exception& ex) {
        {
            ScopedLock
    
      lock(_mutexStop);
            _stop = 
     true; 
     // June 25th, 2012, Michael@YY
        }
        ERROR(
     "Impossible to start the thread : %s",ex.displayText().c_str());
    }
}
```