---
layout: post
title:  OpenRTMFP/Cumulus 原理、源码及实践 9：关键线程逻辑分析
date:   2012-08-05 01:58:17 +0800
categories: rt_tech
tags: [直播技术]
description: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。本文对 RTMFPServer 线程、RTMFPManager 对 RTMFPServer 的影响进行源码解读。
excerpt: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。本文对 RTMFPServer 线程、RTMFPManager 对 RTMFPServer 的影响进行源码解读。
---

**本文目录**
* TOC
{:toc}

### 一、`RTMFPServer` 线程的启动和等待

#### 1、`Poco::Thread`

Cumulus 大量使用了 `Poco` 的线程库。一个简单的 Poco 线程的使用实例如下：

```c++
class PoechantRunnable: public Poco::Runnable {
    virtual void run() {
        // your codes
    }
};
 
int main() {
    PoechantRunnable runnable;  // Image that it's a gift
    Poco::Thread thread;        // And… thread is just like your girl
    thread.start(runnable);     // Okay, give your sweet babe the gift :)
    thread.join();
    return 0;
}
```

#### 2、封装一个可运行线程的类

`Cumulus` 中实现了一个 `StartableProcess` 类，该类继承了 `Runnable`，就是上面那个 `gift` 喽。

```c++
class StartableProcess : public Poco::Runnable{
public:
    StartableProcess(Startable& startable);
private:
    void run();
    Startable& _startable;
};
```

可以看到其中有 `Startable& _startable` 引用成员，它并没有继承 `Runnable`，而是封装了 `StartableProcess` 和 `Poco::Thread`：

```c++
Poco::Thread            _thread;
StartableProcess        _process;
```

这里 `Startable` 封装了一个 `StartableProcess` 成员，与 `StartableProcess` 是有所区别的。接下俩我们看他们是怎么用的。

#### 3、启动 `RTMFPServer` 线程
我们可以看到在 `Startable` 类的构造函数中初始化了 `_process` 成员，初始化线程成员并传入线程名，设定标志域 `（Flag Field）_stop` 为 `true`，因为它还没有调用启动函数。

```c++
Startable::Startable(const string& name)
    : _name(name),
      _thread(name),
      _stop(true),
      _haveToJoin(false),
      _process(*this) {
}
```

初始化 `_process` 时，调用 `StartableProcess` 构造函数：

```c++
StartableProcess::StartableProcess(Startable& startable)
    : _startable(startable){
}
```

传入 `_startable` 的引用。在 `Cumulus` 中所有的线程的可运行类都是继承自 `Startable` 类的，然后通过调用 `start()` 来启动，启动后会响应到 `run()`。下面我们以 `RTMFPServer` 线程为例。

`RTMFPServer` 类是继承自 `Startable` 类的：

```c++
class RTMFPServer
    : private Gateway,
      protected Handler,
      private Startable,
      private SocketHandler
```

`RTMFPServer` 的构造函数：

```c++
RTMFPServer::RTMFPServer(UInt32 cores)
    : Startable("RTMFPServer"),
      _sendingEngine(cores),
      _receivingEngine(cores),
      _pCirrus(NULL),
      _handshake(_receivingEngine,
      _sendingEngine,
      *this,
      _edgesSocket,*this,*this),
      _sessions(*this) {
}
```

其中在初始化时调用了其父类的构造函数。接下来就要启动RTMFPServer线程了。

| 所在线程      | 调用者                                               | 函数                    |
|--------------|-----------------------------------------------------|-------------------------|
| 主线程        | main(…)                                             |                         |  
| 主线程        | RTMFPServer对象                                      | RTMFPServer::start()   |
| 主线程        | RTMFPServer对象                                      | Startable::start()      |
| 主线程        | RTMFPServer从Startable继承来的Thread成员              | Thread::start(…)        |
| RTMFPServer  | RTMFPServer对象从Startable继承来的StartableProcess成员 | StartableProcess::run() |
| RTMFPServer  | RTMFPServer对象                                      | RTMFPServer::prerun()   |
| RTMFPServer  | RTMFPServer对象                                      | Startable::prerun()     |
| RTMFPServer  | RTMFPServer对象                                      | RTMFPServer::run()      |

#### 4、`RTMFPServer` 线程等待

在 `RTMFPServer::run()` 实现线程的持续运行，主要是依靠这两行代码：

```c++
while (!terminate)
    handle(terminate);
```

`handle(…)` 函数很简单，如下只进行了 `sleep(...)` 和 `giveHandle()` 两个操作。

```c++
void RTMFPServer::handle(bool& terminate){
    if (sleep() != STOP) {
        giveHandle();
    } else
        terminate = true;
}
```

`sleep(…)` 是 `RTMFPServer` 是从 `Startable` 继承而来的，声明如下：

```c++
WakeUpType sleep(Poco::UInt32 timeout=0);
```

定义如下：

```c++
Startable::WakeUpType Startable::sleep(UInt32 timeout) {
    if (_stop)
        return STOP;
     WakeUpType result = WAKEUP;
     if (timeout>0) {
         if (!_wakeUpEvent.tryWait(timeout))
             result = TIMEOUT;
     } else {
         _wakeUpEvent.wait();
     }
    if (_stop)
        return STOP;
    return result;
}
```

在运行状态下，`_stop` 为 `false`，而默认参数 `timeout` 为 0，所以会调用：

```c++
_wakeUpEvent.wait();
```

这个 `_wakeUpEvent` 成员是一个 `Poco::Event` 对象，`Poco::Event` 有一个使用方式就是在调用 `Poco::Event::wait()` 后，会一直等待 `Poco::Event::set()` 被调用后，才会跳出 `wait` 的状态。在 `Cumulus` 中 `set` 的动作是由：

* `RTMFPServer::requestHandle()`
* `PoolThread::push(Poco::AutoPtr<RunnableType>& pRunnable)`

执行的。

### 二、`RTMFPManager` 对 `RTMFPServer` 的影响

`RTMFPManager` 与 `RTMFPServer` 同样，继承自 `Startable`。

```c++
class RTMFPManager : private Task, private Startable
```

在构造函数中将 `RTMFPServer` 对象以引用方式传入，用以初始化其 `_server` 引用成员。

```c++
RTMFPManager(RTMFPServer& server)
    : _server(server),
      Task(server),
      Startable("RTMFPManager")  {
    start();
}

/* ...... */

RTMFPServer& _server;
```

在 `RTMFPManager` 的构造函数中调用 `start()` 成员函数，是从 `Startable` 继承而来的。然后会开启一个新的名为 `RTMFPManager` 的线程。然后响应到 `RTMFPManager::run()` 函数。

```c++
void run() {
    setPriority(Thread::PRIO_LOW);
    while(sleep(2000)!=STOP)
        waitHandle();
}
```

这里要强调的是，这里的 `setPriority` 在 Linux 环境下会设置失败，可以参见我在 `Cumulus` 在 Github 上开启的 Issue #75，其中就包括这里的线程优先级设置。

在这里我们可以看到 `RTMFPManager` 的 `handle(…)` 中的 `sleep(…)` 是每 2 秒一次，而这是对 `RTMFPServer` 线程有影响的。还记得我说的 `RTMFPServer` 线程的 `_wakeUpEvent` 成员吗？（在第一部分中）它的激活就是在 `RTMFPManager` 中进行的，所以这里这个 2 秒是会影响到 `RTMFPServer` 的主循环的等待时间的。

```c++
Startable::WakeUpType Startable::sleep(UInt32 timeout) {
    if (_stop)
        return STOP;
     WakeUpType result = WAKEUP;
     if (timeout>0) {
         if (!_wakeUpEvent.tryWait(timeout))
             result = TIMEOUT;
     } else {
         _wakeUpEvent.wait();
     }
    if (_stop)
        return STOP;
    return result;
}
```

你可以自行修改 `RTMFPServer` 中 `sleep(...)` 的参数，这样就会调用 `_wakeUpEvent.tryWait(timeout)` 了，按照指定的等待时间（即 `timeout`）来进行睡眠。

`RTMFPManager` 的作用是什么呢？核心就在于它的 `handle` 成员函数：

```c++
void handle() {
    _server.manage();
}
```

这里就会调用到 `RTMFPServer::manage()`，所以你要在阅读 `RTMFPServer` 源码时知道 `RTMFPServer::manage()` 函数并不是在 `RTMFPServer` 线程内运行的，而是 `RTMFPManager` 线程内运行的。它的定义如下：

```c++
void RTMFPServer::manage() {
    _handshake.manage();
    _sessions.manage();
}
```

它实现对现有 Session 的一些管理，比如终止已经死掉的 `Session`。