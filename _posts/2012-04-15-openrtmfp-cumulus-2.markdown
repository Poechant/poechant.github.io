---
layout: post
title:  OpenRTMFP/Cumulus Primer-2：CumulusServer 启动分析与主进程主循环分析
date:   2012-04-15 22:26:58 +0800
categories: rt_tech
tags: [直播技术]
---

<!-- * TOC
{:toc} -->

首先要知道的是，OpenRTMFP/Cumulus 中使用到的库有 Poco、OpenSSL 和 Lua。

### Cumulus 启动分析

#### 1、`main.cpp` 中的`main()`函数

入口在`main.cpp`中：

```c++
int main(int argc, char* argv[]) {
```

先检查内存泄露，不过目前这个开发中的项目还没有实现这个功能，只是个空函数：

```c++
    DetectMemoryLeak();
```

然后会创建一个匿名的 `CumulusServer` 对象，并调用其`run()`函数，该函数由 `CumulusServer` 从 `ServerApplication` 中继承而来，而 `ServerApplication` 由是从 `Application` 继承而来的。`CumulusServer` 对象调用 `run()` 函数，实际是 `ServerApplication` 的 `run()` 函数，`ServerApplication` 的 `run()` 函数则是调用 `Application` 的函数，而该 `run()` 函数则是先调用 `initialize()` 函数，然后调用 main() 函数，然后调用 `uninitialize()` 函数。如果 `initialize()` 函数被调用时抛出异常，则不会执行 `main()` 函数，但仍然会执行 `uninitialize()` 函数：

```c++
    // Runs the application by performing additional initializations
    // and calling the main() method.
    return CumulusServer().run(argc, argv);
}
```

#### 2、`main.cpp`中的`CumulusServer()`构造函数

`CumulusServer` 的构造函数定义为：

```c++
CumulusServer(): _helpRequested(false), // 显示帮助信息
                 _pCirrus(NULL),
                 _middle(false),
                 _isInteractive(true),
                 _pLogFile(NULL) {
}
```

#### 3、`main.cpp` 中的 `CumulusServer` 的 `initialize()` 成员函数

在执行 `main()` 函数之前，`CumulusServer` 会启动 `initialize()` 函数，传入的参数就是 `CumulusServer` 自己，可以猜到 `Poco::Util::Application` 的 `run` 方法中，调用该函数时的参数是 `this`。

```c++
void initialize(Application& self) {
```

调用父函数`ServerApplication`的初始化函数：

```c++
ServerApplication::initialize(self);
```

再继续下面的源码分析之前，先要知道，根据 `Poco::Util::Application` ，`Application` 类有一些内置的配置属性，如下：

* `application.path`: 可执行文件的绝对路径；
* `application.name`: 可执行文件的文件名（含扩展名）；
* `application.baseName`: 可执行文件的文件名（不含扩展名）
* `application.dir`: 可执行文件的所在目录；
* `application.configDir`: 配置文件所在目录；

所以下面就读取了可执行文件的所在目录，其中第二个参数表示默认值（即当前目录）：

```c++
    string dir = config().getString("application.dir", "./");
```

然后读取配置文件，目录为上一句所得到的 `dir`，文件名（不含扩展名）为 `application.basename` 内置配置属性值，其默认值为 `CumulusServer`，然后加上点和扩展名 `.ini`。

```c++
    loadConfiguration(dir
        + config().getString("application.baseName", "CumulusServer")
        + ".ini");
```

这样就加载完配置了。然后查看当前的进程是从命令行运行的（命令行是交互的，所以是 interactive），还是以 daemon 方式运行的，这个函数是ServerApplication的一个成员函数：

```c++
    _isInteractive = isInteractive();
```

然后获取表示日志文件所在目录的字符串，其中 `logs.directory` 是外置配置属性（配置文件中），其默认值为上面获取到的可执行文件路径（一般为当前路径）与 `logs` 的组合，即一般为当前目录下的 `logs` 目录：

```c++
    string logDir(config().getString("logs.directory", dir + "logs"));
```

创建日志文件目录：

```c++
    File(logDir).createDirectory();

```

日志文件绝对路径，`logs` 为默认的日志文件名（不含扩展名的部分），扩展名用0：

```c++
    _logPath = logDir + "/" + config().getString("logs.name", "log") + ".";
    _pLogFile = new File(_logPath + "0");
```

用日志流打开日志文件（方式为追加写入）：

```c++
    _logStream.open(_pLogFile->path(), ios::in | ios::ate);
```

`Logs` 是一个方法类（其中的 `public` 函数都是静态的），`SetLogger` 的作用就是将Logs中的似有静态成员设置为某个 `Cumulus::Logger` 对象（由于 `CumulusServer` 继承了 `Cumulus::Logger`）。

```c++
    Logs::SetLogger(*this);
}
```

4、`main.cpp` 中的 `CumulusServer` 的 `main()` 成员函数

OpenRTMFP/Cumulus 是一个基于 `Poco::Util::Application` 的服务端应用（准确的说是基于 `Poco::Util::ServerApplication` 的服务端应用）。如果没有特殊的启动要求，可以调用 `Poco/Application.h` 中定义的宏 `POCO_APP_MAIN` 来完成初始化、日志和启动（该宏会根据不同的平台启用不同的 `main()` 函数）。

`run()` 函数在调用完 `initialize()` 函数后，会调用 `CumulusServer` 中的 `main()` 函数，该 `main()` 函数的定义在 `main.cpp` 中：

```c++
int main(const std::vector<std::string>& args) {
```

首先看是否是要求帮助信息，`displayHelp` 是借助 `Poco::Util::HelpFormatter` 实现的，`CumulusServer` 的构造函数会在调用时将 `_helpRequested` 设置为 `false`。

```c++
    if (_helpRequested) {
        displayHelp();
    }
```

如果不是，则进入启动状态，首先创建一个 `RTMFPServerParams` 对象 `params`，用来存储 OpenRTMFP/CumulusServer 的基本配置信息。

```c++
    else {
        try {
            RTMFPServerParams params;
```

`params` 存储 `CumulusServer` 的端口号和 `CumulusEdge` 的端口号：

```c++
            params.port = config().getInt("port", params.port);
            UInt16 edgesPort = config().getInt("edges.port",
                RTMFP_DEFAULT_PORT+1);
            if(config().getBool("edges.activated",false)) {
                if(edgesPort==0)
                    WARN("edges.port must have a positive value if \
                          edges.activated is true. Server edges is \
                          disactivated.");
                params.edgesPort=edgesPort;
            }
```

`_pCirrus` 为 `SocketAddress` 的成员，是封装IP地址和端口号的对象。

```c++
            params.pCirrus = _pCirrus;
            params.middle = _middle;
```

UDB 所使用的缓冲区大小：

```c++
            params.udpBufferSize = config().getInt("udpBufferSize",
                params.udpBufferSize);
```

```c++
            params.keepAliveServer = config().getInt(
                "keepAliveServer",params.keepAliveServer);
```

```c++
            params.keepAlivePeer = config().getInt("keepAlivePeer",
                params.keepAlivePeer);
```

失败前 CumulusEdge 的尝试次数：

```c++
            params.edgesAttemptsBeforeFallback = config().getInt(
                "edges.attemptsBeforeFallback",
                params.edgesAttemptsBeforeFallback);
```

```c++
            Server server(config().getString("application.dir","./"),
                *this,config());
```

```c++
            server.start(params);
```

`waitForTerminationRequest()` 函数是 `main()` 函数中必须调用的，意为等待终止运行的操作请求。

```c++
            // wait for CTRL-C or kill
            waitForTerminationRequest();
```

一旦接收到终止操作的请求，就会执行下面这句，用以退出 OpenRTMFP/Cumulus 的运行：

```c++
            // Stop the server
            server.stop();
```

`catch` 一些可能产生的异常：

```c++
        } catch(Exception& ex) {
            FATAL("Configuration problem : %s",ex.displayText().c_str());
        } catch (exception& ex) {
            FATAL("CumulusServer : %s",ex.what());
        } catch (...) {
            FATAL("CumulusServer unknown error");
        }
    }
```

OpenRTMFP/CumulusServer 停止运行：

```c++
    return Application::EXIT_OK;
}
```