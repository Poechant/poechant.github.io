---
layout: post
title:  OpenRTMFP/Cumulus 原理、源码及实践 2：CumulusServer 源码启动流程分析
date:   2012-04-14 19:20:46 +0800
categories: rt_tech
tags: [直播技术]
description: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的第二篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。本文对 CumulusServer 的启动流程进行了详细的源码解读，其中还包括 CumulusServer 如何处理命令行的各个输入选项、各项命令、如何 dump logs、载入配置、处理日志。
excerpt: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的第二篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。本文对 CumulusServer 的启动流程进行了详细的源码解读，其中还包括 CumulusServer 如何处理命令行的各个输入选项、各项命令、如何 dump logs、载入配置、处理日志。
---

**本文目录**
* TOC
{:toc}

本文对 CumulusServer 的启动流程进行了详细的源码解读，其中还包括 CumulusServer 如何处理命令行的各个输入选项、各项命令、如何 dump logs、载入配置、处理日志。首先要知道的是，OpenRTMFP/Cumulus 中使用到的库有 Poco、OpenSSL 和 Lua。

### 一、Cumulus 启动源码分析

#### 1、`main.cpp` 中的 `main()` 函数

入口在 `main.cpp` 中：

```c++
int main(int argc, char* argv[]) {
```

先检查内存泄露，不过目前这个开发中的项目还没有实现这个功能，只是个空函数：

```c++
    DetectMemoryLeak();
```

然后会创建一个匿名的 `CumulusServer` 对象，并调用其 `run()` 函数，该函数由 `CumulusServer` 从 `ServerApplication` 中继承而来，而 `ServerApplication` 由是从 `Application` 继承而来的。`CumulusServer` 对象调用 `run()` 函数，实际是 `ServerApplication` 的 `run()` 函数，`ServerApplication` 的 `run()` 函数则是调用 `Application` 的函数，而该 `run()` 函数则是先调用 `initialize()` 函数，然后调用 `main()` 函数，然后调用 `uninitialize()` 函数。如果 `initialize()` 函数被调用时抛出异常，则不会执行 `main()` 函数，但仍然会执行 `uninitialize()` 函数：

```c++
    // Runs the application by performing additional initializations
    // and calling the main() method.
    return CumulusServer().run(argc, argv);
}
```

#### 2、`main.cpp` 中的 `CumulusServer()` 构造函数

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

调用父函数 `ServerApplication` 的初始化函数：

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

#### 4、`main.cpp` 中的 `CumulusServer` 的 `main()` 成员函数

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

#### 5、`CumulusServer` 是 `ServerApplication` 的子类

`ServerApplication` 对其子类有如下要求：

* Subsystems must be registered in the constructor.
* All non-trivial initializations must be made in the `initialize()`` method.
* At the end of the `main()` method, `waitForTerminationRequest()` should be called.

#### 6、`ServerApplication` 是 `Application` 的子类

Application 对其子类的要求是，如下这些成员函数必须被覆盖：

* `initialize()` (the one-argument, protected variant)：上一篇已介绍过。
* `uninitialize()`：下面会介绍，Application 的 `run()` 函数会在调用 `main()` 函数后调用 `uninitialize()` 函数。
* `reinitialize()`
* `defineOptions()`：定义命令行启动选项。
* `handleOption()`：响应相应的命令行选项。
* `main()`

#### 7、反初始化

`CumulusServer` 是继承 `ServerApplication` 的，`ServerApplication` 是继承 `Application` 的。`Application` 的 `run()` 函数会先调用 `initialize()`，然后调用 `main()`，最后调用 `uninitialize`。最后这个反初始化过程，在 `CumulusServer` 就是直接调用父类的 `uninitialize` 函数。

```c++
void uninitialize() {
    ServerApplication::uninitialize();
}
```

### 二、`CumulusServer` 各项交互功能的源码解读

#### 1、命令行选项设定

`CumulusServer` 的命令行选项有：`log（l）`、`dump（d）`、`cirrus（c）`、`middle（m）`、`help（h）`。

```c++
void defineOptions(OptionSet& options) {
    ServerApplication::defineOptions(options);
```

设定日志级别（0 - 8，默认是 6，表示 info 级别）。

```c++
    options.addOption(
        Option("log", "l", "Log level argument, must be beetween 0 and 8 : \
            nothing, fatal, critic, error, warn, note, info, debug, trace. \
            Default value is 6 (info), all logs until info level are displayed.")
                .required(false)
                .argument("level")
                .repeatable(false));
```

其他一些选项：

```c++
    options.addOption(
        Option("dump", "d", "Enables packet traces in logs. Optional arguments \
            are 'middle' or 'all' respectively to displays just middle packet \
            process or all packet process. If no argument is given, just outside \
            packet process will be dumped.",false,"middle|all",false)
                .repeatable(false));
 ```

 ```c++
    options.addOption(
        Option("cirrus", "c", "Cirrus address to activate a 'man-in-the-middle' \
            developer mode in bypassing flash packets to the official cirrus \
            server of your choice, it's a instable mode to help Cumulus developers, \
            \"p2p.rtmfp.net:10000\" for example. By adding the 'dump' argument, \
            you will able to display Cirrus/Flash packet exchange in your logs \
            (see 'dump' argument).",false,"address",true)
                .repeatable(false));
 ```

 ```c++
    options.addOption(
        Option("middle", "m","Enables a 'man-in-the-middle' developer mode \
            between two peers. It's a instable mode to help Cumulus developers. \
            By adding the 'dump' argument, you will able to display Flash/Flash \
            packet exchange in your logs (see 'dump' argument).")
                .repeatable(false));
```

显示帮助信息的选项：

```c++
    options.addOption(
        Option("help", "h", "Displays help information about command-line usage.")
            .required(false)
            .repeatable(false));
}
```

`OptionSet` 是 `Poco::Util::OptionSet`，调用addOption可以向其中增加选项Option。其中required和repeatable表示：

Sets whether the option is required (flag == true) or optional (flag == false).
Returns true if the option can be specified more than once, or false if at most once.
当需要显示帮助信息时，调用如下函数：

```c++
void displayHelp() {
    HelpFormatter helpFormatter(options());
    helpFormatter.setCommand(commandName());
    helpFormatter.setUsage("OPTIONS");
    helpFormatter.setHeader("CumulusServer, open source RTMFP server");
    helpFormatter.format(cout);
}
```

* `setCommand()`: Sets the command name.
* `setUsage()`: Sets the usage string.
* `setHeader()`: Sets the header string.
* `format()`: Writes the formatted help text to the given stream.

#### 2、处理命令行选项

参数是选项名和选项值。

```c++
void handleOption(const std::string& name, const std::string& value) {
    ServerApplication::handleOption(name, value);
```

如果选项是帮助：

```c++
    if (name == "help")
        _helpRequested = true;
```

如果是cirrus，即该服务的 IP 和端口号，Poco::URI 中有协议名（Scheme）、IP 地址（Host）、端口号（Port）、查询串（Query）等等。

```c++
    else if (name == "cirrus") {
        try {
            URI uri("rtmfp://"+value);
            _pCirrus = new SocketAddress(uri.getHost(),uri.getPort());
            NOTE("Mode 'man in the middle' : the exchange will bypass to '%s'",value.c_str());
        } catch(Exception& ex) {
            ERROR("Mode 'man in the middle' error : %s",ex.message().c_str());
        }
```

如果选项是dump日志：

```c++
    } else if (name == "dump") {
        if(value == "all")
            Logs::SetDump(Logs::ALL);
        else if(value == "middle")
            Logs::SetDump(Logs::MIDDLE);
        else
            Logs::SetDump(Logs::EXTERNAL);
```

如果选项是middle：

```c++
    } else if (name == "middle")
        _middle = true;
```

如果选项是log，表示设定日志级别：

```c++
    else if (name == "log")
        Logs::SetLevel(atoi(value.c_str()));
}
```

#### 6、Dump logs

先加一个作用域锁，然后再向日志流写数据。

```c++
void dumpHandler(const UInt8* data,UInt32 size) {
    ScopedLock<FastMutex> lock(_logMutex);
    cout.write((const char*)data, size);
    _logStream.write((const char*)data,size);
    manageLogFile();
}
```

调用 manageLogFile，主要做一些日志大小超出限制的处理。

```c++
void manageLogFile() {
```

先判断是否超过日志文件的大小上线，LOG_SIZE是1000000字节（即约 1 MB）。

```c++
    if (_pLogFile->getSize() > LOG_SIZE) {
        _logStream.close();
        int num = 10;
```

打开新日志文件：

```c++
        File file(_logPath + "10");

```

如果该文件已经存在，则先删除：

```c++
        if (file.exists())
            file.remove();

        while (--num >= 0) {
            file = _logPath + NumberFormatter::format(num);
            if (file.exists())
                file.renameTo(_logPath + NumberFormatter::format(num + 1));
        }
        _logStream.open(_pLogFile->path(), ios::in | ios::ate);
    }   
}
```

#### 3、停止运行

`CumulusServer` 继承了 `ApplicationKiller`，该类中有纯虚函数 `kill()` 需要被实现，于是有：

```c++
void kill() {
    terminate();
}
```

`ApplicationKiller` 的定义在 `ApplicationKiller.h` 中，如下：


```c++
class ApplicationKiller {
public:
    ApplicationKiller(){}
    virtual ~ApplicationKiller(){}
 
    virtual void kill()=0;
};
```

#### 4、载入配置

在initialize()函数中调用，上一篇已提到过。

```c++
void loadConfiguration(const string& path) {
    try {
        ServerApplication::loadConfiguration(path);
    } catch(...) {
    }
}
```

#### 5、处理日志

```c++
void logHandler(Thread::TID threadId,
                const std::string& threadName,
                Priority priority,
                const char *filePath,
                long line, 
                const char *text) {
```

作用域锁：

```c++
    ScopedLock<FastMutex> lock(_logMutex);
 
    Path path(filePath);
    string file;
    if (path.getExtension() == "lua")
        file += path.directory(path.depth()-1) + "/";
```

如果是命令行交互模式（即不是 daemon 模式）：

```c++
    if (_isInteractive)
        printf("%s  %s[%ld] %s\n",
            g_logPriorities[priority - 1],
            (file + path.getBaseName()).c_str(),
            line,
            text);
```

向日志流输出一句日志：

```c++
    _logStream << DateTimeFormatter::format(LocalDateTime(),"%d/%m %H:%M:%S.%c  ")
            << g_logPriorities[priority-1] 
            << '\t' << threadName 
            << '(' << threadId << ")\t"
            << (file + path.getFileName()) 
            << '[' << line << "]  " 
            << text << std::endl;
 
    _logStream.flush();
```

日志文件的善后处理（主要处理文件大小限制可能产生的问题）：

```c++
    manageLogFile();
}
```

### 三、`main.cpp` 的 `main()` 函数源码分析

#### 1、`main.cpp` 中的 `main()` 函数中的 `server`

`main.cpp` 中真正启动的是 `server`，它继承自 `Cumulus::RTMFPServer`，而 `Cumulus::RTMFPServer` 又继承自 `Cumulus::Startable`、`Cumulus::Gateway`、`Cumulus::Handler`。而 `Cumulus::Startable` 继承自 `Poco::Runnable`，所以其是一个可以运行的线程。在 `OpenRTMFP/CumulusServer` 中，这是主线程。

```c++
Server server(config().getString("application.dir", "./"), *this, config());
server.start(params);
```

这是 `CumulusServer/Server.h` 中定义的，其构造函数的原型为：

```c++
Server(const std::string& root,
       ApplicationKiller& applicationKiller,
       const Poco::Util::AbstractConfiguration& configurations);
```

个参数含义如下：

> The Path Root for the Server Application Killer for Termanting the Server Application Server Configuration

距离来说，在我的 Worksapce 中：

`root` 是 `/Users/michael/Development/workspace/eclipse/OpenRTMFP-Cumulus/Debug/` 构造函数的初始化列表极长：

```c++
Server::Server(const std::string& root,
               ApplicationKiller& applicationKiller,
               const Util::AbstractConfiguration& configurations) 
    : _blacklist(root + "blacklist", *this),
      _applicationKiller(applicationKiller),
      _hasOnRealTime(true),
      _pService(NULL),
      luaMail(_pState=Script::CreateState(),
              configurations.getString("smtp.host","localhost"),
              configurations.getInt("smtp.port",SMTPSession::SMTP_PORT),
              configurations.getInt("smtp.timeout",60)) {
```

下面调用 `Poco::File` 创建目录：

```c++
    File((string&)WWWPath = root + "www").createDirectory();
```

因为 `roor` 是 `/Users/michael/Development/workspace/eclipse/OpenRTMFP-Cumulus/Debug/` 目录，所以 `WWWPath` 就是 `/Users/michael/Development/workspace/eclipse/OpenRTMFP-Cumulus/Debug/www` 目录。然后初始化 `GlobalTable`，这个 `GlobalTable` 是和 Lua 有关的东东，这里暂不细说，先知道与 Lua 相关就好。

```c++
    Service::InitGlobalTable(_pState);
```

下面就涉及到了 Lua script 了：

```c++
    SCRIPT_BEGIN(_pState)
        SCRIPT_CREATE_PERSISTENT_OBJECT(Invoker,LUAInvoker,*this)
        readNextConfig(_pState,configurations,"");
        lua_setglobal(_pState,"cumulus.configs");
    SCRIPT_END
}
```

其中 `SCRIPT_BEGIN`、`SCRIPT_CREATE_PERSISTENT_OBJECT和SCRIPT_END` 都是宏，其定义在 `Script.h` 文件中，如下：

```
#define SCRIPT_BEGIN(STATE) \
    if (lua_State* __pState = STATE) { \
        const char* __error=NULL;
 
#define SCRIPT_CREATE_PERSISTENT_OBJECT(TYPE,LUATYPE,OBJ) \
    Script::WritePersistentObject<TYPE,LUATYPE>(__pState,OBJ); \
    lua_pop(__pState,1);
 
#define SCRIPT_END }
```

`SCRIPT_BEGIN和SCRIPT_END` 经常用到，当与 Lua 相关的操作出现时，都会以这两个宏作为开头和结尾。

#### 2、`main.cpp` 中 `main()` 函数的 `server.start()`

```c++
void RTMFPServer::start(RTMFPServerParams& params) {
```

如果 `OpenRTMFP/CumulusServer` 正在运行，则返回并终止启动。

```c++
    if(running()) {
        ERROR("RTMFPServer server is yet running, call stop method before");
        return;
    }
```

设定端口号，如果端口号为 0，则返回并终止启动。

```c++
    _port = params.port;
    if (_port == 0) {
        ERROR("RTMFPServer port must have a positive value");
        return;
    }
```

设定 `OpenRTMFP/CumulusEdge` 的端口号，如果其端口号与 `OpenRTMFP/CumulusSever` 端口号相同，则返回并终止启动：

```c++
    _edgesPort = params.edgesPort;
    if(_port == _edgesPort) {
        ERROR("RTMFPServer port must different than RTMFPServer edges.port");
        return;
    }
```

Cirrus：

```c++
    _freqManage = 2000000; // 2 sec by default
    if(params.pCirrus) {
        _pCirrus = new Target(*params.pCirrus);
        _freqManage = 0; // no waiting, direct process in the middle case!
        NOTE("RTMFPServer started in man-in-the-middle mode with server %s \
             (unstable debug mode)", _pCirrus->address.toString().c_str());
    }
```

middle：

```c++
    _middle = params.middle;
    if(_middle)
        NOTE("RTMFPServer started in man-in-the-middle mode between peers \
              (unstable debug mode)");
```

UDP Buffer：

```c++
    (UInt32&)udpBufferSize = 
        params.udpBufferSize==0 ? 
            _socket.getReceiveBufferSize() : params.udpBufferSize;
 
    _socket.setReceiveBufferSize(udpBufferSize);
    _socket.setSendBufferSize(udpBufferSize);
    _edgesSocket.setReceiveBufferSize(udpBufferSize);
    _edgesSocket.setSendBufferSize(udpBufferSize);
 
    DEBUG("Socket buffer receving/sending size = %u/%u",
        udpBufferSize,
        udpBufferSize);
 
    (UInt32&)keepAliveServer = 
        params.keepAliveServer < 5 ? 5000 : params.keepAliveServer * 1000;
    (UInt32&)keepAlivePeer = 
        params.keepAlivePeer < 5 ? 5000 : params.keepAlivePeer * 1000;
    (UInt8&)edgesAttemptsBeforeFallback = params.edgesAttemptsBeforeFallback;
 
    setPriority(params.threadPriority);
```

启动线程，进入循环运行：

```c++
    Startable::start();
}
```

上句具体的源码实现为：

```c++
void Startable::start() {
    if (running())
        return;
```

如果在运行则返回并终止启动。然后加一个局部锁。

```c++
    ScopedLock<FastMutex> lock(_mutex);
```

如果不得不join()到主线程中，那就join()吧

```c++
    if(_haveToJoin) {
        _thread.join();
        _haveToJoin=false;
    }
```

然后就运行这个线程吧：

```c++
    _terminate = false;
    _thread.start(*this);
    _haveToJoin = true;
}
```

#### 3、回顾一下整个启动流程

到此我们先回顾一下启动过程：

从 `main.cpp` 的启动入口 `main()` 函数开始，创建 `Server` 对象并启动（调用 `start()` 函数）。`Server::start()` 中调用其父类（`RTMFPServer`）的父类（`Startable`）的方法 `Startable::start()` 开启线程。
调用 `Startable::start()` 函数后，开启线城时传入的参数为 `*this`，所以就会运行 `Startable::run()`；

#### 4、`RTMFPServer::prerun()`、`Startable::prerun()` 和 `RTMFPServer::run(...)` 函数源码

`Startable::run()` 调用 `Startable::prerun()` 函数，但这个函数被 `RTMFPServer` 覆盖，所以会运行 `RTMFPServer::prerun()`，其源码如下：

```c++
bool RTMFPServer::prerun() {
    NOTE("RTMFP server starts on %u port",_port);
```

如果CumulusEdge：

```c++
    if (_edgesPort>0)
        NOTE("RTMFP edges server starts on %u port",_edgesPort);
 
    bool result = true;
    try {
        onStart();
```

运行线程：

```c++
        result = Startable::prerun();
```

处理异常：

```c++
    } catch(Exception& ex) {
        FATAL("RTMFPServer : %s",ex.displayText().c_str());
    } catch (exception& ex) {
        FATAL("RTMFPServer : %s",ex.what());
    } catch (...) {
        FATAL("RTMFPServer unknown error");
    }
```

如果跳出了，则终止运行：

```c++
    onStop();
 
    NOTE("RTMFP server stops");
    return result;
}
```

该函数内部又会调用父类的 `Startable::prerun()` 函数，该函数调用：

```c++
virtual void Startable::run(const volatile bool& terminate) = 0;
```

它是一个纯虚函数，由 `RTMFPServer` 实现。

`Startable::prerun()` 会调用 `void run(const volatile bool& terminate)` 方法，该方法被 `RTMFPServer` 覆盖了。

```c++
bool Startable::prerun() {
    run(_terminate);
    return !_terminate;
}
```

`RTMFPServer` 覆盖 `Startable` 的 `run(const volatile bool &terminate)` 方法。

```c++
void RTMFPServer::run(const volatile bool& terminate) {
    ...
}
```