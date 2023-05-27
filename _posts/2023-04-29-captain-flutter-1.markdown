---
layout: post
title: 麦克船长的全栈指南：Google Flutter 开发跨平台 APP
date:   2023-04-29 16:10:13 +0800
categories: web
tags: [前端, 开发, web, app, 客户端, 前端, android, ios, html5, 小程序, 应用]
description: 在一个人就是一支队伍的时代里（在 AI 急速发展的推动下，未来会越来越趋于《未来简史》所描述的「半神」），产品、运营还是工程，最好都要会一些前端，这样方便自己做一点东西便于提高日常的生产力。今天介绍的是 2018 年开始的 Google Flutter 跨多平台终端的技术，支持 iOS、Android、Web、macOS、Linux、Windows ……
excerpt: 在一个人就是一支队伍的时代里（在 AI 急速发展的推动下，未来会越来越趋于《未来简史》所描述的「半神」），产品、运营还是工程，最好都要会一些前端，这样方便自己做一点东西便于提高日常的生产力。今天介绍的是 2018 年开始的 Google Flutter 跨多平台终端的技术，支持 iOS、Android、Web、macOS、Linux、Windows ……
katex: True
location: 杭州
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

## 第一章 · 准备与 First Step

### 第 1 节 · 环境配置与 Hello World

Flutter 是 Google 推出的一套跨平台的终端开发框架，有中文社区及官方网站支持 `https://flutter.cn/`。

![](/img/src/2023/05/2023-05-01-captain-flutter-1.png)

Flutter 是为跨端而生，那么就要和船长一起了解下跨端这件事儿的演进历史。最早在万维网（World Wide Web，简称 WWW）出现后，第一个端就是 Web，其基础编程语言是 HTML。但是 HTML 不是一个图灵完备的语言，只是一个标记语言，就像它的名字一样（Hypter Text Mark Language）。为了展现不同的样式，配套出现了 CSS 用于控制 Web 的展现样式，其含义是 Cascading Style Sheets（层叠样式表）。而让 Web 活起来，不是死气沉沉的一对按钮、文字、段落 …… 则需要一款图灵完备的语言，可以随意控制 Web 上显示的各种组建，以及控制状态、数据，还有掌控逻辑流程等等，就配套地出现了 JavaScript。因此第一个阶段我们可以称其为 HTML 时代，始于 1993 年，其编程语言是 HTML+CSS+JS。这个阶段还只有桌面端，主要就是适配浏览器。

第二阶段，是从移动互联网开始，大概在 2009 年前后，Android/iOS 平台出现，最初都是每个平台要维护一套代码的。但是很快，业界就在推动从原生开发到逐渐过渡到 Hybrid 混合模式（半原生半 Web），由原生提供一套统一 API 供 JavaScript 调用。因此核心的代码逻辑都是在 HTML+JS 中实现的，然后展示在 WebView 中（你可以理解为浏览器，iOS 和 Android 中都有 WebView）。这时候已经从为 iOS、Android 写多套代码过渡成一套代码维护了，虽然实现了跨平台，但是整体性能比较差。

第三阶段，Facebook 推出 React Native，理念是「Learn Once, Write Anywhere」，用 JavaScript 语言、React 的设计模式、原生的界面渲染/动画/网络请求。开发者编写的 JS 代码通过 React Native 中间层转换为各平台原生代码，实现原生控件和操作。

第四阶段，阿里巴巴在 2016 年推出 Weex，理念是「Write One, Run Anywhere」，也是用 JavaScript 语言，基于 Vue 的设计模式支持 Web、Android、iOS。

第五阶段，Google 在 2018 年推出 Flutter，支持 AOT 编译（Ahead Of Time，即运行前编译）从而实现机器码，也支持 JIT 编译（Just In Time，即时编译）热重载，用 Dart 语言开发。目前国内大厂基本都是用了 Flutter，包括阿里巴巴、腾讯、百度、美团、字节跳动、京东等。

好了就说这么多，首先下载 Flutter `https://flutter.cn/docs/get-started/install`，然后：

```bash
mikecaptain@CVN % unzip ~/Downloads/flutter_macos_3.7.12-stable.zip
mikecaptain@CVN % mv ~/Downloads/flutter_macos_3.7.12-stable ~/workspace/flutter
mikecaptain@CVN % cd ~/workspace/flutter
mikecaptain@CVN % export PATH="$PATH:`pwd`/flutter/bin"
```

一般前端开发者都安装过 XCode、Android Studio、Chrome 等，所以配置 Flutter 也都会比较顺利，这里船长就不赘述了，安装成功后用 `flutter doctor` 测试应该有如下提示：

```bash
(base) mikecaptain@CVN ~ % flutter doctor                   
Doctor summary (to see all details, run flutter doctor -v):
[✓] Flutter (Channel stable, 3.7.12, on macOS 13.2.1 22D68 darwin-arm64, locale zh-Hans-CN)
[✓] Android toolchain - develop for Android devices (Android SDK version 33.0.2)
[✓] Xcode - develop for iOS and macOS (Xcode 14.3)
[✓] Chrome - develop for the web
[✓] Android Studio (version 2022.1)
[✓] Connected device (2 available)
[✓] HTTP Host Availability

• No issues found!
```

详细的安装指南请看 Flutter 官网，例如 macOS 上的安装 `https://flutter.cn/docs/get-started/install/macos`。

### 第 2 节 · First Step：在 macOS 平台跑第一个 App

```bash
mikecaptain@CVN % flutter create hello_world
mikecaptain@CVN % cd hello_world
mikecaptain@CVN % flutter run
```

正常的话这样会看到运行成功的提示：

![](/img/src/2023/05/2023-05-01-captain-flutter-2.png)

如果选择 `[1]: macOS(macos)`，会弹出一个 macOS 的应用程序窗口：

![](/img/src/2023/05/2023-05-01-captain-flutter-3.png){: width="720"}

### 第 3 节 · 在 iOS 平台（设备及模拟器）上跑起第一个 App

将你的 iPhone 连接到 macOS 设备上，在 iPhone 的设置 - 隐私与安全性 - 开发者模式 - 打开 - 重启 iPhone - 重启后输入密码进入设备 - 在弹窗上同意开启开发者模式。此时在 XCode - Windows - Devices and Simulators 中选择你的 iPhone 后会看到如下类似页面：

![](/img/src/2023/05/2023-05-01-captain-flutter-4.png){: width="720"}

设置结束后，会看到如下窗口页面：

![](/img/src/2023/05/2023-05-01-captain-flutter-5.png){: width="720"}

在 Xcode 中打开该 Flutter 项目命令如下：

```bash
mikecaptain@CVN % open ios/Runner.xcworkspace
```

则会看到如下 XCode 项目窗口：

![](/img/src/2023/05/2023-05-01-captain-flutter-6.png){: width="720"}

如果有类似如下提示：

```bash
.../hello_world/ios/Runner.xcodeproj Signing for "Runner" requires a development team. Select a development team in the Signing & Capabilities editor.
```

这个提示是因为你还没有设置项目的开发者团队。要解决这个问题，请按照以下步骤进行操作：

1. 在 Xcode 中打开 Runner.xcworkspace 文件。
2. 在左侧导航栏中选择 Runner。
3. 在中间面板中选择 "Signing & Capabilities"。
4. 在 "Signing & Capabilities" 下的 "Team" 中选择你的开发者团队。

如果提示如下错误：

```bash
Failed to register bundle identifier
The app identifier "com.example.helloWorld" cannot be registered to your development team because it is not available. Change your bundle identifier to a unique string to try again.
```

这个错误信息是因为你的 Bundle Identifier 在开发者中心已经被其他开发者使用了。你需要在你的项目中使用一个唯一的 Bundle Identifier。你可以按照以下步骤修改项目的 Bundle Identifier：

1. 在 Xcode 中打开你的项目。
2. 选择你的项目在左侧导航栏中的顶层项。
3. 在 "General" 选项卡中，找到 "Identity" 部分。
4. 在 "Bundle Identifier" 中输入一个唯一的标识符，例如 "com.yourcompany.yourapp"。
5. 保存并重新编译你的项目。

正常到这里应该就没有什么错误了，可以运行一下程序，可能会有如下提示：

![](/img/src/2023/05/2023-05-01-captain-flutter-7.jpg){: width="240"}

这个问题通常是因为你的开发者账号没有被信任。解决这个问题的步骤如下：

1. 在你的 iPhone 上打开 "设置" 应用程序。
2. 滚动到 "通用" > "设备管理"。
3. 找到你的开发者账号并点击它。
4. 点击 "信任"。
5. 重启你的 iPhone，然后再次尝试运行你的应用程序。

![](/img/src/2023/05/2023-05-01-captain-flutter-8.png){: width="320"}

再次运行就成功了。

![](/img/src/2023/05/2023-05-01-captain-flutter-9.png){: width="320"}

同样尝试一下在 Simulator 上运行试试：

![](/img/src/2023/05/2023-05-01-captain-flutter-10.png){: width="360"}

### 第 4 节 · 在 Android 平台跑起第一个 App

在 Android 的 Tools - Device Manager - 右侧「Create device」，比如创建一个「Pixel 6」设备，点击「Next」。

![](/img/src/2023/05/2023-05-01-captain-flutter-14.png)

### 第 5 节 · 在 Chrome 浏览器跑起第一个 App

更新最新版本的 Flutter：

```bash
mikecaptain@CVN % flutter channel stable
mikecaptain@CVN % flutter upgrade
```

查看 flutter 目前可以连接的设备：

```bash
(base) mikecaptain@CVN testspace % flutter devices
3 connected devices:

iPhone 14 (mobile) • D89CD348-58C2-4F3B-B07D-54E2A72AB1BA • ios            • com.apple.CoreSimulator.SimRuntime.iOS-16-4 (simulator)
macOS (desktop)    • macos                                • darwin-arm64   • macOS 13.2.1 22D68 darwin-arm64
Chrome (web)       • chrome                               • web-javascript • Google Chrome 112.0.5615.137
```

使用 Chrome 运行刚刚的 Hello World 程序：

```bash
(base) mikecaptain@CVN hello_world % flutter run -d chrome
Launching lib/main.dart on Chrome in debug mode...
Waiting for connection from debug service on Chrome...                 ⣾
```

浏览器会打开 `http://localhost:56049/#/` 链接：

![](/img/src/2023/05/2023-05-01-captain-flutter-11.png)

如果项目在创建时没有支持 Web，可以运行下面命令支持 Web：

```bash
mikecaptain@CVN % flutter create --platforms web .
```

## 第二章 · Flutter 的开发理念与原理

### 第 6 节 · Flutter 的开发框架和线程模型

Flutter 包括三层：框架层 Framework（Dart 编写）、引擎层 Engine（C/C++）、嵌入层 Embedder（Platform Specific）。

再来跟着船长理解下 Flutter 的线程模型。Embedder 进行线程的创建和管理，提供四种 Task Runner 给 Engine，包括 Platform Task Runner（主线程）、UI Task Runner（渲染和处理 Vsync 信号并将 Widget 转换成 Layer Tree）、GPU Task Runner、IO Task Runner（文件读写与资源加载）。

Engine 启动时，会分别创建一个 UI Task Runner、一个 GPU Task Runner、一个 IO Task Runner。所有 Engine 共享一个 Platform Task Runner。

### 第 7 节 · Flutter 的基础理念：一切皆为 Widgets

* Widget 的主要工作是提供一个 `build()` 方法来描述如何用 Widget 树状结构下自己的叶子 Widget 们如何组织起来显示自己。
* Material 是一种标准的移动端和 web 端的视觉设计语言。 Flutter 提供了一套丰富的 Material widgets。
* 你可以把 Widget 理解为一个配置文件，描述了 UI 界面的展示。Widget 状态变化时会重新绘制 UI，Flutter 框架会对比状态变化前后的区别来重新渲染 UI。
* Flutter 中一般说的生命周期，指的是 StatefulWidget 的生命周期。生命周期涉及到一系列的方法，下面会讲到。

### 第 8 节 · 理解 Flutter 的生命周期

![](/img/src/2023/05/2023-05-01-captain-flutter-15.png){: width="480"}

Flutter 生命周期就是 StatefulWidget 的生命周期，从 `createState` 方法开始。

#### 8.1、初始化阶段

* `createState`：当 StatefulWidget 被调用时会立即执行 `createState` 方法，然后是 `initState`。
* `initState`：一般会在这里进行各个变量的初始赋值，与服务端进行最初的交互。

#### 8.2、组件创建阶段

* `didChangeDependencies`：当 State 发生变化时会调用。
* `build` 主要是返回需要渲染的 Widget。`build` 会被调用多次，所以这里只能返回 **widget 的相关逻辑**，避免因为多次执行导致的状态异常。

#### 8.3、触发组件 build

触发组件 build 的方法有 `didChangeDependencies`、`setState`、`didUpdateWidget`。

* `didUpdateWidget`：在 Widget 重新构建时，Flutter Framework 会调用 Widget.canUpdate 来检测 Widget Tree 中同一位置的新旧节点，然后决定是否需要更新，如果得到的是 True 就会调用这个回调方法 `didUpdateWiedget`。另外父亲 Widget 的 `build` 时，孩子 Widget 的 `didUpdateWidget` 也会被调用。这个方法调用后，一定会再次调用孩子的 `build`。

#### 8.4、组件的销毁阶段

* `deactivate` 在组件被移除节点后会被调用。
* `dispose` 是在一个组件被移除，并没有插入到其他节点时，会被调用，即永久删除，并释放组件资源。

### 第 9 节 · Platform Channel

Flutter 是一个跨平台的 UI 框架，Flutter 本身无法提供各种系统能力，比如蓝牙、相机、GPS 等等。用 Platform Channel 来与底层 OS 平台进行异步的消息通信，在发送给底层 OS 之前会编译成二进制消息，在接收到 Flutter 框架时则会解码成 Dart 可理解的值（所有解码器都只是空消息）。涉及到三种主要 Platform Channel：

* `BasicMessageChannel`：传递数据的通道。
* `MethodChannel`：传递方法调用的通道。
* `EventChannel`：传递事件的通道。

在 Flutter 与 iOS 或 Android 之间，通过调用 invokeMehotd 发起通信，接收方通过 MethodCallHandle 接收。

### 第 10 节 · 包管理

类似于以前在 Java 中会使用 Ant、Maven，在 Android 会使用 Gradle，在 iOS 中使用 Cocoapods，在 Node 中使用 npm（Node Package Management）等等，下面说说 Flutter 中的包管理。Flutter 的项目中用 `pubspec.yaml` 文件来管理依赖关系。类比来看的话：

* Android 的 Gradle 用 `.gradle` 文件管理配置依赖关系。
* iOS 的 Cocoapods 用 `Podfile` 文件管理配置依赖关系。
* Flutter 的 Packages 用 `pubspec.yaml` 文件管理配置依赖关系。

下面是一个典型的、简单的 `pubspec.yaml` 文件：

```yaml
name: hello_world
description: A new Flutter project.
# The following line prevents the package from being accidentally published to
# pub.dev using `flutter pub publish`. This is preferred for private packages.
publish_to: 'none' # Remove this line if you wish to publish to pub.dev

version: 1.0.0+1

environment:
  sdk: '>=2.19.6 <3.0.0'

dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.3
  timeago: ^3.1.0

  # The following adds the Cupertino Icons font to your application.
  # Use with the CupertinoIcons class for iOS style icons.
  cupertino_icons: ^1.0.2
  intl: ^0.18.1

dev_dependencies:
  flutter_test:
    sdk: flutter

  flutter_lints: ^2.0.0

# The following section is specific to Flutter packages.
flutter:

  # The following line ensures that the Material Icons font is
  # included with your application, so that you can use the icons in
  # the material Icons class.
  uses-material-design: true
```

* 首先是项目名字：`name: hello_world`
* 应用的版本号：`version: 1.0.0+1`
* Flutter 环境：`environment:`
  * SDK 支持版本范围：`sdk: '>=2.19.6 <3.0.0'`
* 生产环境所有依赖：`dependencies:`，这里实例一些常用依赖。依赖包都会被编译到最终的包里。
  * HTTP 包：`http: ^0.13.3`
  * 一个用于时间描述表示的包：`timeago: ^3.1.0`
  * CSS 定义颜色包：`css_colors: ^1.1.1`
* 开发环境额外依赖：`dev_dependencies` 是一些为了提高开发效率依赖的包，比如一些自动化测试工具之类的。
* `flutter` 是一些 Flutter 项目的相关配置，比如是否使用 material 设计样式、本地资源配置路径

在 VSCode 的「终端」里输入 `flutter pub get` 即可以把所有依赖安装到项目中，每次更新 `pubspec.yaml` 后都要运行一下 `flutter pub get`。

### 第 11 节 · Flutter 渲染原理

#### 11.1、了解下 Flutter 的三棵树

我们来具体看一段代码：

```dart
Container(
  color: Colors.blue,
  child: Row(
    children: <Widget>[
      Image.asset(
        'image'
      ), // Image.asset
      Text(
        'text'
      ) // Text
    ], // <Widget>[]
  ), // row
); // Container
```

Flutter 视图包含了三棵树，渲染时间就是花费在了这三棵树上：Widget、Element、RenderObject。

Widget Tree 是一个所有 Widget 的数据结构，创建过程是非常轻量的，因为并不涉及到 UI 渲染，只是树结构配置的生成。页面刷新时也随时会重建，Widget Tree 重建本身不涉及任何 UI 渲染。

<br/>

<div style="text-align: center;">
{% graphviz %}
digraph G {
  rankdir=TB

  {rank=same Image Text}
  {rank=same RawImage RichText}

  Container -> DecoratedBox
  DecoratedBox -> Row
  Row -> Image
  Image -> RawImage
  Row -> Text
  Text -> RichText
}
{% endgraphviz %}
</div>

<br/>

根据 Widget 会生成 Element，如下图。Element Tree 同时持有 Widget Tree 和 RenderObject Tree，Element Tree 存放上下文信息，通过 Element Tree 来遍历视图树，支撑 UI 结构。

<br/>

<div style="text-align: center;">
{% graphviz %}
digraph G {
  rankdir=TB

  ComponentElement1[label="ComponentElement"]
  ComponentElement2[label="ComponentElement"]
  RenderObjectElement2[label="RenderObjectElement"]
  RenderObjectElement3[label="RenderObjectElement"]
  RenderObjectElement4[label="RenderObjectElement"]

  {rank=same ComponentElement1 ComponentElement2}
  {rank=same RenderObjectElement3 RenderObjectElement4}

  ComponentElement -> RenderObjectElement1
  RenderObjectElement1 -> RenderObjectElement2
  RenderObjectElement2 -> ComponentElement1
  RenderObjectElement2 -> ComponentElement2
  ComponentElement1 -> RenderObjectElement3
  ComponentElement2 -> RenderObjectElement4
}
{% endgraphviz %}
</div>

<br/>

RenderObject Tree 是真正负责渲染的，根据 Widget 的布局属性，进行布局、绘制。

<br/>

<div style="text-align: center;">
{% graphviz %}
digraph G {
  rankdir=TB

  {rank=same RenderImage RenderParagraph}

  RenderObjectBox -> RenderFlex
  RenderFlex -> RenderImage
  RenderFlex -> RenderParagraph
}
{% endgraphviz %}
</div>

<br/>

整个流程是，根据 Widget Tree 生成 Element Tree，然后创建 RenderObject Tree 并管理到 Element.renderObject 属性上，最后通过 RenderObject 来完成布局、绘制。

#### 11.2、页面刷新时的操作流程

在更新页面时，Framework 向下通知 Engine，Engine 会等到下一个 Vsync 信号到达时向上通知 Framework 进行 animate、build、layout 和 paint，最后生成 layer 向下交给 Engine。Engine 会把 layer 进行组合、生成纹理，最后通过底层的 OpenGL API 提交数据给 GPU，GPU 经过处理后在显示设备上显示，如下图（[来源](https://www.bilibili.com/video/BV11R4y1r7tL)）。

![](/img/src/2023/05/2023-05-01-captain-flutter-19.png){: width="480"}

以 Text 或 Image 发生改变为例，会触发哪些操作呢？这一帧刷新开始：

* 首先 Widget 是不可改变的，所以会创建一颗新的树；
* 开始 `build`；
* 对上一帧的 Element 树进行遍历，调用 Element 的 updateChild 看子节点类型跟之前是不是一致。不一致的话：就扔掉子节点，并造一个新的。一致的话：1）只做内容上的更新（`updateRenderObject`）；2）再判断节点属性是否有改动，如果有则标记为 dirty，重新 layout、paint。
* 再生成新的 layer 交给 GPU。

如下图（[来源](https://www.bilibili.com/video/BV11R4y1r7tL)）。

![](/img/src/2023/05/2023-05-01-captain-flutter-20.png){: width="240"}

## 第三章 · Flutter 组件介绍及应用

### 第 12 节 · 编写一个 News App 练手

#### 12.1、下载 VSCode 并创建 Flutter

我们这里选择用目前最主流的 IDE —— VSCode 来编写程序，首先在 `https://code.visualstudio.com/` 下载 VSCode，解压后就可以使用。

然后通过 `cmd + shift + p` 打开 VSCode 的控制台，输入 `Flutter: New Project` 然后选择 `Application`，就创建成功一个 VSCode 下的 Flutter 项目了。这个过程创建出来的整个目录结构与在 CLI 下输入 `flutter create hello_world` 效果是一样的。目录结构如下：

![](/img/src/2023/05/2023-05-01-captain-flutter-13.png){: width="240"}

所有依赖库都在 `pubspec.yaml` 文件里管理，我们可以为其添加一个 `http` 依赖库：

```bash
dependencies:
  flutter:
    sdk: flutter
  http: ^0.13.3
```

然后运行 `flutter pub get` 命令完成依赖的获取。

```bash
(base) mikecaptain@CVN yongxian_news % flutter pub get            
Running "flutter pub get" in yongxian_news...
Resolving dependencies... 
  async 2.10.0 (2.11.0 available)
  characters 1.2.1 (1.3.0 available)
  collection 1.17.0 (1.17.1 available)
  js 0.6.5 (0.6.7 available)
  matcher 0.12.13 (0.12.15 available)
  material_color_utilities 0.2.0 (0.3.0 available)
  meta 1.8.0 (1.9.1 available)
  path 1.8.2 (1.8.3 available)
  source_span 1.9.1 (1.10.0 available)
  test_api 0.4.16 (0.5.2 available)
Got dependencies!
```

#### 12.2、Flutter 项目文件结构

```
android
ios
linux
macos
web
windows
build
lib
--- main.dart
test
--- widget_test.dart
pubspec.lock
pubspec.yaml
README.md
analysis_options.yaml
hello_world.iml
```

各个平台的生成文件都有独立的目录，android、ios、web、windows、macos、linux 六大平台。

#### 12.3、了解 Dart 的一些基本语法

语法是一个很庞杂的事情，没必要专门去学完了再写程序。这里船长先提到的，都是你在你的第一个程序中会用到的。在 Dart 中引用一个文件很简单，只需要如下写法：

```dart
import 'package:project_name/.../xxx.dart';
```

注意 Dart 中每一句最后都需要 `;`，这一点与 java、c++ 类似。

Dart 也是面向对象语言，其面向对象的类写法也与 java 类似，是 `class`，并且如果是从一个类扩展而来的，就会用到与 java 类似的 `extends` 关键字。

Dart 语法中也有 `final`，其含义与 Java 中的 `final` 类似。

一个函数带有多个参数，那么每个参数之间都用 `,` 隔开，而最后一个参数后面加不加 `,`，Dart 都不会报错。

字符串用 `''` 而不是 `""`。

#### 12.4、Flutter 程序入口 `main.dart`

其程序一般从 `lib/main.dart` 文件开始运行，默认调用 `void main() {}`，其中会有一句简单的启动 App 的语句：

```dart
runApp(const MyApp());
```

`main` 函数也可以写成 `void main() => runApp(MyApp());` 的单行函数写法。

如果你命名的默认 `main.dart` 中的类名为 `MyApp` 的话，则需要在 `main.dart` 文件中定义一下 `MyApp`，大致的结构如下：

```dart
class MyApp extends StatelessWidget {
	const MyApp({super.key});

	@override
	Widget build(BuildContext context) {
		return MaterialApp(
			title: 'Flutter Demo',
			theme: ThemeData(
				primarySwatch: Colors.blue,
			),
			home: NewsListPage(),
		);
	}
}
```

`MyApp` 继承自 `StatelessWidget` 使得 App 整体也是一个 Widget（还记得前面说的在 Flutter 中一切皆为 Widget。其实在 Flutter 中连一些操作属性的东西都是 Widget，比如 Layout、Alignment 等等。

#### 12.5、App 获取 Http 请求的服务

船长要创建一个首页获取 news list，首页中有 news list，其中每一项都可以点击进入到一个 news detail 页。这样我们就需要三个 dart 程序文件：

* `news_service.dart` 发送 news list 请求；
* `news_list_page.dart` 展示 news list；
* `news_details_page.dart` 展示 news detail；

和船长一起来看下 `news_service.dart` 文件：

```dart
import 'dart:convert';
// 注意这里的 as http 写法
import 'package:http/http.dart' as http;

class NewsService {
  final String apiUrl =
      "https://newsapi.org/v2/top-headlines?country=cn&apiKey=<YOUR_API_KEY>";

  // 定义个 fetchNews 函数，这里注意返回值写法 Future<List<dynamic>>
  // 注意这里的 async 可以异步反馈，否则同步就会阻塞
  Future<List<dynamic>> fetchNews() async {

  	// 注意这里 response 是可以不定义类型的，与 Java 就很不同了
  	// 用 String 类型的 apiUrl 作为 Uri.parse() 函数的参数
  	// http.get() 函数用 await 关键词修饰，则可以等待返回。但因为函数 async，因此不会阻塞
    final response = await http.get(Uri.parse(apiUrl));

    if (response.statusCode == 200) {
      return jsonDecode(response.body)['articles'];
    } else {
      throw Exception('Failed to load news');
    }
  }
}
```

#### 12.6、App 的列表页

再看下 `news_list_page.dart`：

```dart
import 'package:flutter/material.dart';
import 'package:yongxian_news/services/news_service.dart';
import 'package:yongxian_news/pages/news_details_page.dart';

class NewsListPage extends StatefulWidget {
  @override
  _NewsListPageState createState() => _NewsListPageState();
}

class _NewsListPageState extends State<NewsListPage> {
  Future<List<dynamic>>? _newsList;

  @override
  void initState() {
    super.initState();
    _newsList = NewsService().fetchNews();
  }

  @override
  Widget build(BuildContext context) {

  	// 这里用到了脚手架 Scaffold，也是 Material Library 的一个 Widget，它提供默认的导航栏、标题、包含主屏幕 Widget 树状结构的 body 属性。
    return Scaffold(
      appBar: AppBar(title: Text('Flutter Demo App')),
      body: FutureBuilder<List<dynamic>>(
        future: _newsList ?? Future.value([]),
        builder: (context, snapshot) {
          if (snapshot.hasData) {
            return ListView.builder(
              itemCount: snapshot.data?.length ?? 0,
              itemBuilder: (context, index) {
                return ListTile(
                  title: Text(snapshot.data?[index]['title'] ?? ''),
                  subtitle: Text(snapshot.data?[index]['source']['name'] ?? ''),
                  onTap: () => _navigateToDetails(snapshot.data?[index]),
                );
              },
            );
          } else if (snapshot.hasError) {
            return Text('Error: ${snapshot.error}');
          }
          return CircularProgressIndicator();
        },
      ),
    );
  }

  void _navigateToDetails(Map<String, dynamic> article) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => NewsDetailsPage(article: article),
      ),
    );
  }
}

```

`NewsListPage` 可以看到与 `main.dart` 文件中的 `MyApp` 的定义形成对比，前者扩展自 `StatefulWidget`，后者扩展自 `StatelessWidget`。可以逐渐看到这是 Dart 程序的一个特点，就是每一个类都搭配一个 State 类，除了扩展自 `StatelessWidget` 类没有 State。

所以这里就是成对儿的，一个 `NewsListPage` 类，和一个 `_NewsListPageState` 类。在 `_NewsListPageState` 类中创建一个刚刚船长自己写的 `Service` 实例，并在其中的 `initState()` 函数中调用我们写的 `fetchNews` 函数。

然后在 `build` 函数中创建脚手架 `Scaffold`，其中定义了：

* `appBar`：直接创建一个带有 title 的 `AppBar(title: Text('涌现 AI 资讯'))`。
* `body`：这里可以和船长一起看到，你想要定义的页面的样式，就在这层层嵌套的 `ListView` 实例中，这个写法确实不太优雅。
	* `ListView` 的参数里传递了 `itemBuilder`，你想定义的关于 item 的内容也就在其中的了。
	* `ListTitle` 的参数里，我们可以看到 `title`、`subtitle`、`onTap` 点击后的动作函数，这里定义了一个 `_navigateToDetails`。

在 `_navigateToDetails` 函数中，调用了 `Navigator` 类的函数 `push`，其中传入的参数是 `context` 和`MaterialPageRoute` 实例。其中 `MaterialPageRoute` 创建实例的参数里有 `builder`，这里出现了 `NewsDetailsPage` 类。

#### 12.7、App 的二级页面实现

```dart
import 'package:flutter/material.dart';

class NewsDetailsPage extends StatelessWidget {
  final Map<String, dynamic> article;

  NewsDetailsPage({required this.article});

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(title: Text('涌现 AI 资讯详情')),
      body: SingleChildScrollView(
        child: Column(
          children: [
            article['urlToImage'] != null ? Image.network(article['urlToImage']) : Container(),
            SizedBox(height: 8),
            Text(
              article['title'],
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),),
            SizedBox(height: 8),
            Text(
              article['description'] ?? '',
              style: TextStyle(fontSize: 18),),
            SizedBox(height: 8),
            Text(
              'Source: ${article['source']['name']}',
              style: TextStyle(fontSize: 14, fontStyle: FontStyle.italic),
            ),
            SizedBox(height: 8),
            Text(
              'Published at: ${article['publishedAt']}',
              style: TextStyle(fontSize: 14, fontStyle: FontStyle.italic),
            ),
          ],
        ),
      ),
    );
  }
}

```

#### 12.8、对练手 App 的整体回顾

一般 Flutter 应用程序是在 `main.dart` 文件中实现一个继承自 `StatelessWidget` 的类比如叫 `MyApp`，然后在 `main.dart` 文件中的 `main` 方法里实例化一个 `MyApp` 对象，作为 Flutter 应用的根组件。

Flutter 在构建页面时，会调用继承自 `StatelessWidget` 的 `MyApp` 类实例的 `build` 方法。`build` 方法一般会提供一个 `MaterialApp`（也是一个 Widget）的 Widget 来描述这个 UI 界面。在这个 `MaterialApp` 里用 `title`、`theme` 和 `home` 三个组件。其中 `home` 就是一个继承自 `StatefulWidget` 的类，比如船长这里可以叫 `NewsListPage` 类，它是应用的首页 Widget 的，有状态的。

继承自 `StatefulWidget` 的 `NewsListPage` 组件，通过 createState 创建了状态管理类。

### 第 13 节 · Widget、导航路由、网络请求和项目发布

#### 13.1、常用 Widget

* `StatefulWidget`。
* `State`：通过 `setState` 方法告诉框架，某个 UI 发生了变化，Flutter 框架就会重新运行 State 实例的 `build` 方法来刷新 UI 的显示。
* `StatelessWidget`。
* 基础 Widget：`Text`、`Image`、`Button`、`TextField` 等组件。
	* Text：显示文本，有 `textAlign`、`maxLines` 最大行数、`overflow` 文本截断的方式、`style` 文本样式（对应 `TextStyle` 有很多属性，比如 color、fontSize、height、fontFamily、backgroundColor、下划线等 decoration 等等）等属性。
	* Button：比如 `FloatingActionButton` 悬浮按钮，其 onPressed 属性一般是一个回调函数。
* 布局 Widget：线性布局组件 Row、Column，弹性布局组件 Flex，Stack 组件。
	* 要了解主轴的概念，它与你是 Row 还是 Column 有关系，两种情况的 `mainAxisAlignment` 和 `crossAxisAlignment` 是相反的。
	* 其实 Row 和 Column 都是继承自 Flex 弹性布局的。Flex 里面支持 Expanded 孩子（只能在 Flex 里），Expanded 有一个 flex 属性控制怎么分割主轴上的全部空闲空间。
* 容器 Widget：Container、Padding、Center、DecoratedBox 组件。
* 交互 Widget：GestureDetector 组件。
* 滚动 Widget：ListView、GridView、CustomScrollView 组件。
	* ListView 跟 iOS、Android 类似组件都是滚动到了才创建，性能上更有优势。

#### 13.2、导航路由

iOS、Android 的页面导航管理都会维护一个路由栈，对应的方法是 `Navigator.push()` 和 `Navigator.pop()`，后者回退同时可以返回数据给主屏界面。`Navigator.pop()` 可以接受第二个参数（可选），数据通过 future 的方法传递返回值。

#### 13.3、网络请求

```dart
HttpClient httpClient = HttpClient();
HttpClientRequest request = await httpClient.getUrl(uri);
HttpClientResponse response = await request.close();
String responseBody = await response.transform(utf8.decoder).join();
httpClient.close();
json.decode(json);
```

#### 13.4、项目发布

发布 Release 版本，以 web 端为例。

```bash
mikecaptain@CVN % flutter build web
```

![](/img/src/2023/05/2023-05-01-captain-flutter-12.png)

## 第四章 · 调试

### 第 14 节 · 启用 Flutter DevTools

我们可以在 VSCode 中「Command + P」输入「>Flutter: Open DevTools」，下图是使用 DevTools 中的 Widget Inspector：

![](/img/src/2023/05/2023-05-01-captain-flutter-16.png)

也可以在「Command + P」输入「>Flutter: Open DevTools」选择「Open DevTools in Web Browser」在浏览器中打开。

![](/img/src/2023/05/2023-05-01-captain-flutter-17.png)

可以看到浏览器地址是 `http://127.0.0.1:9100/`，但是如果直接在浏览器打开看到的是下面的页面，需要输入所要连接的 App 的 URL：

![](/img/src/2023/05/2023-05-01-captain-flutter-18.png)

在 DevTools 我们可以看到各种开发调试工具：

* Flutter Inspector：用于调试和分析应用的 UI 布局。开发者可以查看和编辑小部件树、查看布局边界、调整布局参数等，以便更好地理解和调试应用的 UI。
* Performance：针对 UI 性能过低问题，比如滑动卡顿。
* CPU Profiler：分析应用的 CPU 使用情况。它提供了一个时间轴视图，显示应用在一段时间内 CPU 的占用情况。
* Memory：内存问题排查，可以帮助开发者查看应用的内存使用情况，并识别内存泄漏问题。
* Debugger：调试工具。
* Network：网络请求分析工具，可以帮助开发者查看应用发送和接收的网络请求，包括请求详情、请求时间线等。
* Logging：帮助开发者查看应用的日志输出，包括调试信息、错误日志等。开发者可以根据关键字过滤日志，以便更快地定位和解决问题。
* App Size：分析包大小。

#### 14.1、性能分析（Performance）

只能在真机或 Profile 模式下运行。先看下 Performance Overlay，绿色线是展示当前帧的耗时，UI thread、GPU thread 分开显示。耗时超过 16.6 毫秒（对应低于 60 FPS，即会丢帧）。如果点开「Performance Overlay」即如下：

![](/img/src/2023/05/2023-05-01-captain-flutter-21.png)

当点开「Performance Overlay」，则在运行 App 后能看到如下 UI thread、GPU thread 性能图表（Raster 对应的就是 GPU），上面是 UI Task Runner 线程耗时，下面是 GPU Task Runner 线程耗时，如下图：

![](/img/src/2023/05/2023-05-01-captain-flutter-22.png){: width="300"}

UI Task Runer 是 Flutter Engine 执行 Dart isolate code 的耗时，GPU Task Runner 是被执行设备 GPU 的相关的调用。

#### 14.2、性能优化常见关注点以及一个常见的性能优化案例

一般我们关注性能优化的点有：

* 提高 build 效率；
* 提高 paint 效率，经常通过 RepainBoundry 创建单独的 layer 减少重绘区域；
* 减少 build 中的逻辑处理：毕竟页面刷新时 Widget 的 build 会被频繁调用（每一帧都调用），所以 build 里应该只处理 UI 相关的逻辑；
* 减少 ClipPath、saveLayer 使用：saveLayer 会在 GPU 中分配一块新的绘图缓冲区，切换绘图目标，这个操作非常耗时；clipPath 会影响每个绘图指令，做一些相交操作，相交部分之外都会被剔除掉，也很耗时。
* 减少 Opacity Widget 的使用：尤其是在动画中，因为它会导致 Widget 每一帧都会被重建，可以使用 AnimatedOpcity 或 FadeInImage 进行代替。

在一个构建的 Widget 树结构中，看看从上到下的 Widget，如果 setState 的修改是在一个比较上面的 Widget 的 setState 中发生的，而改变的是一个比较下面的节点，这其实是存在性能优化空间的。优化的方法，就是把变化放在树结构里比较下面那个 Widget 的 setState 里。

总价一下，就是提高 build 的效率，让 setState 刷新数据尽量下发到比较底层（指树的底层）的节点。

## 附录

### `flutter doctor`

```bash
(base) mikecaptain@CVN ~ % flutter doctor                   
Doctor summary (to see all details, run flutter doctor -v):
[✓] Flutter (Channel stable, 3.7.12, on macOS 13.2.1 22D68 darwin-arm64, locale zh-Hans-CN)
[✓] Android toolchain - develop for Android devices (Android SDK version 33.0.2)
[✓] Xcode - develop for iOS and macOS (Xcode 14.3)
[✓] Chrome - develop for the web
[✓] Android Studio (version 2022.1)
[✓] Connected device (2 available)
[✓] HTTP Host Availability

• No issues found!
```

### `flutter create hello_world` 创建项目

```shell
(base) mikecaptain@CVN codespace % flutter create hello_world
Signing iOS app for device deployment using developer identity: "Apple Development: zhongchao.ustc@gmail.com (U4H338U5TW)"
Creating project captain_flutter...
Running "flutter pub get" in captain_flutter...
Resolving dependencies in captain_flutter... (1.3s)
+ async 2.10.0 (2.11.0 available)
+ boolean_selector 2.1.1
+ characters 1.2.1 (1.3.0 available)
+ clock 1.1.1
+ collection 1.17.0 (1.17.1 available)
+ cupertino_icons 1.0.5
+ fake_async 1.3.1
+ flutter 0.0.0 from sdk flutter
+ flutter_lints 2.0.1
+ flutter_test 0.0.0 from sdk flutter
+ js 0.6.5 (0.6.7 available)
+ lints 2.0.1 (2.1.0 available)
+ matcher 0.12.13 (0.12.15 available)
+ material_color_utilities 0.2.0 (0.3.0 available)
+ meta 1.8.0 (1.9.1 available)
+ path 1.8.2 (1.8.3 available)
+ sky_engine 0.0.99 from sdk flutter
+ source_span 1.9.1 (1.10.0 available)
+ stack_trace 1.11.0
+ stream_channel 2.1.1
+ string_scanner 1.2.0
+ term_glyph 1.2.1
+ test_api 0.4.16 (0.5.2 available)
+ vector_math 2.1.4
Changed 24 dependencies in hello_world!
Wrote 127 files.

All done!
You can find general documentation for Flutter at: https://docs.flutter.dev/
Detailed API documentation is available at: https://api.flutter.dev/
If you prefer video documentation, consider: https://www.youtube.com/c/flutterdev

In order to run your application, type:

  $ cd hello_world
  $ flutter run

Your application code is in hello_world/lib/main.dart.
```

### `flutter run` 运行项目

在 Chrome 中运行命令如下：

```bash
(base) mikecaptain@CVN hello_world % flutter run -d chrome
Launching lib/main.dart on Chrome in debug mode...
Waiting for connection from debug service on Chrome...                 ⣾
```

默认情况下，浏览器会打开 `http://localhost:56049/#/` 链接。

### `flutter devices`

```bash
(base) mikecaptain@CVN testspace % flutter devices
3 connected devices:

iPhone 14 (mobile) • D89CD348-58C2-4F3B-B07D-54E2A72AB1BA • ios            • com.apple.CoreSimulator.SimRuntime.iOS-16-4 (simulator)
macOS (desktop)    • macos                                • darwin-arm64   • macOS 13.2.1 22D68 darwin-arm64
Chrome (web)       • chrome                               • web-javascript • Google Chrome 112.0.5615.137
```

### `flutter pub get`

```bash
(base) mikecaptain@CVN yongxian_news % flutter pub get            
Running "flutter pub get" in yongxian_news...
Resolving dependencies... 
  async 2.10.0 (2.11.0 available)
  characters 1.2.1 (1.3.0 available)
  collection 1.17.0 (1.17.1 available)
  js 0.6.5 (0.6.7 available)
  matcher 0.12.13 (0.12.15 available)
  material_color_utilities 0.2.0 (0.3.0 available)
  meta 1.8.0 (1.9.1 available)
  path 1.8.2 (1.8.3 available)
  source_span 1.9.1 (1.10.0 available)
  test_api 0.4.16 (0.5.2 available)
Got dependencies!
```

### 其他

`flutter build web`、`flutter channel stable`、`flutter upgrade`。

## 参考

* https://flutter.cn/docs/get-started/install
* https://flutter.cn/
* https://flutter.cn/docs/get-started/web
* https://community.jiguang.cn/article/464286
* https://www.bilibili.com/video/BV1p14y1T79R/
* https://api.flutter.dev/flutter/material/BottomNavigationBar-class.html
* https://blog.logrocket.com/building-news-app-flutter/
* https://flutterawesome.com/flutter-news-app-using-newsapi/
* https://flutter.cn/docs/development/ui/widgets-intro
* https://flutter.cn/docs/get-started/flutter-for/uikit-devs