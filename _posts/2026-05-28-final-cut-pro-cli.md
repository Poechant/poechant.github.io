---
layout: post
title:  fcp：写给人和 AI Agent 用的 Final Cut Pro 命令行
date:   2026-05-28 09:00:00 +0800
categories: ai
tags: [fcp, Final Cut Pro, CLI, Python, AI Agent, Video Editing Automation, FCPXML, MCP, SKILL.md, 命令行工具, 麦克船长]
description: final-cut-pro-cli 是我最近开源的一个 Apple Final Cut Pro 命令行工具。它面向剪辑准备阶段，把 Library、Event、Project、Media 和 Clip 操作做成可组合的 CLI 与 MCP 工具；更重要的是，它从第一天就按 Agent-friendly 的方式设计：JSON 输出、稳定 errorCode、FCPXML-first、不碰 UI scripting，并和 davinci-resolve-cli 保持命令面一致。
excerpt: final-cut-pro-cli 是我给 Apple Final Cut Pro 写的命令行工具。它不是要替代 FCP UI，而是把剪辑准备阶段里 Library、Event、Project、Media、Clip 这些可脚本化操作，变成给人和 AI Agent 都能稳定调用的接口……
location: 杭州
author: 麦克船长
---

## 1. 一句话讲清楚

`final-cut-pro-cli` 是我最近开源的一个 Python 命令行工具，命令叫 `fcp`，干的事很明确：

**把 Apple Final Cut Pro 的剪辑准备流程，变成命令行和 AI Agent 都能稳定调用的操作面。**

这里的关键词不是“替代 Final Cut Pro”，而是“剪辑准备”。专业剪辑软件的 GUI 很强，但很多前置工作其实并不需要人一下一下点：

* 创建 Library / Event / Project
* 批量导入素材
* 把素材追加到时间线
* 做基础 cut / move / trim / delete
* 生成可以被 Final Cut Pro 打开的 Library
* 把同一套能力暴露给 MCP，让 Agent 也能调用

这些事放在命令行里，价值就出来了：人可以写脚本，Agent 可以按步骤执行，团队可以把粗剪准备变成可复现的流程。

包名叫 `final-cut-pro-cli`，命令是短一点的 `fcp`。现在是 `0.1.0`，定位很克制：**面向 rough-cut preparation 的 MVP**。

## 2. 为什么要给 Final Cut Pro 写 CLI？

我之前写过一个 [`davinci-resolve-cli`](https://github.com/poechant/davinci-resolve-cli)，命令叫 `dvr`，主要是把 DaVinci Resolve 的一部分能力暴露给命令行和 Agent。

那为什么还要再写一个 Final Cut Pro 版？

因为视频编辑生态里，Final Cut Pro 和 DaVinci Resolve 是两类非常重要、但模型不同的专业工具。Resolve 更像是 Project / Timeline 这一套模型，而 Final Cut Pro 有它自己的 Library / Event / Project 层级：

```text
Final Cut Pro:
Library
└── Event
    └── Project
        └── Timeline / Clips
```

这套模型如果硬塞进一个通用视频 CLI，会很别扭。`fcp` 的设计思路是：**诚实暴露 Final Cut Pro 自己的领域模型，同时在 Project 层以下尽量和 `dvr` 对齐。**

也就是说，Agent 学会了：

```bash
dvr media import ...
dvr clip cut ...
dvr export ...
```

迁移到 FCP 时，只需要理解多出来的 `--library` 和 `--event` 作用域：

```bash
fcp media import ... --library MyLib --event "拍摄"
fcp clip cut --library MyLib --event "拍摄" --project "Vlog-001" --at 00:00:05
```

这件事对 Agent 很重要。人可以读文档、慢慢理解差异；Agent 更需要稳定、可迁移、可枚举的命令面。

## 3. 30 秒上手

安装：

```bash
pip install final-cut-pro-cli
```

环境要求也很明确：

| 项 | 要求 |
|---|---|
| 操作系统 | macOS |
| Final Cut Pro | 10.7+ |
| FCPXML | 1.11+ |
| Python | 3.10+ |

装好后先跑体检：

```bash
fcp --json doctor
```

如果环境不满足，`fcp` 会给出结构化错误，比如：

* `platform_unsupported`：不是 macOS
* `fcp_not_installed`：没找到 Final Cut Pro
* `fcp_version_too_old`：FCP 版本太老

这也是 Agent-friendly CLI 的基本功：不要让调用方从一段自由文本里猜“到底哪里坏了”，而是直接给稳定的 `errorCode`。

## 4. v0.1 的命令面

`fcp` v0.1 主要覆盖六组命令。

### Library

```bash
fcp library new <name> [--path <dir>]
fcp library list
fcp library open <name|path>
```

Final Cut Pro 的 Library 是 `.fcpbundle`，这是 FCP 和 Resolve 最大的模型差异之一。所以 `library` 是 `fcp` 独有的命令组，不强行和 `dvr` 对齐。

### Event

```bash
fcp event new <name> --library <ref>
fcp event list --library <ref>
```

Event 也是 FCP 特有概念。它很适合承载“拍摄日”“素材批次”“节目期数”这类分组。

### Project

```bash
fcp project new <name> --library <ref> --event <ref> [--resolution 1920x1080] [--fps 30]
fcp project list --library <ref> [--event <ref>]
fcp project info <name> --library <ref> --event <ref>
```

Project 层开始，`fcp` 就尽量和 `dvr` 保持同一种思路：新建、列表、信息读取，都用明确参数，不依赖 GUI 当前选中了什么。

### Media

```bash
fcp media import <video>... --library <ref> --event <ref>
```

v0.1 的 `media import` 会写入 FCPXML asset 引用，为后续时间线操作准备素材。

### Clip

```bash
fcp clip add --library <ref> --event <ref> --project <name> --asset <asset-id> [--duration <dur>]
fcp clip cut --library <ref> --event <ref> --project <name> --at <timecode>
fcp clip move --library <ref> --event <ref> --project <name> --clip <index> --to <timecode>
fcp clip trim --library <ref> --event <ref> --project <name> --clip <index> --start <tc> --end <tc>
fcp clip delete --library <ref> --event <ref> --project <name> --clip <index>
```

这组命令是 rough cut 最核心的部分：把素材放进来、切开、移动、修剪、删除。它们的参数名尽量和 `davinci-resolve-cli` 对齐，降低 Agent 迁移成本。

### MCP

```bash
fcp mcp serve
```

`fcp` 也可以直接以 MCP stdio server 的方式跑起来，给 Cursor、Claude Desktop、Claude Code 这类 Agent 客户端调用。MCP tool 名会是类似 `library.new`、`event.list`、`project.info`、`clip.cut`、`export.submit` 这样的 dotted name。

## 5. 一条 rough-cut 链路

假设我们要做一期 Vlog 的粗剪准备，可以从命令行跑这样一串：

```bash
fcp --json library new VlogE2E --path /tmp/fcp-test
fcp --json event new "拍摄" --library VlogE2E
fcp --json project new "Vlog-001" --library VlogE2E --event "拍摄" --resolution 1920x1080 --fps 30
fcp --json media import /tmp/test-assets/clip1.mov /tmp/test-assets/clip2.mov --library VlogE2E --event "拍摄"
fcp --json clip add --library VlogE2E --event "拍摄" --project "Vlog-001" --asset r2 --duration 60s
fcp --json clip cut --library VlogE2E --event "拍摄" --project "Vlog-001" --at 00:00:05
fcp --json library open VlogE2E
```

这条链路的目标不是“全自动剪出大片”，而是把一堆机械的准备工作可靠地跑完：

* Library 建好
* Event 建好
* Project 建好
* 素材引用写进去
* Clip 放到时间线上
* 基础 cut 做完
* 最后打开到 Final Cut Pro，交给人继续精修

这正是我对这类工具的定位：**CLI 和 Agent 负责结构化、重复性、可验证的部分；最终创作判断仍然留给人。**

## 6. 几个 Agent-friendly 的设计决定

写一个能被人敲的 CLI 不难；写一个 Agent 能稳定调用的 CLI，难点不一样。

### 6.1 默认支持 JSON

Agent 不应该解析彩色表格，也不应该从自然语言里抠字段。`fcp` 所有命令都支持：

```bash
fcp --json ...
```

成功输出走 JSON，错误也走结构化 JSON。这样 Agent 可以用字段做判断，而不是靠 prompt 猜。

### 6.2 稳定 errorCode

错误码按语义分段：

| 范围 | 含义 |
|---|---|
| 1-19 | 用户输入错误 |
| 20-39 | FCP 状态错误 |
| 40-59 | 长时操作错误 |
| 60+ | 内部错误 |

比如 `library_not_found`、`event_not_found`、`invalid_timecode`、`export_unavailable` 这些错误，对 Agent 来说就是控制流的一部分。

`library_not_found` 不是一段“看起来像找不到”的文案，而是一个可以稳定匹配的状态。

### 6.3 FCPXML-first

`fcp` 的主要编辑路径是 FCPXML：Library、Event、Project、Media、Clip 这些内容改动，尽量通过 FCPXML 文件操作完成。

这带来一个很重要的好处：**大部分剪辑准备操作不要求 Final Cut Pro 正在运行。**

Final Cut Pro 只在需要打开 Library 或做运行时检查时参与。v0.1 里 AppleScript 主要用在 `library open` 这种 runtime trigger 上。

### 6.4 不用 UI scripting

这是我很坚持的一点：`fcp` 不用 System Events，不模拟坐标点击，不发 key code，不靠 keystroke 硬怼 GUI。

为什么？

因为这类自动化对 Agent 来说太脆了。窗口位置、系统语言、菜单状态、焦点、权限弹窗，任何一个小变化都会让流程失控。

所以 v0.1 宁可明确告诉用户“这个能力还不能稳定自动化”，也不做看起来能跑、实则靠运气的 UI scripting。

### 6.5 SKILL.md 是发布物的一部分

项目里有一份 `SKILL.md`，描述了 Agent 该怎么用 `fcp`：先 `doctor`，再按 Library / Event / Project / Media / Clip 的顺序组织粗剪流程，调用时默认加 `--json`。

这件事我在 `dvr` 里也做了。2026 年写 CLI，如果它面向 Agent，那最好别只给人类 README，也要给 Agent 一份“操作手册”。

## 7. 一个必须讲清楚的限制：导出

Final Cut Pro 11.1.1 的 scripting dictionary 里，没有暴露稳定的 `export` / `share` / `render` 命令。

这意味着什么？

意味着 v0.1 不能诚实地承诺“命令行自动导出 MP4”。如果要硬做，大概率只能走 UI scripting：点菜单、按快捷键、等弹窗、猜按钮。

我不想这么做。

所以 `fcp export submit` 在 v0.1 保留了命令面，但会返回结构化的：

```json
{
  "errorCode": "export_unavailable",
  "message": "...",
  "hint": "..."
}
```

这看起来不如“我已经自动导出了”爽，但它更诚实，也更适合 Agent。Agent 至少能知道：这不是参数错了，不是 FCP 崩了，而是当前版本的自动导出路径不可用。

同理，`subtitle` 命令组在 v0.1 也是预留状态，还没有完整写入字幕或标题内容。把边界讲清楚，比把 roadmap 包装成已完成能力更重要。

## 8. 和 dvr 的关系：两座桥，不是一个终点

`dvr` 和 `fcp` 其实是一组姊妹项目：

* `dvr`：面向 DaVinci Resolve
* `fcp`：面向 Apple Final Cut Pro

它们共同回答的是一个更大的问题：

**专业视频软件如何进入 Agent 工作流？**

短期看，答案是先把现有专业软件的可脚本化部分接出来，变成 CLI、JSON、errorCode、MCP tool、SKILL.md。这样 Agent 才能可靠地做事，而不是盯着 GUI 猜。

但长期看，我仍然认为“在传统 GUI 软件外面套 CLI”不是 Video Editing AI 的终局。它更像一座过渡桥：让今天还在 FCP / Resolve 里工作的创作者，先获得一部分自动化和 Agent 协作能力。

真正更远的方向，还是我之前反复提到的 **Video-editing Agent** 和 **Proactive Video Editing**：AI 不只是等你下命令，而是在剪辑工作流里理解上下文、预测下一步、和人一起推进创作。

`fcp` 做的是这条路上非常实际的一步：先把当下能稳定交付的那部分能力，变成可调用、可测试、可组合的接口。

## 9. Roadmap

v0.1 已经完成 rough-cut preparation 的 MVP：

* Library / Event / Project
* Media import
* Clip add / cut / move / trim / delete
* `library open`
* `export_unavailable` 结构化 fallback
* MCP stdio server
* Real FCP e2e 测试：验证生成的 Library 能被 Final Cut Pro 打开

后面几步大概是：

* **v0.2**：继续和 `davinci-resolve-cli` v0.2 对齐，补 cross-platform FCPXML 操作、recipe mode、批量 media glob/import helper
* **v0.3**：增强 razor-cut 工作流，支持 MCP HTTP，补完整 subtitle/title support 和 config files

这里有个很现实的原则：FCPXML 能稳定做的，就往前推；Final Cut Pro 没有公开自动化接口的，不靠 UI scripting 硬装。

## 10. 试用入口

```bash
pip install final-cut-pro-cli
fcp --json doctor
```

项目地址：

* GitHub：[github.com/poechant/final-cut-pro-cli](https://github.com/poechant/final-cut-pro-cli)
* PyPI：`pip install final-cut-pro-cli`

如果你是 Final Cut Pro 用户，或者正在做视频工作流自动化 / AI Agent 剪辑工具，欢迎试试，也欢迎提 issue。

写给 Agent 用的 CLI，表面上是在做命令行工具；往深处看，其实是在给下一代创作软件铺接口。
