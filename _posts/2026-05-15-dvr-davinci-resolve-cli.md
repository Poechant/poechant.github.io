---
layout: post
title:  dvr：写给人和 AI Agent 用的 DaVinci Resolve 命令行
date:   2026-05-15 09:00:00 +0800
categories: ai
tags: [dvr, DaVinci Resolve, CLI, Python, AI Agent, Video Editing Automation, SKILL.md, MCP, 命令行工具, 麦克船长]
description: dvr 是我最近开源的一个 DaVinci Resolve 18+ 的 Python 命令行工具，五个命令域（doctor / project / media / render / timeline）覆盖剪辑流水线里大量重复操作；更重要的是它被刻意做成一个 Agent-first CLI——JSON-first 输出、稳定的 errorCode 枚举、异步 jobId 状态机、自带 SKILL.md 打包发布。本文是 dvr 的介绍、上手实战，以及几个 Agent-first 设计决定的随手记。
excerpt: dvr 是我给 DaVinci Resolve 18+ 写的一个开源 Python 命令行——既给人用，也给 AI Agent 用。本文带你 5 分钟上手，并讲讲几个 Agent-first 的设计决定……
location: 杭州
author: 麦克船长
---

## 1. 一句话讲清楚

`dvr` 是我最近开源的一个 Python 命令行小工具，干的事很直白——**在命令行里操作 DaVinci Resolve 18+**。

为啥写它？说白了两点：

* **给人用**：剪辑这活儿，太多操作可以脚本化了——批量导素材、给 clip 打 flag、跑渲染、加 marker，这些重复劳动谁也不想手动点
* **给 AI Agent 用**：传统 GUI 软件 + 自由文本输出，agent 基本没法稳定地用。dvr 默认吐 JSON、错误用 `errorCode` 枚举、长任务用 `jobId` 状态机管起来——它是个**地地道道的 Agent-first CLI**

包名叫 `davinci-resolve-cli`（装的时候敲这个），命令是短一点的 `dvr`（日常敲这个）。现在 0.1 alpha，五个命令域（`doctor` / `project` / `media` / `render` / `timeline`）都已经跑通了——**能用了，欢迎来试**。

## 2. 30 秒上手

```bash
pipx install davinci-resolve-cli
```

前提是你本机已经装好了 DaVinci Resolve 18+（推荐 Studio 版）。装完先跑一个体检：

```bash
dvr doctor
```

正常情况下，Resolve 路径、版本、bridge 状态、API 端点会一路全绿。要是哪里不对，dvr 不会含糊——**具体哪一步坏了它都明明白白告诉你**：Resolve 没启动？External Scripting 那个开关没在 Preferences → System → General 里打开？版本太老？返回的 JSON 里都会带一个 `hint` 字段，照着改就行。

## 3. 五个命令域

| 域 | 干啥 | 主要子命令 |
|---|---|---|
| `doctor` | 环境体检 | `dvr doctor` |
| `project` | 项目操作 | `list / open / new / close / save / export / import / current` |
| `media` | 媒体管理 | `import / list / tag` |
| `render` | 渲染（异步） | `presets / submit / status / list / wait / cancel` |
| `timeline` | 时间轴 | `list / current / open / new / clips / marker add/delete/list` |

所有命令都吃 `--format json|yaml|table`；凡是会改 timeline 的命令，全部带 `--dry-run`——动手前可以先看看到底要改成啥样。

## 4. 实战四例

### 例一：批量导素材

```bash
dvr media import ~/Footage/Day1 --recursive --bin "Day1"
```

把 `~/Footage/Day1/` 整个目录递归扫一遍导进来，全扔到 master bin 下的 `Day1` 文件夹里。导完之后 bin 会自动切回你原来打开的那个——**不会动你 GUI 里当前的状态**，挺友好。

### 例二：异步渲染 + 等结果

```bash
# 提交任务，立刻拿到 jobId
JOB=$(dvr render submit \
  --preset "H.264 Master" \
  --timeline cur \
  --output ~/out.mp4 \
  --start \
  --format json | jq -r .jobId)

# 等任务完成（每 2 秒 poll 一次，进度打在 stderr）
dvr render wait "$JOB" --interval 2 --format json
```

`render submit` 一秒就返回，任务状态写到 `~/.dvr/jobs.json` 里——你关掉终端、重启 Resolve 都不要紧，下次 `dvr render status $JOB` 一样查得到。

### 例三：批量打 flag color

```bash
# 把 Day1 bin 里所有还没标 Green 的 clip 都标成 Green
IDS=$(dvr media list --bin "Day1" --format json \
  | jq -r '.[] | select(.flags | index("Green") | not) | .id')

dvr media tag $IDS --color Green --format json
```

DaVinci Resolve 自己那 16 种 flag color，全都能用。哪个 clip 打不上，会落到返回 JSON 的 `failed[]` 数组里——**不会全部回滚、也不会假装成功**，该报错就老实报错。

### 例四：脚本化加 marker（含 dry-run）

```bash
# 先 dry-run 看一眼会发生什么
dvr timeline marker add --at 01:00:05:00 --note "review" --dry-run

# 真做
dvr timeline marker add --at 01:00:05:00 --note "review" --color Blue
```

AI agent 处理这类会改东西的操作，比较稳的姿势是：先 `--dry-run`，把返回的 `planned[]` 给用户瞅一眼——「我准备这么改，你看行不行？」——确认了再真做。

## 5. 几个 Agent-first 的设计决定

写个 CLI 本身不难。难的是让 LLM agent **不靠太多 prompt 工程，就能稳稳地用**。下面这几个设计决定，给打算做类似 CLI 的同学参考。

### 5.1 双 mode 输出：TTY → table，pipe → JSON

```bash
dvr media list           # 终端里跑，出 rich 表格
dvr media list | jq ...  # 接了 pipe，自动切 JSON
```

判断逻辑就一行 `sys.stdout.isatty()`。当然你也可以用 `--format json|yaml|table` 强制指定，或者环境变量 `DVR_OUTPUT=yaml` 一把全局覆盖。

这事儿为啥要这么做？因为 agent 调 CLI 永远是非 TTY 的——它要的是稳定能解析的结构化输出。但人类敲同一条命令的时候，没人想看一坨 JSON 满屏滚。**一条命令、两类用户、两套最佳形态**，dvr 自己识别，不用你操心。

### 5.2 errorCode 枚举 + stderr JSON

所有错误都以 JSON 的形式打到 stderr：

```json
{"errorCode": "resolve_not_running", "message": "...", "hint": "..."}
```

`errorCode` 是一个**稳定枚举**——`resolve_not_running` / `version_unsupported` / `api_unavailable` / `validation_error` / `not_found` / `api_call_failed` / `internal_error`——不会因为我哪天心情好就改文案。agent 拿 `errorCode` 做分支判断，不用从自由文本里 grep 关键字猜意思。

退出码走 sysexits 那一套：`0` 成功 / `1` 用户用错了 / `2` Resolve 没起来 / `3` Resolve API 自己出问题。和 errorCode 是互补关系——shell 脚本看退出码，agent 看 JSON。

### 5.3 长任务用 jobId 状态机

`render` 是会跑挺久的操作。同步等 agent 会被卡死在那儿啥也干不了。所以 `render submit` 一调就立刻返回 `jobId`，状态写到 `~/.dvr/jobs.json`：

```bash
dvr render submit ... → 立刻返回 jobId
dvr render status JOB → 查状态
dvr render wait JOB   → 想等就阻塞等完
dvr render list       → 看所有 job
dvr render cancel JOB → 取消
```

agent 比较合理的姿势是：**`submit` → 该干啥干啥 → 适时 `wait` 或 `status` 看一眼**，别把整段对话同步阻塞掉。

### 5.4 SKILL.md 是发布物的一部分

包里直接塞了一份 [`SKILL.md`](https://github.com/poechant/davinci-resolve-cli/blob/master/SKILL.md)——Anthropic 那种 agent skill 系统能直接读的 markdown 格式，里头有五个工作示例（render / list media / batch tag / wait on job / preflight）。

`pyproject.toml` 用 `shared-data` 把它装到 `share/dvr/SKILL.md`：

```toml
[tool.hatch.build.targets.wheel.shared-data]
"SKILL.md" = "share/dvr/SKILL.md"
```

**把 agent skill 文件作为发布物的一部分塞进去**——这样 agent 系统装上 CLI 就能「开箱即发现、立马上手」——是我觉得 2026 年的 CLI 都应该开始养成的一个新习惯。

## 6. 工程上的几个克制

**140 个单测，CI 完全不依赖 Resolve**。DaVinci Resolve 是闭源 GUI 软件，没法在 CI 里跑——这点挺麻烦。所以我写了一个 `FakeResolve`，本质就是一个内存状态机——把所有命令域的状态用 Python dict 模拟出来，单测全部打 FakeResolve。Integration test 单独打 `pytest -m integration` 标记，要本机 Resolve 真在跑才会执行。开发循环 < 1 秒，PR 也不用专门拉个 mac 同学帮跑。

**有些命令暂时没做**。比如 `timeline cut` 和 `timeline move`，看着很直观、用户也想要——但 DaVinciResolveScript 压根没暴露这俩 API，要硬做就得走 Workflow Integrations 桥接，工作量直接翻倍。所以 v0.1 干脆不做，明明白白写在 CHANGELOG 的 "Deferred to v0.2" 段里。**说不做就是不做、不硬做、不偷偷骗用户**——这是开源工具最基本的诚信。

**全文档我没敢写「production-ready」这种词**。它就是 0.1 alpha——能跑通五个命令域、能让前期试用者帮我验证设计、但还有不少坑等着踩。v0.2 路线图：`timeline cut/move`、Windows / Linux 一类支持、MCP server 打包。

## 7. 但 dvr 不是终极方案：更远的目标是 Proactive Video Editing

写到这儿得讲一句心里话——

**我其实并不觉得 dvr 这种「在 GUI 软件外面套一层 CLI、再让 agent 调用」的形态，是 AI 做 Video Editing 的终极答案**。它更像一座**过渡桥**：当下还有一大堆朋友在用 DaVinci Resolve，那就先把它装进命令行、让 agent 也能用上——先把眼下的问题解了再说。

但真正有价值的、Prosumer 真正该用的**专业级 Video Editor AI 产品**，**形态不应该是这样**。它应该在另一个方向，我把这个方向叫做 **Proactive Video Editing**。

为啥？因为对 Prosumer 来说，真正有价值的内容创作**离不开人在回路**。AI 不可能、也不应该把人从创作链路里拿掉——那样剩下的就是流水线，不是创作了。但 AI 可以做一件传统软件做不到的事——**主动**：主动预判下一步、主动给出选项、在你意图还没完全成形的那一刻，已经把可能性铺到你面前。

这种「**人在回路 + AI 主动协同**」的形态，我管它叫 **Proactivation**——既不是 AI 替你干活、也不是你手动驱动 AI，而是两边**主动咬合**。我觉得只有走到这一层，Video Editing 这种创作场域里，效率和效果才可能同时往上抬一个台阶。

**而这件事，其实正是我们团队在做的产品**——一款面向 Prosumer 的、专业级的 Proactive Video Editor AI 产品。具体长什么样、什么时候放出来，以后会有专门的博客来聊；这里就先点一下，不展开。

所以现在的局面挺清楚的：**dvr 是当下能交付的过渡桥，Proactive Video Editing 是更远但更值得做的事**。两件事不冲突——很多创作者眼下还在 Resolve 里，那就先把 Resolve 这一段接好。

## 8. 试用入口

```bash
pipx install davinci-resolve-cli
dvr doctor
```

如果你也在做剪辑流水线的自动化、或者在给 DaVinci Resolve 写 agent skill，欢迎来试用、提 issue、拍 PR：

* GitHub：[github.com/poechant/davinci-resolve-cli](https://github.com/poechant/davinci-resolve-cli)
* PyPI：`pipx install davinci-resolve-cli`

---

写 CLI 这事儿听起来不性感，但**给 agent 写 CLI** 这件事，正在变成 2026 年开发者工具栈里越来越重要的一块拼图。
