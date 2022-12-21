---
layout: post
title:  "如何使用 Jekyll 基于 Github Pages 搭建个人博客"
date:   2022-12-21 15:53:57 +0800
categories: jekyll update
---
# 用 Jekyll 创建一个 Github Pages 网站

#### 1、GitHub 上的准备

在 Github 上创建一个新的仓库，命名为「账户名.github.io」。然后将仓库拉取到本地：

    $ git clone https://github.com/username/username.github.io

创建一些 web 文件后再推到 Github 上就可以了：

    $ git add --all
    $ git commit -m "Initial commit"
    $ git push -u origin main

#### 2、了解 Ruby 和 Jekyll

Ruby 目前业界的主要应用都在 Web 开发领域，有不少框架，比如 Ruby on Rails、Sinatra、Padrino. 我们这里要用到的 Jekyll 是用 Ruby 实现的一个构建静态网站的工具，用 HTML 和 Markdown 作为源码，再通过布局和模板生成网页文件。

Jekyll 特别适合构建博客，支持标签、分类、搜索，并支持自定义模板和布局。

#### 3、了解 Gem

Gem 是 Ruby 常用的一个管理库的工具，类似于 Pip 是 Python 常用的一个管理库的工具。

为 Gem 配置国内的源，这样访问速度更快：

    gem sources --add https://mirrors.tuna.tsinghua.edu.cn/rubygems/ --remove https://rubygems.org/
    gem sources -l

#### 4、安装 Homebrew

Homebrew 是一个专门为 macOS 设计的开源软件包管理工具，熟悉 Linux 的朋友可以把 Homebrew 理解成 macOS 的 apt-get。先安装 Homebrew：

    $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

为了让 Homebrew 在国内安装快一些，可以替换下镜像源：

    echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.aliyun.com/homebrew/homebrew-bottles' >> ~/.bash_profile

以上用的是阿里云的源，也可以用网易的源：

    echo 'export HOMEBREW_BOTTLE_DOMAIN=http://mirrors.163.com/homebrew/bottles' >> ~/.bash_profile

Homebrew 安装、卸载软件的命令都很简单，brew install wget和brew uninstall wget。

#### 5、用 Homebrew 安装 Ruby

用 Homebrew 安装 chruby 和 ruby-install

    $ brew install chruby ruby-install xz

安装 Ruby 的最新版本：

    $ ruby-install ruby

这时候提示如下问题：

    >>> Updating ruby versions ...
    !!! Failed to download https://raw.githubusercontent.com/postmodern/ruby-versions/master/ruby/versions.txt to /Users/captain/.cache/ruby-install/ruby/versions.txt!
    !!! Failed to download ruby versions!

因为 raw.githubusercontent.com 在国内是被 blocked，所以用https://www.ipaddress.com查一下 IP 地址，然后修改下/etc/hosts：

    $ echo "185.199.111.133 raw.githubusercontent.com" >> /etc/hosts

然后再运行ruby-install ruby就可以正常安装了，这个过程会非常的慢，安装完成后，配置 zsh 脚本的 .zshrc 文件以便后续可以使用 chruby：

    $ echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
    $ echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.zshrc
    $ echo "chruby ruby-3.1.2" >> ~/.zshrc # run 'chruby' to see actual version

再看下 Ruby 版本对不对：

    $ ruby -v

Jekyll 官网要求 Ruby 版本大于 3.1.2p20.

#### 6、安装 Jekyll 和 Bundler

    $ gem install jekyll bundler

上面顺便安装了 Bundler，Bundler 是 Ruby 常用的管理项目依赖关系的工具，类似于 virtualenv 之于 Python，可以简化项目的包依赖管理，帮你维护一份 Gemfile 文件，里面包含了所有依赖关系。这个工具的名字叫 Bundler，使用的时候都是用这个词的动词 bundle 命令。

#### 7、使用 bundle 管理包依赖关系

创建 Gemfile 文件，Gemfile 是 Ruby 项目的依赖包管理文件：

    source 'https://rubygems.org'
    gem 'nokogiri'
    gem 'rack', '~> 2.2.4'
    gem 'rspec'
    gem 'jekyll'

然后安装依赖包，这里默认会根据运行命令时所在的目录的 Gemfile 来安装：

    $ bundle install

Gemfile.lock 是 Gemfile 的锁定版本，记录了当前项目所使用的所有依赖包的版本信息。下面把这两个文件都加入到 Git 版本控制中。

    $ git add Gemfile Gemfile.lock

#### 8、本地启动一下看看

先用 bundle 如下命令来启动：

    $ bundle exec jekyll serve

启动日志如下：

    Configuration file: none
                Source: /Users/captain/Workspace/poechant.github.io
           Destination: /Users/captain/Workspace/poechant.github.io/_site
     Incremental build: disabled. Enable with --incremental
          Generating... 
                        done in 0.014 seconds.
     Auto-regeneration: enabled for '/Users/captain/Workspace/poechant.github.io'
        Server address: http://127.0.0.1:4000
      Server running... press ctrl-c to stop.

然后打开浏览器输入http://localhost:4000看看效果：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1Lk3lbmW2Z7jOm96/img/238a75d0-8901-4d6c-9a62-524b5b604dd8.png)

这就说明 Jekyll 本地配置已经成功了。然后把当前的版本同步到 Git 上：

    $ git pull --no-rebase
    $ git push -u origin main

#### 9、用 Jekyll 创建一个项目

    $ jekyll new CaptainMikeBlog
    $ cd CaptainMikeBlog
    $ jekyll server

启动日志如下：

    Configuration file: /Users/captain/Workspace/poechant.github.io/CaptainMikeBlog/_config.yml
                Source: /Users/captain/Workspace/poechant.github.io/CaptainMikeBlog
           Destination: /Users/captain/Workspace/poechant.github.io/CaptainMikeBlog/_site
     Incremental build: disabled. Enable with --incremental
          Generating... 
           Jekyll Feed: Generating feed for posts
                        done in 0.365 seconds.
     Auto-regeneration: enabled for '/Users/captain/Workspace/poechant.github.io/CaptainMikeBlog'
        Server address: http://127.0.0.1:4000/
      Server running... press ctrl-c to stop.

再打开浏览器输入http://localhost:4000看看效果：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1Lk3lbmW2Z7jOm96/img/954955e5-0afd-49f9-8db1-44f77291f9fb.png)

#### 10、修改 Gemfile 文件

注释掉gem "jekyll"开头的这一行，修改# gem "github-pages"开头的这一行为：

    $ gem "github-pages", "~> GITHUB-PAGES-VERSION", group: :jekyll_plugins

其中的GITHUB-PAGES-VERSION改为具体的版本号，版本号参考https://pages.github.com/versions/，我写本文的时候github-pages最新版本号是227。关闭 Gemfile 文件然后命令行运行如下命令：

    $ bundle install

再本地启动服务器测试：

    $ jekyll server

得到如下提示：

    You have already activated i18n 1.12.0, but your Gemfile requires i18n 0.9.5. Prepending `bundle exec` to your command may solve this. (Gem::LoadError)

参考https://github.com/Homebrew/brew.sh/issues/845这个 issue 后如下解决：

    $ bundle add webrick
    $ bundle exec jekyll serve

这里注意jekyll server和bundle exec jekyll serve两个的区别是前者基本本地 Jekyll 版本启动服务，后者基于目录下的 Gemfile 文件启动服务，所以我们要用后者。

#### 11、配置 Github Pages

在 Github 的仓库页面进入「Settings - Code and Automation - Pages - Build and Deploy」，选择「Deploy from a branch」，然后选择你设定的分支。再选发布源的文件夹，这里我设置为根目录。然后「保存」。再修改 \_config.yml 文件：

    baseurl: ""
    url: "http://your-username.github.io"

将本地代码push到 Github 仓库中，在浏览器访问your-username.github.io即可，有时候可能要等几分钟。

#### 12、配置一个 Jekyll Theme

#### 参考

1、[https://bundler.io](https://bundler.io)

2、[https://jekyllrb.com/docs/](https://jekyllrb.com/docs/)

3、[https://zhuanlan.zhihu.com/p/87225594](https://zhuanlan.zhihu.com/p/87225594)

4、[https://chat.openai.com/chat](https://chat.openai.com/chat)

5、[https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll)

6、[https://docs.github.com/zh/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site](https://docs.github.com/zh/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

---

安装 Ruby、RubyGems、GCC、G++、Makefile，如何查看自己是否已经安装了：

:::
Ruby：版本 2.5.0 或更高，运行ruby -v看看。

RubyGems：运行gem -v。

GCC、G++ 和 Makefile：运行gcc -v、g++ -v和make -v。
:::

更新一下：

    sudo gem update --system

检查系统中是否有可用的软件更新，这一步会比较慢，需要一些耐心：

    $ softwareupdate --install -a

    $ gem install racc -v '1.6.1' --source 'https://rubygems.org/'

    $ bundle init

创建一个文件，内容如下：

    source 'https://rubygems.org'
    gem 'nokogiri'
    gem 'rack', '~> 2.2.4'
    gem 'rspec'

    $ bundle install

    $ git add Gemfile Gemfile.lock

如果你的电脑上已经有 Ruby、RubyGems、GCC、G++、Make，则可以直接安装 jekyll：

    $ gem install jekyll