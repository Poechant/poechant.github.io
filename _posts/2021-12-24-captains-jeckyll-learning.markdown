---
layout: post
title:  麦克船长的 Jekyll 快速教程
date:   2021-12-24 03:43:02 +0800
categories: web
tags: [Jekyll, Web, 前端]
description: Jekyll 是一个用 Ruby 实现的、使用 Liquid 模板引擎的静态网站生成器，它可以通过 Markdown 或者 HTML 等文件生成完整的静态网站。它特别适用于博客或者文章类的网站，因为可以自动生成博客的首页、分类页、标签页等等。因为使用 Liquid 引擎所以能在页面中使用变量、循环、条件语句等等，非常方便。虽然基于 Ruby 实现但使用起来并不需要掌握 Ruby，只需要了解一些基本的语法即可。
excerpt: Jekyll 是一个用 Ruby 实现的、使用 Liquid 模板引擎的静态网站生成器，它可以通过 Markdown 或者 HTML 等文件生成完整的静态网站。它特别适用于博客或者文章类的网站，因为可以自动生成博客的首页、分类页、标签页等等。因为使用 Liquid 引擎所以能在页面中使用变量、循环、条件语句等等，非常方便。虽然基于 Ruby 实现但使用起来并不需要掌握 Ruby，只需要了解一些基本的语法即可 ……
---

* 作者：麦克船长（钟超）
* 微信：sinosuperman

### 写在前面

Jekyll 是一个用 Ruby 实现的、使用 Liquid 模板引擎的静态网站生成器，它可以通过 Markdown 或者 HTML 等文件生成完整的静态网站。它特别适用于博客或者文章类的网站，因为可以自动生成博客的首页、分类页、标签页等等。因为使用 Liquid 引擎所以能在页面中使用变量、循环、条件语句等等，非常方便。虽然基于 Ruby 实现但使用起来并不需要掌握 Ruby，只需要了解一些基本的语法即可。

### Part 1、基本特点

#### 一、基本语法

* 变量：用双大括号表示变量 ```{{ site.title }}```  
* 过滤器：可以使用过滤器对变量进行操作，例如 `{{ site.title | upcase }}` 表示把网站的标题转换为大写。
* 支持循环与分支结构：比如 ```for-endfor``` 和 ```if-elsif-else-endif``` ：可以使用 `fo-endfor` 循环遍历列表或集合，例如 ```{% for page in site.pages %}{% endfor %}``` 表示遍历网站的所有页面。

#### 二、典型 Jekyll 项目结构及重要文件介绍

##### 1、配置文件 ```_config.yml```

首先看到下作为一个网站的基础设置，这里要特别注意不要遗漏 ```encoding: utf-8``` 这一条。

```yaml
# Site settings
encoding: utf-8
title: 麦克船长的技术、产品与商业博客
description: "麦克船长对于技术、产品、商业等领域的分享|AI,A.I.,NLP,神经网络,人工智能,自然语言处理,BERT,GPT,ChatGPT,OpenAI,阿里巴巴,P9,运营,淘宝,天猫,总监,高管"
url: "https://www.mikecaptain.com"
author:
  name: "Your Name"
  url: "https://www.mikecaptian.com"
```

然后是 Markdown 引擎的设置，及其高亮语法 Rouge 部分。

```yaml
# Markdown and highlighter
markdown: kramdown
highlighter: rouge
kramdown:
  input: GFM
  syntax_highlighter: rouge
```

一些要用到的插件也要设置进来，本博客只用到了基础插件两个。

```yaml
# Plugins
plugins:
  - jekyll-paginate
  - jekyll-sitemap
```

另外构建项目的一些关键设置，比如文章放在哪里、如何进行分页（每页多少条文章）等等作为一个静态博客网站的 build 类设置都在此。

```yaml
# Build settings
baseurl:  # Change this to your relative path (ex: /blog/), or leave just a /
source: .
destination: ./_site
permalink: /:title
paginate: 20
paginate_path: /page:num/
collections:
  posts:
    output: true
    permalink: /:year/:month/:day/:title/
    directory: _posts
```

##### 2、布局文件：```_layouts``` 目录下的文件规则

Jekyll 的 ```_layouts``` 目录包含了你的 Jekyll 站点中所使用的页面布局。每个页面布局是一个 ```HTML```模板，定义了你的站点中页面的框架和外观。你可以通过在你的文章或页面的头部添加一个 ```layout``` 字段来指定使用哪个布局来渲染该页面。

布局文件通常包含用于渲染页面的常见元素，例如头部、尾部和侧边栏。你可以在布局文件中使用 ```include``` 语句来插入你的站点的其他文件，例如 header.html 和 footer.html 文件。这样，你就可以在一个地方维护站点的头部和尾部，而不必在每个页面中都进行更新。

##### 3、页面文件及其头部

在一个页面的开头，用如下语法表示页面头部：

```markdown
---
layout: page
permalink: /categories/
title: Categories
---
```

每个页面文件的头部都会有layout，并与 ```_layouts``` 目录下的某个文件对应。

### Part 2、Jekyll 中的全局变量

Jekyll 中有许多全局变量可供使用，它们可以在模板中调用。这些变量提供了有关网站，页面，文章和其他内容的信息，可用于在模板中进行条件判断或显示信息。以下是 Jekyll 中常用的一些全局变量：

*   ```site```：包含有关网站的信息，如网站标题，描述，域名等。
*   ```page```：包含有关当前页面的信息，如标题，内容，布局等。
*   ```post```：包含有关当前文章的信息，如标题，作者，日期等。    
*   ```content```：包含当前页面或文章的内容。    
*   ```paginator```：包含有关分页的信息，如当前页码，总页数等。    
*   ```tags```：包含有关网站的所有标签的信息。    
*   ```related_posts```：包含与当前文章有关的文章的信息。

这些变量可以在模板中使用，比如：

```html
<h1>{{ page.title }}</h1>
<p>{{ site.description }}</p>
<ul>
  {% for category in site.categories %}
    <li>{{ category }}</li>
  {% endfor %}
</ul>    
```

Jekyll 还支持自定义全局变量，可以在配置文件 ```_config.yml``` 中添加任意的键值对，然后就可以在模板文件中使用这些变量了。例如，你可以在配置文件中添加如下内容：

```yaml
my_custom_variable: "Hello World"
```

然后就可以在模板文件中使用 ```site.my_custom_variable``` 访问这个自定义变量了。

#### 一、site变量

```site``` 的数据结构里包含了所构建的网站的各种基本信息和结构。

##### 1、```site.categories```

```site.categories``` 是一个 array，每个元素取出它的 ```first``` 就是 ```category``` 的名字，如下使用：

```
{% for category in site.categories %}
	{% assign temp_var = category | first %}
{% endfor %}
```

##### 2、site.pages

```site.pages``` 是一个包含所有页面的数组，不仅包括根目录下的页面，还包括所有子目录下的页面。因此，```site.pages``` 中包含的是整个网站中所有的页面。

```
{% for page in site.pages %}
	{{ page.title }}
{% endfor %}
```

##### 3、其他常用属性

*   ```site.title```：是网站的标题。    
*   ```site.related_posts```：相关文章的列表。    
*   ```site.data```：从 ```_data``` 目录加载的数据。    
*   ```site.static_files```：静态文件的列表。    
*   ```site.collections```：自定义集合的列表。

#### 二、```page``` 变量

在 Jekyll 中，```page``` 变量表示单独页面的数据。它是一个包含多个属性的对象，可以用来存储页面的信息并在模板中使用。一些常见的 ```page``` 变量属性包括：

* ```layout```：表示页面使用的布局模板的名称。    
* ```title```：表示页面的标题。    
* ```date```：表示页面的发布日期。    
* ```categories```：表示页面所属的分类列表。    
* ```tags```：表示页面所属的标签列表。
* ```content```：表示页面的内容（用 Markdown 格式书写）。

在模板中，可以使用 ```{{ page.属性名 }}``` 的方式来访问 ```page``` 变量的属性。例如，如果想在模板中输出页面的标题，可以使用 ```{{ page.title }}```。此外，```page``` 变量还有其他属性，如 ```permalink```、```excerpt```、```url``` 等，可以根据需要调用。

### Part 3、控制结构

#### 1、```if-else``` 分支结构

```
{% if tmp_var == "type1" %}
{% elsif tmp_var == "type2" %}
{% elsif tmp_var == "type3" %}
{% elsif tmp_var == "type4" %}
{% else tmp_var == "type5" %}
{% endif %}
```

#### 2、```for-endfor``` 循环结构

不带条件判断的 ```for``` 循环如下：

```html
{% for post in paginator.posts %}
	<!-- Your other sentences -->
{% endfor %}
```

带条件循环的 ```for``` 用 Jekyll 里的「过滤器」来实现：

```html
{% for page in site.pages | where: "dir", "categories" %}
	{{ page.title }}
{% endfor %}
```

##### 3、Jekyll 支持的其他结构包括：

* ```case``` 用于在多个可能的条件中执行代码的结构。
* ```capture``` 用于捕获输出的结构。
* ```cycle``` 用于循环一组字符串的结构。
* ```include``` 用于包含其他文件的结构。
* ```unless``` 用于在不满足指定条件时执行代码的结构。
* ```while``` 用于在满足指定条件时执行代码的结构。

### 参考：

1、[https://learn.cloudcannon.com/jekyll/list-posts-by-category/](https://learn.cloudcannon.com/jekyll/list-posts-by-category/)
2、[https://jekyllrb.com/docs/](https://jekyllrb.com/docs/)