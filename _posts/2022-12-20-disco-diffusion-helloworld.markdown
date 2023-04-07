---
layout: post
title:  
date:   2022-12-20 23:30:19 +0800
categories: ai
tags: [Disco Diffusion, Diffusion, AIGC, CV, 计算机视觉, Google, Colab]
description: 
excerpt: 
location: 杭州
author: 麦克船长
---

### 一、

https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb

#### 1、Set up

#### 2、Diffusion and CLIP model settings

#### 3、Settings

##### Settings

* 参数`batch_name`：
* 参数`steps`：步长，默认值是 250，增加步长将为 AI 提供更多调整图像的机会，并且每次调整都会更小，从而产生更精确、更详细的图像。增加步骤是以更长的渲染时间为代价的。此外，虽然增加步数通常会提高图像质量，但超过 250~500 步的额外步数的回报会逐渐减少。通常来说，设置成 100 用来预览草图，250足以看到不错的结果，再之上常见的设置是 500~800，大幅增加了渲染时间缺可能看不出什么效果。

##### Extra Settings：
* `intermediate_saves`

##### Prompts

这里是最重要的，你的 text2image 的 text 就是这一段。默认的内容如下：

```python
# Note: If using a pixelart diffusion model, try adding "#pixelart" to the end of the prompt for a stronger effect. It'll tend to work a lot better!
text_prompts = {
    0: ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.", "yellow color scheme"],
    100: ["This set of prompts start at frame 100","This prompt has weight five:5"],
}

image_prompts = {
    # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
}
```

#### 4、Diffuse

都设置完成后就可以全部运行了。

### 二、问题解决

```shell
FileNotFoundError                         Traceback (most recent call last)
<ipython-input-15-1d47f7ced3c1> in <module>
      1 #@title ### 1.4 Define Midas functions
      2 
----> 3 from midas.dpt_depth import DPTDepthModel
      4 from midas.midas_net import MidasNet
      5 from midas.midas_net_custom import MidasNet_small

2 frames
/content/MiDaS/midas/backbones/next_vit.py in <module>
      6 from .utils import activations, forward_default, get_activation
      7 
----> 8 file = open("./externals/Next_ViT/classification/nextvit.py", "r")
      9 source_code = file.read().replace(" utils", " externals.Next_ViT.classification.utils")
     10 exec(source_code)

FileNotFoundError: [Errno 2] No such file or directory: './externals/Next_ViT/classification/nextvit.py'
```

### 参考

1. https://www.bilibili.com/video/BV1J44y1P7ti/?spm_id_from=333.999.0.0
2. https://zhuanlan.zhihu.com/p/527521419