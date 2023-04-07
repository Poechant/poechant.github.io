---
layout: post
title:  自然语言处理的常见词表示法（Word Representation）
date:   2022-12-18 23:33:09 +0800
categories: ai
tags: [AI, 人工智能, NLP, 自然语言处理, One-Hot, Word2Vec, GloVe, fastText]
description: 
excerpt: 
location: 杭州
author: 麦克船长
---

### 二、静态词向量

#### 1、One-Hot

这就要讲到一个很有名也很基础的独热（One-Hot）编码。比如我们有一个 12 个不同词汇的语料库，按照 One-Hot 编码，我们需要构建一个 12x12 的矩阵，比如下面这样：
我喜欢梅西。

> 我喜欢阿根廷球队。
> 阿根廷拿下世界杯冠军。

首先我们分次成「我 喜欢 梅西 带领 阿根廷 球队 赢得 2022 卡塔尔 世界杯 冠军 。」这些词，并根据整段话中每个两个词共同出现的频次（这就是为什么叫 Hot 的原因），构建如下矩阵：

|       | 我     | 喜欢  | 梅西   | 阿根廷| 球队   | 拿下  | 世界杯 | 冠军   | 。    |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 我    | 0     |  2    |  1    |  1    |  1    |   0   |   0   |    0  |   2   |
| 喜欢  | 2      | 0     |  1    | 1     | 1     |  0    |  0    |   0   |  2    |
| 梅西  | 1      | 1     |  0    | 0     | 0     |  0    |  0    |   0   |  1    |
| 阿根廷 | 1      | 1     | 0     | 0     | 1     | 1     | 1     |  1    | 2     |
| 球队  |  1     | 1     |  0    |  0    |  0    |  0    |  0    |  0    |  1    |
| 拿下  |  0     | 0     |  0    |  1    |  0    |  0    |  1    |  1    |  1    |
| 世界杯 | 0      | 0     | 0     | 1     | 0     | 1     | 0     | 1     | 1     |
| 冠军  |  0     |  0    |  0    |  1    |  0    |  1    |  1    |  0    |  1    |
| 。    |  2     |  2    |  1    |  2    |  1    |  1    |  1    |  1    |  0    |

这是一个词频共现表，这样每个词就被表示成了矩阵里的一行，也可以看成一个向量。这些词向量就组成了一个向量空间，空间中向量之间的关系可以表征词之间的关系，到此该方法的编码就结束了。

因为语言表达都有相似性，我们拿到语料用 One-Hot 编码方式表示的词，其实在后续其他 NLP 问题上一定程度可以复用，也就是这个方法是支持预训练（Pre-Traning）的，这大大节省了工程师的时间。但是从这样的分词和 One-Hot 编码表示词的方式里，你能看出什么问题吗？很明显这样的表示方式，是有损的，即有信息丢失的。都丢失了哪些信息？

* 问题 1：分词颗粒度粗，引发的子词信息丢失。把词进一步拆分能挖掘到的信息其实都丢失了，比如英文的词根（intelligence、intelligent、intelligen 其实有相关性，但是无法体现）。
* 问题 2：有序结构背后的信息丢失。词的表示只考虑了两个词的共现频次，语言的有序结构是不是丢了，比如「我打了他一巴掌」和「他打了我一巴掌」在 One-Hot 编码里体现不出任何差异。
* 问题 3：忽略一词多义。比如「东西」表示方向，也表示物品，也会被忽略。
* 问题 4：高频低频词。另外，像「我、的、。」这种高频词会影响两个可能并不强关联的词在空间中的表达，反过来低频词的关联也变相地被弱化了，我们可以概括为高低频词的问题。
* 问题 5：OOV（Out Of Vocabulary）问题，即不认识新词。
* 问题 6：数据稀疏。基于当下的机器学习处理方法和工具，这个矩阵里的空洞太多了，也就是 0 太多了，有数据稀疏的问题，而且非常浪费存储空间、算力，尤其想象一个 100 万个词的语料库，这个矩阵非常的大。

每个问题都是机会。比如高频低频词引发的问题，就引入了「缩小高频词权重、放大低频词权重」的思路，具体采用的指标常用到 PMI（Pointwise Mutual Information，点互信息）。我们继续往下看。

#### 2、Word2Vec（2013）

#### 3、GloVe（2014）

2014 年 Stanford 的 Jeff Pennington 等人在论文《GloVe: Global Vectors for Word Representation》中提出的 GloVe，是一种词表示法（word representation）。

从这里获取 GloVe 的最新代码：https://github.com/stanfordnlp/GloVe 。下载并解压后：

```shell
cd GloVe-master && make
```

正常运行将看到如下信息：

```shell
mkdir -p build
gcc -c src/vocab_count.c -o build/vocab_count.o -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
clang: warning: -lm: 'linker' input unused [-Wunused-command-line-argument]
gcc -c src/cooccur.c -o build/cooccur.o -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
clang: warning: -lm: 'linker' input unused [-Wunused-command-line-argument]
gcc -c src/shuffle.c -o build/shuffle.o -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
clang: warning: -lm: 'linker' input unused [-Wunused-command-line-argument]
gcc -c src/glove.c -o build/glove.o -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
clang: warning: -lm: 'linker' input unused [-Wunused-command-line-argument]
gcc -c src/common.c -o build/common.o -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
clang: warning: -lm: 'linker' input unused [-Wunused-command-line-argument]
gcc build/vocab_count.o build/common.o -o build/vocab_count -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
gcc build/cooccur.o build/common.o -o build/cooccur -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
gcc build/shuffle.o build/common.o -o build/shuffle -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
gcc build/glove.o build/common.o -o build/glove -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
```

然后运行 demo 脚本：

```shell
./demo.sh
```

正常运行后将得到如下：

```shell
mkdir -p build

$ build/vocab_count -min-count 5 -verbose 2 < text8 > vocab.txt
BUILDING VOCABULARY
Processed 17005207 tokens.
Counted 253854 unique words.
Truncating vocabulary at min count 5.
Using vocabulary of size 71290.

$ build/cooccur -memory 4.0 -vocab-file vocab.txt -verbose 2 -window-size 15 < text8 > cooccurrence.bin
COUNTING COOCCURRENCES
window size: 15
context: symmetric
max product: 13752509
overflow length: 38028356
Reading vocab from file "vocab.txt"...loaded 71290 words.
Building lookup table...table contains 94990279 elements.
Processed 17005207 tokens.
Writing cooccurrences to disk.........2 files in total.
Merging cooccurrence files: processed 60666468 lines.

$ build/shuffle -memory 4.0 -verbose 2 < cooccurrence.bin > cooccurrence.shuf.bin
Using random seed 1672234516
SHUFFLING COOCCURRENCES
array size: 255013683
Shuffling by chunks: processed 60666468 lines.
Wrote 1 temporary file(s).
Merging temp files: processed 60666468 lines.

$ build/glove -save-file vectors -threads 8 -input-file cooccurrence.shuf.bin -x-max 10 -iter 15 -vector-size 50 -binary 2 -vocab-file vocab.txt -verbose 2
TRAINING MODEL
Read 60666468 lines.
Initializing parameters...Using random seed 1672234542
done.
vector size: 50
vocab size: 71290
x_max: 10.000000
alpha: 0.750000
12/28/22 - 09:35.53PM, iter: 001, cost: 0.071355
12/28/22 - 09:36.03PM, iter: 002, cost: 0.052712
12/28/22 - 09:36.15PM, iter: 003, cost: 0.046689
12/28/22 - 09:36.26PM, iter: 004, cost: 0.043382
12/28/22 - 09:36.37PM, iter: 005, cost: 0.041456
12/28/22 - 09:36.47PM, iter: 006, cost: 0.040181
12/28/22 - 09:36.58PM, iter: 007, cost: 0.039276
12/28/22 - 09:37.11PM, iter: 008, cost: 0.038593
12/28/22 - 09:37.27PM, iter: 009, cost: 0.038052
12/28/22 - 09:37.39PM, iter: 010, cost: 0.037616
12/28/22 - 09:37.51PM, iter: 011, cost: 0.037249
12/28/22 - 09:38.04PM, iter: 012, cost: 0.036944
12/28/22 - 09:38.15PM, iter: 013, cost: 0.036681
12/28/22 - 09:38.26PM, iter: 014, cost: 0.036450
12/28/22 - 09:38.37PM, iter: 015, cost: 0.036244
$ python eval/python/evaluate.py
capital-common-countries.txt:
ACCURACY TOP1: 64.23% (325/506)
capital-world.txt:
ACCURACY TOP1: 26.68% (951/3564)
currency.txt:
ACCURACY TOP1: 4.19% (25/596)
city-in-state.txt:
ACCURACY TOP1: 25.45% (593/2330)
family.txt:
ACCURACY TOP1: 36.43% (153/420)
gram1-adjective-to-adverb.txt:
ACCURACY TOP1: 3.63% (36/992)
gram2-opposite.txt:
ACCURACY TOP1: 3.04% (23/756)
gram3-comparative.txt:
ACCURACY TOP1: 26.28% (350/1332)
gram4-superlative.txt:
ACCURACY TOP1: 9.68% (96/992)
gram5-present-participle.txt:
ACCURACY TOP1: 14.58% (154/1056)
gram6-nationality-adjective.txt:
ACCURACY TOP1: 53.58% (815/1521)
gram7-past-tense.txt:
ACCURACY TOP1: 13.01% (203/1560)
gram8-plural.txt:
ACCURACY TOP1: 25.08% (334/1332)
gram9-plural-verbs.txt:
ACCURACY TOP1: 5.75% (50/870)
Questions seen/total: 91.21% (17827/19544)
Semantic accuracy: 27.60%  (2047/7416)
Syntactic accuracy: 19.80%  (2061/10411)
Total accuracy: 23.04%  (4108/17827)
```

#### 4、fastText（2016）


### 二、动态词向量

