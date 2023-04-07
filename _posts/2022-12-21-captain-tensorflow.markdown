---
layout: post
title:  麦克船长 TensorFlow 技术笔记
date:   2022-12-21 03:45:31 +0800
categories: ai
tags: [AI, 人工智能, TF, TensorFlow, 张量]
description: 
excerpt: 
katex: True
location: 杭州
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

#### 一、tf

##### `tf.arg_max`

`tf.arg_max` 是 TensorFlow 中的一个函数，用于返回一个张量中最大值的位置。

这个函数接收两个参数：

* `input`：需要求最大值位置的张量
* `dimension`：指定求最大值位置的维度。如果 `dimension = -1`，则表示在最后一维求最大值位置；如果 `dimension = 0`，则表示在第一维求最大值位置。

返回值是一个整数张量，表示最大值位置。

##### `tf.abs` 求张量所有元素的绝对值

`tf.abs` 这个函数用来求给定张量中所有元素的绝对值，它会返回一个和输入张量类型相同，但元素值都是绝对值的新张量。例如，`tf.abs([-1, -2, 3])` 会返回 `[1, 2, 3]`。

##### `tf.one_hot`



##### `tf.ones_like`

`tf.ones_like` 函数的作用是根据给定的张量创建一个新的全部元素为 `1` 的张量。这个新张量的形状和类型与给定的张量相同。

例如：

```python
import tensorflow as tf
a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.ones_like(a)
print(b)
```

输出结果为：

```python
[[1 1 1]
 [1 1 1]]
```

即 `b` 是一个全部元素都是 `1` 的矩阵，形状和类型和 `a` 相同。

##### `tf.sign` 求张量所有元素的符号函数

`tf.sign`：这个函数用来求给定张量中所有元素的符号，它会返回一个和输入张量类型相同，但元素值为 1, 0 或 -1 的新张量。对于正数，返回 1，对于 0，返回0，对于负数，返回 -1。例如，`tf.sign([-1, -2, 3])` 会返回 [-1, -1, 1]。

##### `tf.reduce_sum`

`tf.reduce_sum` 函数是 TensorFlow 中的数学运算函数之一，用于沿着指定的维度对张量进行求和运算，所以是先「reduce」维度，再「sum」被消掉这个维度上的数值和。它接受三个参数：

* `input_tensor`：一个待求和的张量。
* `axis`：一个整数或者整数列表，表示沿着哪些维度求和。如果没有指定，则默认对所有元素求和。`axis=-1` 表示最后一个维度，`axis=-2` 表示倒数第二个维度。
* `keepdims`：一个布尔值，表示是否保留被求和的维度。

最终输出是一个降了一阶的张量，降哪一阶取决于参数 `axis`。下面是一个代码示例，比如有一个张量 `a`，形状为 (2, 3, 4)，如果我们使用 `tf.reduce_sum(a, axis=1)`，结果会是一个新的张量，形状为 (2, 4)，每个元素的值都是对应维度上的和。

```python
import tensorflow as tf
a = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
b = tf.reduce_sum(a, axis = 1)
print(b)
```

输出结果为：

```python
tf.Tensor(
[[15 18 21 24]
 [15 18 21 24]], shape=(2, 4), dtype=int32)
```

我们可以看出，这个例子中，原始矩阵 `a` 是 `2 x 3 x 4` 的三维张量，我们对它求和的时候使用了 `axis = 1`，所以将第二维上的所有数相加，得到了一个新的 2 x 4 的矩阵。

##### `tf.tile`

tf.tile 函数用于重复一个张量，其中参数是需要重复的张量和重复次数。例如：

a = tf.constant([1, 2, 3])
b = tf.tile(a, [2])

此时 b 的值就是 [1, 2, 3, 1, 2, 3]。另一个例子：

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.tile(a, [2, 3])
```

此时 b 的值就是 [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4], [1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]。可以看到在第一维重复了 2 次，在第二维重复了 3 次。还可以使用多维重复，例如：

```python
a = tf.constant([1, 2, 3])
b = tf.tile(a, [2, 2, 2])
# Output: [[[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]], [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]]
```

##### `tf.transpose`

`tf.transpose` 是 TensorFlow 中的一个函数，用于对一个 Tensor 进行转置操作。转置操作是指将一个矩阵中的行和列互换位置。在 TensorFlow 中，`tf.transpose` 函数接受两个参数：

第一个参数为需要转置的 `Tensor`。
第二个参数为一个整型数组，表示转置后 `Tensor` 的维度排列顺序。
例如，对于一个 `3 x 4` 的 `Tensor`，如果想要将其行列转置，则可以使用如下语句：

```python
original_tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
transposed_tensor = tf.transpose(original_tensor, [1, 0])
```

这样 `transposed_tensor` 就是一个 `4 x 3` 的 `Tensor` 了。值得关注的是，`transpose` 不止对矩阵可以转置，也可以对高维的张量进行转置操作。

`outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))` 这句话中的 `tf.transpose(K_, [0, 2, 1])` 意思是调换 `K_` 的第 0 维和第 2 维的位置，也就是对 `K_` 做矩阵转置操作。那么这句话的整体意思是对 `Q_` 和 `K_` 进行矩阵乘法运算，结果的维度为 `(h*N, T_q, T_k)`。

##### `tf.where`

`tf.where(condition, x=None, y=None)` 函数会根据 `condition` 中的元素值，从 `x` 和 `y` 中选择元素。当 `condition` 中的元素值为 `True` 时，从 `x` 中选择元素，否则从 `y` 中选择元素。`x` 和 `y` 的形状必须相同。

例如：

```python
condition = tf.constant([True, False, True])
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])
tf.where(condition, x, y)
# Output: [1, 5, 3]
```

在这个例子中，`condition` 数组为 `[True, False, True]`，所以从 `x` 数组中选择第一个和第三个元素，从 `y` 数组中选择第二个元素。

#### 二、`tf.nn`

##### `tf.nn.moments`





#### 三、`tf.layer`

##### `tf.layers.dense`

用来构建一个全连接层(fully connected layer)。这个层会将输入数据映射到输出结果，输入数据可以是一个二维张量，输出结果是一个二维张量。

* 在这个函数中可以传入两个参数，第一个是输入数据，第二个是输出结果中维度的大小。
* 这个层会自动创建一个权重矩阵 `W` 和偏置向量 `b`，将输入数据乘上 `W` 并加上 `b`，得到输出结果。
* 如果输入数据是 `nm` 的矩阵，输出结果是 `nk` 的矩阵，则权重矩阵 `W` 是 `m*k` 的矩阵，偏置向量 `b` 是 `k` 维向量。

在这里，`self.logits = tf.layers.dense(self.dec, len(en2idx))` 中 `self.dec` 为输入数据，`len(en2idx)` 为输出结果的维度，这里的输出维度就是类别数。