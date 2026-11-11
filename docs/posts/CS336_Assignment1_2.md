---
title: CS336 Assignment1 (Part 2)
date: 2026-11-11
description: CS336 Assignment1 学习笔记
---

# Transformer Language Model Architecture
## Transformer LM
给定一串 token ID，模型先把这些离散的数字编号转换成连续的向量表示，然后经过多层 Transformer Blocks 进行上下文信息处理，最后再把处理后的向量转换成对词表中每个 token 的预测分数（logits），用来预测下一个 token。

![](\images\CS336_Assignment1\img32.png)

### Token Embeddings
Transformer 不能直接处理离散的 Token ID，因此首先使用 Token Embedding Layer，把每个 Token ID 转换成一个连续的高维向量。每个嵌入层接收一个形状为 (batch_size, sequence_length) 的整数张量，并输出一个形状为 (batch_size, sequence_length, d_model) 的向量序列。

假设我们要用 Transformer 同时处理 两句话：
- Sequence 1：我 擅长 python
- Sequence 2：你 擅长 java 

每句话都有 3 个 token。

* `batch_size = 2`
* `sequence_length = 3`
* `d_model = 4`

输入 Token IDs：

```text
[
  [12, 35, 91],
  [28, 63, 10]
]
```

输入 shape：

$$(2,3)$$

经过 Token Embedding：

```text
[
  [
    [0.2, 0.5, -0.1, 0.7],
    [0.6, -0.3, 0.8, 0.1],
    [0.4, 0.9, -0.2, 0.3]
  ],
  [
    [0.1, 0.2, 0.6, -0.4],
    [0.7, 0.3, -0.5, 0.8],
    [0.9, -0.1, 0.4, 0.2]
  ]
]
```

输出 shape：

$$
(2,3,4)
$$

形状变化：

$$
(2,3)\rightarrow(2,3,4)
$$

即：

$$
(batch\_size,\ sequence\_length)
\rightarrow
(batch\_size,\ sequence\_length,\ d_{\text{model}})
$$
### Pre-norm Transformer Block
嵌入之后，激活值会经过多个结构相同的神经网络层进行处理。一个标准的decoder-only Transformer语言模型由 num_layers 个相同的层（通常称为Transformer“块”）组成。

每个Transformer块接收一个形状为 (batch_size, sequence_length, d_model) 的输入，并输出一个相同形状的张量 (batch_size, sequence_length, d_model)。每个块通过自注意力机制在序列中聚合信息，并通过前馈网络层对其进行非线性变换。

Token Embedding 之后，得到的 activations 会继续经过 num_layers 个结构相同的 Transformer Block。对于 decoder-only Transformer，每个 Block 的输入和输出 shape 都是：

$$
(batch\_size,\ sequence\_length,\ d_{\text{model}})
\rightarrow
(batch\_size,\ sequence\_length,\ d_{\text{model}})
$$

每个 Transformer Block 主要做两件事：通过 Self-Attention 聚合序列中不同位置的信息，并通过 Feed-Forward Layers 对表示进行非线性变换。因此，经过多个 Block 后，shape 不变，但 token 的表示会不断融合上下文信息。

本文实现的是 Pre-norm Transformer Block。由于采用 pre-norm 结构，在最后一个 Transformer Block 之后还需要额外进行一次层归一化，Layer Normalization，使最终输出保持合适的数值尺度。

最后，使用一个可学习的线性变换（LM Head），把 Transformer 的最终输出映射到整个词表，得到 next-token logits。
## Remark: Batching, Einsum and Efficient Computation
这部分主要解决如何在 transfomer 中进行相对高效的计算。

首先了解 "batch-like dimensions" 的概念，batch-like dimensions 是“独立实例索引维度”。这些维度的大小决定了有多少个互相独立的样本/实例/并行任务；算子在语义上会对每个实例独立地做同样的操作。具体例子如下：

对于向量 x，x.shape = $32, 128, 512$，(32: batch、128: sequence length 代表每个 batch 有多少 token、512: 每个 token 的向量维度)，现在有一个权重矩阵 A，A.shape = $512, 512$，我们希望对每个 batch 每个 token 都乘以向量矩阵，x -> xA。

最基础的写法为：
`for b in range(batch_size):`
    `for s in range(sequence_length):`
        `out[b, s] = x[b, s] @ A`

但完全可以直接计算
`out = x @ A`

pytorch 会直接进行矩阵乘法，shape 变化：(b, s, d) @ $d, d$ -> $b, s, d$，这里的 $b, s$ 维度就被称为 batch-like demensions。

在正常计算中，我们常常将 batch-like demensions 保留在 pytorch 张量形状的最前面，所以在操作过程中可能需要多次使用 view、reshape 和 transpose 操作来调整形状。这样比较繁琐，而且常常使代码难以阅读，难以追踪张量的实际形状，因此在 torch.einsum 中使用 einsum 表示法。其中两个核心操作是：
- einsum：可对任意维度的张量进行缩并（tensor contraction）
- rearrange：可重新排列、拼接或分割张量维度

下面为具体的例子展示：
![](\images\CS336_Assignment1\img33.png)

![](\images\CS336_Assignment1\img34.png)

原始图片 `images`

```python
images = torch.randn(64, 128, 128, 3)
```

shape：

```text
(64, 128, 128, 3)
  ↑    ↑    ↑    ↑
  B    H    W    C
```

其中：

```text
B = batch = 64
H = height = 128
W = width = 128
C = channel = 3
```

即：

```text
64 张图片
每张图片 128 × 128
每个像素包含 RGB 三个值
```

一个具体元素：

```python
images[b, h, w, c]
```

表示：

> 第 `b` 张图片，第 `h` 行、第 `w` 列像素的第 `c` 个颜色通道。

数学上可以写成：

$$
X_{b,h,w,c}
$$

---

生成 10 个缩放因子

```python
dim_by = torch.linspace(
    start=0.0,
    end=1.0,
    steps=10
)
```

大约得到：

```text
[0.0000,
 0.1111,
 0.2222,
 0.3333,
 0.4444,
 0.5556,
 0.6667,
 0.7778,
 0.8889,
 1.0000]
```

shape：

```text
(10,)
```

可以记成：$s_d$

其中：

```text
d = 0, 1, ..., 9
```

不同数值代表不同亮度：

```text
0.0 × image → 最暗
0.5 × image → 大约一半亮度
1.0 × image → 保持原值
```

---

第一次 `rearrange`

```python
dim_value = rearrange(
    dim_by,
    "dim_value -> 1 dim_value 1 1 1"
)
```

原来：

```text
dim_by.shape = (10,)
```

现在：

```text
dim_value.shape = (1, 10, 1, 1, 1)
```

可以按照下面的维度理解：

```text
(1, 10, 1, 1, 1)
 ↑   ↑   ↑  ↑  ↑
 B   D   H  W  C
```

其中：

```text
D = dim_value = 10
```

注意：

> 这里的 `1` 表示新增一个长度为 1 的维度，不是向张量中添加数值 `1`。

原来的数值完全没有变化。

原本：

```python
dim_by[d]
```

现在对应：

```python
dim_value[0, d, 0, 0, 0]
```

并且：

$dim\_value[0,d,0,0,0]=dim\_by[d]$

例如：

```text
dim_value[0,0,0,0,0] = 0.0000
dim_value[0,1,0,0,0] = 0.1111
...
dim_value[0,9,0,0,0] = 1.0000
```

---

第二次 `rearrange`

原来的图片：

```text
images.shape
=
(64, 128, 128, 3)

(B, H, W, C)
```

执行：

```python
images_rearr = rearrange(
    images,
    "b height width channel -> b 1 height width channel"
)
```

得到：

```text
images_rearr.shape
=
(64, 1, 128, 128, 3)
```

也就是：

```text
(64, 1, 128, 128, 3)
  ↑  ↑   ↑    ↑    ↑
  B  D   H    W    C
```

原来的：

```python
images[b, h, w, c]
```

现在变成：

```python
images_rearr[b, 0, h, w, c]
```

但是数据值没有变化：

$images\_rearr[b,0,h,w,c]=images[b,h,w,c]$

这里新增的第二维长度为 `1`，是为了之后生成：

```text
10 个不同亮度版本
```

---

两个 Tensor 的 shape 对齐

现在有：

```text
                 B    D    H     W    C

images_rearr:   64    1   128   128   3

dim_value:       1   10     1     1   1
```

然后执行：

```python
dimmed_images = images_rearr * dim_value
```

这里的 `*` 是：

> 逐元素乘法 + broadcasting。

---

Broadcasting 如何进行

逐个维度比较：

```text
                 B    D    H     W    C

images_rearr:   64    1   128   128   3
dim_value:       1   10     1     1   1
------------------------------------------------
result:         64   10   128   128   3
```

广播规则：

> 两个维度相等，或者其中一个维度为 `1`，就可以广播。广播就是把长度为 1 的维度在逻辑上重复到目标大小，从而让两个张量形状兼容后进行逐元素运算。

在 batch 维度上，`dim_value` 的大小是 1，所以广播时会在逻辑上扩展为 64；因此 64 张图片都会复用同一套 10 个缩放因子进行相乘，效果上等价于把 `dim_value` 复制 64 份，但实际通常不会真的在内存中复制。其他维度同理，此处不再赘述。

---

最终结果

最终：

```python
dimmed_images.shape
```

得到：

```text
(64, 10, 128, 128, 3)
```

可以解释成：

```text
(batch, dim_version, height, width, channel)
```

即：

```text
64 张原始图片
×
每张图片 10 个亮度版本
×
128 × 128 × 3
```

所以共有：

$64\times10=640$

个图片版本。

---

## Basic Building Blocks: Linear and Embedding Modules

### Parameter Initialization
在训练神经网络时，初始化模型参数是十分重要的，因为不恰当的初始化会导致梯度爆炸或梯度消失等情况，我们采取以下初始化规则：
![](\images\CS336_Assignment1\img35.png)

![](\images\CS336_Assignment1\img36.png)

可以通过 torch.nn.init.trunc_normal_ 来初始化权重。

我们在本作业中使用**列向量表示法**。在许多论文中，使用行向量的线性变换通常写作：

$$
y = xW^\top,
$$

其中 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，$x \in \mathbb{R}^{1 \times d_{\text{in}}}$。

我们将采用等价的**列向量形式**：

$$
y = Wx,
$$

其中 $x \in \mathbb{R}^{d_{\text{in}}}$，$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$。

在使用 PyTorch 默认的行优先（row-major）内存布局实现时，若直接进行矩阵乘法需注意转置问题；使用 `einsum` 可避免此类问题。

### Linear Module
线性层 (Linear layer) 是 transformer 和一般神经网络的基本构件模块，接下来，我们将实现一个继承自 torch.nn.Module 并执行线性变换的自定义 Linear 类。这里的线性变换为：y = Wx，遵循大多数现代 LLMs，没有偏置项。

![](\images\CS336_Assignment1\img37.png)

![](\images\CS336_Assignment1\img38.png)

![](\images\CS336_Assignment1\img39.png)

### Embedding Module
![](\images\CS336_Assignment1\img40.png)

![](\images\CS336_Assignment1\img41.png)

### Pre-Norm Transformer Block
每个 Transformer 模块包含两个子层：(1) 多头自注意力机制（Multi-Head Self-Attention）和 $2$ 位置级前馈网络（Position-wise Feed-Forward Network）

在最初 Transformer 论文中，采用的是后归一化（post-norm），即每个子层先残差连接后再进行层归一化，假设模块输入为 x，则后归一化中模块执行如下操作：

自注意力子层：$y = RMSNorm(x + MultiHeadSelfAttention(x))$

前馈网络子层：$y = RMSNorm(x + FFN(x))$

但在之后研究中发现，**预归一化**可以提高 Transformer 的训练稳定性，所以现在预归一化的 Transformer 已成为当今语言模型的标准。具体操作如下：

自注意力子层：$y = x + MultiHeadSelfAttention(RMSNorm(x))$

前馈网络子层：$y = x + FFN(RMSNorm(x))$

#### 均方根归一化 （Root Mean Square Layer Normalization）
![](\images\CS336_Assignment1\img42.png)

![](\images\CS336_Assignment1\img43.png)

![](\images\CS336_Assignment1\img44.png)

#### Position-Wise Feed-Forward Network 位置级前馈网络
![](\images\CS336_Assignment1\img45.png)

在 transformer 的原始论文中，transformer 前馈网络由两个线性变换组成，两者之间采用 ReLU 激活函数，在这个架构中，内部前馈层的维度通常是输入维度的 4 倍，将神经网络先把输入映射到更高维的特征空间，进行非线性处理，再映射回原始维度。

![](\images\CS336_Assignment1\img46.png)

现代语言模型相较于这一原始设计引入了两个主要变化：使用了不同的激活函数，并采用了门控机制。具体来说，实现了一种名为“SwiGLU”的激活函数，该函数已被诸如 Llama 3 和 Qwen 2.5 等大语言模型（LLM）所采用。SwiGLU结合了SiLU（常被称为Swish）激活函数和一种称为门控线性单元（Gated Linear Unit, GLU）的门控机制。此外，还将省略线性层中有时使用的偏置项（bias），这是自PaLM 和 LLaMA 以来大多数现代大语言模型的做法。

首先介绍 SiLU 函数，其定义为：$SiLU(x)=x \cdot \sigma(x)=\frac{x}{1+e^{-x}}$
如上面 Figure 3 可得，SiLU 激活函数与 ReLU 函数类似，但在 0 附近平滑，不像 ReLU 在 0 处存在导数不连续。对负数不是直接清零，而是保留部分负值信息。

门控线性单元（GLU）的定义为一个经过Sigmoid函数变换的线性变换与另一个线性变换之间的逐元素乘积：$GLU(x, W_1, W_2)=\sigma(W_1x) \odot W_2x$

将 SiLU 与 GLU 函数结合起来，就是 SwiGLU 函数，我们将其用于构建前馈神经网络：

![](\images\CS336_Assignment1\img47.png)

![](\images\CS336_Assignment1\img48.png)

![](\images\CS336_Assignment1\img49.png)

#### Relative Positional Embeddings
为了将位置信息注入模型，transformer 中采取了旋转位置嵌入（Rotary Positin Embeddings, RoPE）。RoPE的核心思想是根据 token 的位置，对 Query 和 Key 向量进行旋转（不用于 Value 向量），从而将位置信息融入注意力机制。

对于每个位置 $i$ 和 一个 d 维向量 $q^{(i)}$，RoPE 将向量中相邻的两个维度划分为一组，把每组看作二维向量，并根据 token 的位置旋转相应角度。例如，维度为 4 的向量可以划分为 $(q_1,q_2)$ 和 $(q_3,q_4)$ 两组。对于其中一组向量 $(a,b)$，按照题目采用的旋转方向约定，其旋转过程为：

 $\begin{pmatrix}a'\\b'\end{pmatrix} = \begin{pmatrix} \cos\theta& -\sin\theta\\\ \sin\theta&\cos\theta \end{pmatrix} \begin{pmatrix}a\\b\end{pmatrix}$ 

因此，旋转后的分量为 $a'=a\cos\theta-b\sin\theta\\，b'=a\sin\theta+b\cos\theta$。旋转不会改变向量的长度，而是改变向量的方向，从而利用方向变化编码位置信息。
RoPE 中每对维度的旋转角度由 token 位置和维度对共同决定。标准公式为：

$\theta_{i,k}=\frac{i}{\Theta^{(2k-2)/d}}, \quad for \quad k \in \{1, ..., d/2\}$
其中 $\Theta$为常数，一般取 10000。这些所有的 2 x 2 矩阵组合成完整的 d x d 分块对角矩阵。
![](\images\CS336_Assignment1\img50.png)

![](\images\CS336_Assignment1\img51.png)

![](\images\CS336_Assignment1\img52.png)

![](\images\CS336_Assignment1\img53.png)

#### Scaled Dot-Product Attention
在学习注意力操作之前，我们学习注意力操作中使用的softmax，定义为

![](\images\CS336_Assignment1\img54.png)

当输入的值很大时，$exp(v_i)$ 会变为 $inf$（$\frac{inf}{inf}=NaN$），为了避免这种情况，我们可以利用 softmax(a) = soft(a + c) 的特性，先找到输入向量的最大值，再让所有元素减去这个最大值。这样处理后，向量中最大的元素就会变成 0，其他元素都小于或等于 0。对这些元素进行指数运算，所有的值都在 (0, 1] 之间，从而有效避免指数运算溢出。
![](\images\CS336_Assignment1\img55.png)

![](\images\CS336_Assignment1\img56.png)

![](\images\CS336_Assignment1\img57.png)

现在引出注意力操作的定义：
![](\images\CS336_Assignment1\img58.png)

掩码（Masking）：有时我们希望对注意力操作的输出进行掩码处理。掩码矩阵的形状为 $M \in \{True, False\}^{n \times m}$，其每一行 $i$ 表示第 $i$ 个查询（query）应当关注哪些键（key）。通常（但稍显反直觉地），掩码中位置 $(i, j)$ 处的值为 True 表示查询 $i$ 可以关注键 $j$，而 False 表示不能关注。换句话说，信息只在值为 True 的位置流动。

从计算角度看，使用掩码比在子序列上单独计算注意力更高效。我们可以通过在 softmax 前的注意力分数矩阵 $(QK^T/\sqrt{d_k})$，将掩码为 False 的位置加上 $-inf$ 来实现这一效果（实际实现中通常用极小的负数代替）。这样，softmax 会将这些位置的权重置为接近零，从而屏蔽对应的信息流动。

![](\images\CS336_Assignment1\img59.png)

![](\images\CS336_Assignment1\img60.png)

#### Causal Multi-Head Self-Attention
我们将实现 A.Vaswani 等人提出的多头注意力机制，从数学角度来看，定义如下：
![](\images\CS336_Assignment1\img61.png)

![](\images\CS336_Assignment1\img62.png)

**Causal masking:** 在自回归语言模型中，预测当前位置的下一个词时，模型只能关注当前位置及之前的 token，不能访问未来 token 的表示，否则未来信息会泄漏，使模型在预训练时获得推理阶段无法获取的信息，导致训练目标失去意义。理论上，可以针对序列的每个前缀分别运行多头自注意力，但这种方法会产生重复计算。因此，通常使用因果注意力掩码，一次性并行计算整个序列的注意力，同时限制位置 $i$ 只能关注满足 $j\leq i$ 的位置。可以使用 PyTorch 的 `torch.triu` 或广播式索引比较构造掩码，并将其传入已有的缩放点积注意力（Scaled Dot-Product Attention）实现，从而屏蔽未来位置，避免信息泄漏，提高计算效率。

**Applying RoPE：** RoPE 应用于 Query（Q）和 Key（K）向量，而不应用于 Value（V）向量，因为它通过对 Q 和 K 进行位置相关的旋转，使注意力分数能够体现 token 之间的位置信息，而 V 保持不变。在多头注意力机制中，每个注意力头独立计算注意力，但所有头都应使用相同的位置旋转规则。也就是说第 1 个头中位置 0 的 token，使用位置 0 对应的旋转。第 2 个头中位置 0 的 token，也使用相同的旋转。因此，实现 RoPE 时，应将 head 维度视为批次维度，对每个头的 Q 和 K 独立应用相同的 RoPE 旋转，并保持 V 不变。

![](\images\CS336_Assignment1\img63.png)

多头注意力机制大致如下图所示。假设一个 token 的特征向量维度为 $8 \ (d_{\text{model}}=8)$，如果设置 2 个注意力头（num_heads=2），那么每个头的维度就是 $d_{\text{head}}=d_{\text{model}}/num\_heads=4$。模型首先通过线性投影分别计算 Q、K、V，再将它们的 8 维特征拆分为两个 4 维的注意力头，各个头独立计算注意力。最后，将所有头的输出拼接起来，恢复为 8 维向量，并通过输出投影矩阵进行融合，得到最终的输出。

![](\images\CS336_Assignment1\img64.png)

没有 RoPE 的因果多头自注意力机制如下所示

![](\images\CS336_Assignment1\img65.png)

有 RoPE 的如下图所示：

![](\images\CS336_Assignment1\img66.png)

![](\images\CS336_Assignment1\img67.png)

### The Full Transformer LM

![](\images\CS336_Assignment1\img68.png)

![](\images\CS336_Assignment1\img69.png)

![](\images\CS336_Assignment1\img70.png)

解释一下上述代码中为什么
```python
token_positions = torch.arange(x.shape[1], device=x.device)
```

Transformer 接收的输入张量形状为：(batch_size, sequence_length, d_model)

其中：

- batch_size：一个 batch 中包含的序列数量。
- sequence_length：每条序列包含的 token 数量。
- d_model：每个 token 的特征维度。

假设一个 batch 包含两条序列：

```text
Sequence 1: 我 喜欢 学习 AI
Sequence 2: 你 喜欢 使用 Python
```

每条序列包含 4 个 token，假设 d_model = 128，则：

```python
x.shape = (2, 4, 128)

token_positions = torch.arange(x.shape[1])

# tensor([0, 1, 2, 3])
```

对应关系如下：

| 位置索引 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Sequence 1 | 我 | 喜欢 | 学习 | AI |
| Sequence 2 | 你 | 喜欢 | 使用 | Python |


<!-- 解释 token_positions:

transformer 接收的输入是 in_features: Float[Tensor, " batch sequence_length d_model"]。其中：
- batch size，一批有多少条序列。
- sequence_length：序列长度，每条序列有多少个 token。
- d_model：每个 token 的特征维度。

所以 token_positions 为每个序列分配一个位置。 -->

<!-- ![](\images\CS336_Assignment1\img71.png) -->

![](\images\CS336_Assignment1\img72.png)

![](\images\CS336_Assignment1\img73.png)

![](\images\CS336_Assignment1\img74.png)

# Training a Transformer LM
接下来将编写所有用于支持训练的代码：
- Loss: 定义损失函数 （cross-entropy）
- Optimizer: 定义优化器来最小化损失 （AdamW）
- Training loop: 所有用于加载数据、保存检查点以及管理训练的辅助基础设施。
## Cross-entropy loss
![](\images\CS336_Assignment1\img75.png)

![](\images\CS336_Assignment1\img76.png)

![](\images\CS336_Assignment1\img77.png)

![](\images\CS336_Assignment1\img78.png)

为了防止数据溢出，将交叉熵公式简化，过程如下所示：
![](\images\CS336_Assignment1\img79.png)

![](\images\CS336_Assignment1\img80.png)

## The SGD Optimizer
有了损失函数，接下来是优化器，最简单的基于梯度的优化器是随机梯度下降（SGD），我们从随机初始化的参数 $\theta_0$ 开始。然后对于每一步 $t = 0, ..., T-1$，执行以下更新：

![](\images\CS336_Assignment1\img81.png)

### Implementing SGD in Pytorch
为了实现优化器，我们将继承 pytorch 的 torch.optim.Optimizer 类，Optimizer 的子类必须实现以下两个方法：


![](\images\CS336_Assignment1\img82.png)

下面给出具体示例，但为了丰富示例，我们将实现 SGD 的一个变体，其中学习率随训练过程衰减，从一个初始学习率 $\alpha$ 开始，随着时间推移逐步缩小步长：


![](\images\CS336_Assignment1\img83.png)


![](\images\CS336_Assignment1\img84.png)

## AdamW
现代语言模型通常使用比 SGD 更复杂的优化器进行训练。AdamW 对于每个参数，会维护其一阶和二阶矩的滑动估计值，以额外的内存消耗为代价，换取了更好的训练稳定性和收敛性。
![](\images\CS336_Assignment1\img85.png)

![](\images\CS336_Assignment1\img86.png)

![](\images\CS336_Assignment1\img87.png)

![](\images\CS336_Assignment1\img88.png)

## Learning rate scheduling
在训练过程中，能使损失值下降最快的学习率值通常会发生变化。在训练 Transformer 模型时，通常会采用学习率调度策略：起初使用较大的学习率，使模型在训练初期进行更快速的更新，然后随着训练的进行，学习率逐渐衰减至较小的值。在本次作业中，我们将实现用于训练LLaMA的余弦退火调度策略 。

调度器本质上是一个函数，它接受当前步骤 t 及其他相关参数（例如初始和最终学习率），并返回在步骤 t 进行梯度更新时应使用的学习率。最简单的调度方案是常数函数，它对于任何 t 都会返回相同的学习率。
![](\images\CS336_Assignment1\img89.png)

![](\images\CS336_Assignment1\img90.png)

![](\images\CS336_Assignment1\img91.png)

## Gradient clipping
在训练过程中，我们有时会遇到会产生较大梯度的训练样本，这可能会导致训练不稳定。为缓解这一问题，实践中常采用的一种技术是梯度剪切。其原理是在每次反向传播后，在执行优化器步骤之前，对梯度的范数施加限制。

![](\images\CS336_Assignment1\img92.png)


![](\images\CS336_Assignment1\img93.png)


![](\images\CS336_Assignment1\img94.png)

# Training loop
## Data Loader
分词后的数据（例如你在 tokenizer_experiments 中准备的数据）是一个单一的令牌序列 x = (x₁, ..., xₙ)。尽管原始数据可能由多个独立的文档组成（例如不同的网页或源代码文件），一种常见的做法是将所有这些文档连接成一个单一的令牌序列，并在它们之间添加一个分隔符（例如 token）。

数据加载器会将这个长序列转换为一批批的数据流，其中每个批次包含 B 个长度为 m 的序列，以及对应的下一个令牌序列（长度也为 m）。例如，当 B = 1，m = 3 时，([x₂, x₃, x₄], [x₃, x₄, x₅]) 就是一个可能的批次。

以这种方式加载数据在多个方面简化了训练过程。首先，任何满足 1 ≤ i < n − m 的位置 i 都可以生成一个有效的训练序列，因此序列的采样非常简单。由于所有训练序列长度相同，无需进行填充操作，这提高了硬件利用率（同时也可以增大批次大小 B）。最后，我们也不必一次性将整个数据集全部加载到内存中即可进行训练样本的抽取，从而更容易处理那些可能无法完全放入内存的大型数据集。

![](\images\CS336_Assignment1\img95.png)

![](\images\CS336_Assignment1\img96.png)

## Checkpointing

除了加载数据外，我们还需要在训练过程中保存模型。在运行训练任务时，我们常常希望能够在训练中途意外停止后（例如由于任务超时、机器故障等）恢复训练。即使一切顺利，我们也可能希望之后能够访问训练过程中的中间模型（例如，事后研究训练动态、从不同训练阶段的模型中生成样本等）。

一个检查点（checkpoint）应包含所有能够恢复训练所需的状态。最基本的是，我们必须能够恢复模型的权重。如果使用了带有状态的优化器（例如 AdamW），我们还需要保存优化器的状态（例如，AdamW 中的动量估计值）。最后，为了能够恢复学习率调度，我们还需要知道训练停止时所处的迭代次数。PyTorch 使得保存这些信息变得非常简单：每个 nn.Module 都有一个 state_dict() 方法，它会返回一个包含所有可学习参数的字典；之后我们可以通过对应的 load_state_dict() 方法恢复这些权重。优化器（torch.optim.Optimizer）也同样支持 state_dict() 和 load_state_dict() 方法。最后，torch.save(obj, dest) 可以将一个对象（例如，字典，其值中包含张量，也可以包含整数等普通 Python 对象）序列化并保存到文件（路径）或类文件对象中，之后可以通过 torch.load(src) 将其重新加载回内存。


![](\images\CS336_Assignment1\img97.png)

![](\images\CS336_Assignment1\img98.png)

<!-- ## Training loop

![](\images\CS336_Assignment1\img99.png)
![[Pasted image 20261013185431.png]]  -->