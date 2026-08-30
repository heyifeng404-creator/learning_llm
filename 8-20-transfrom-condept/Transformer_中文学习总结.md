# Transformer 中文学习总结：从 QKV 到 GPT

> 面向初学者的复习笔记。先建立直觉，再理解公式与工程优化。

## 目录

1. [Transformer 总览](#1-transformer-总览)
2. [Tokenization：把文字切成模型的输入单位](#2-tokenization把文字切成模型的输入单位)
3. [Embedding：把 Token 变成向量](#3-embedding把-token-变成向量)
4. [位置编码与 RoPE](#4-位置编码与-rope)
5. [Q、K、V 到底是什么](#5-qkv-到底是什么)
6. [Self-Attention：让 Token 交换信息](#6-self-attention让-token-交换信息)
7. [Scaled Dot-Product Attention 公式](#7-scaled-dot-product-attention-公式)
8. [手算一次 Attention](#8-手算一次-attention)
9. [Softmax：把分数变成权重](#9-softmax把分数变成权重)
10. [Multi-Head Attention](#10-multi-head-attention)
11. [FFN：每个 Token 独立加工信息](#11-ffn每个-token-独立加工信息)
12. [Residual、LayerNorm 与 RMSNorm](#12-residuallayernorm-与-rmsnorm)
13. [一个 Transformer Block](#13-一个-transformer-block)
14. [GPT：Decoder-only Transformer](#14-gptdecoder-only-transformer)
15. [Causal Mask：不能偷看未来](#15-causal-mask不能偷看未来)
16. [Next-token Prediction](#16-next-token-prediction)
17. [训练流程](#17-训练流程)
18. [推理与 KV Cache](#18-推理与-kv-cache)
19. [MHA、GQA 与 MQA](#19-mhagqa-与-mqa)
20. [时间与空间复杂度](#20-时间与空间复杂度)
21. [完整流程图](#21-完整流程图)
22. [推荐学习路线](#22-推荐学习路线)
23. [一页速记](#23-一页速记)

---

## 1. Transformer 总览

Transformer 的核心任务可以概括为：

> 让每个 Token 根据上下文更新自己的表示，从而理解当前语境中的真正含义。

例如：

> 我今天去**银行**取钱。

“银行”仅凭自身可能有歧义；看到“取钱”后，模型更容易判断这里指金融机构。Transformer 用 **Attention（注意力）** 建立这种上下文联系。

经典 Transformer 最初包含 Encoder 和 Decoder：

```text
输入 → Encoder（理解输入）→ Decoder（生成输出）→ 输出
```

现代常见架构：

| 架构 | 代表模型 | 更适合做什么 |
|---|---|---|
| Encoder-only | BERT | 文本理解、分类、抽取 |
| Decoder-only | GPT、LLaMA、Qwen | 自回归文本生成 |
| Encoder–Decoder | T5、原始 Transformer | 翻译、摘要、输入到输出的转换 |

本文重点讲 GPT 使用的 **Decoder-only Transformer**。

---

## 2. Tokenization：把文字切成模型的输入单位

模型不能直接处理字符串，首先要由 Tokenizer 把文本切成 Token，再映射成整数 ID。

```text
“我喜欢人工智能”
        ↓ Tokenizer
[“我”, “喜欢”, “人工”, “智能”]
        ↓ 查词表
[100, 532, 8291, 2048]
```

需要记住：

- Token 不一定是一个字，也不一定是完整单词。
- 生僻词可能被拆成多个子词；标点和空格也可能成为 Token。
- 模型看到的是 Token ID，不是原始文字。
- 同一句文本在不同 Tokenizer 中可能得到不同切分结果。

Tokenization 会影响序列长度、计算成本和模型处理不同语言的效率。

---

## 3. Embedding：把 Token 变成向量

Token ID 只是编号，本身没有“大小代表语义”的含义。Embedding 层通过查表，将每个 ID 映射为一个可学习的高维向量：

$$
X = E[\text{token\_ids}]
$$

若序列长度为 $n$，模型维度为 $d_{model}$，则：

$$
X \in \mathbb{R}^{n \times d_{model}}
$$

例如一个 Token 可以被表示为：

```text
“苹果” → [0.12, -0.31, 0.77, ..., 0.44]
```

直觉上，Embedding 是把文字翻译成模型能计算的“数字语言”。训练后，相似语境中的 Token 往往具有某些相似的表示特征。

注意：初始 Embedding 只是起点。经过多层 Transformer Block 后，同一个词在不同句子里会得到不同的上下文表示。

---

## 4. 位置编码与 RoPE

Attention 本身只比较向量，不天然知道 Token 的顺序。若没有位置信息，“我爱你”和“你爱我”就很难区分。

常见处理方式包括：

- **绝对位置编码**：为第 1、2、3……个位置添加不同向量。
- **可学习位置 Embedding**：位置向量也通过训练得到。
- **RoPE（Rotary Position Embedding，旋转位置编码）**：根据位置旋转 Q、K 向量，使它们的点积自然包含相对位置信息。

RoPE 的直觉不是简单贴一个“第几个”的标签，而是：

> 让 Q、K 随位置发生有规律的旋转，因此两个 Token 的匹配分数会感知它们之间的相对距离。

RoPE 通常作用于 **Q 和 K**，而不是直接加在 V 上。现代 GPT 类模型常使用 RoPE 或其变体。

---

## 5. Q、K、V 到底是什么

每个输入向量都会经过三组不同的线性变换：

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V
$$

其中 $W_Q$、$W_K$、$W_V$ 都是训练得到的参数。

可以用“搜索”来记忆：

| 名称 | 英文 | 直觉问题 | 搜索类比 |
|---|---|---|---|
| Q | Query | 我想找什么？ | 用户输入的搜索词 |
| K | Key | 我能否与这个需求匹配？ | 每篇文章的关键词或索引 |
| V | Value | 匹配后真正提供什么内容？ | 文章的实际内容 |

一句话记忆：

> 用 Q 找最匹配的 K，再按匹配程度取走对应的 V。

例如处理“小明吃了苹果，因为它很甜”中的“它”时：

- “它”的 Q 与所有 Token 的 K 比较；
- 若它与“苹果”的 K 得分最高；
- 模型就更多读取“苹果”的 V，并融入“它”的新表示。

Q、K、V 不是三个固定词库，也不是人工编写的语义标签；它们都是输入向量经过不同投影后得到的中间表示。

---

## 6. Self-Attention：让 Token 交换信息

Self-Attention 中，Q、K、V 都来自同一段输入，所以叫“自注意力”。它让每个 Token：

1. 用自己的 Q 向所有位置提问；
2. 与所有 K 计算相关性；
3. 把相关性转成权重；
4. 按权重汇总所有 V；
5. 得到融合上下文后的新表示。

例如：

```text
“它”的注意力：
小明  0.05
吃了  0.05
苹果  0.75
因为  0.03
很甜  0.12
```

输出不是简单选择“苹果”一个位置，而是所有 V 的**加权和**。权重较高的位置贡献更大。

---

## 7. Scaled Dot-Product Attention 公式

核心公式为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

各部分含义：

| 部分 | 含义 |
|---|---|
| $QK^T$ | 每个 Query 与每个 Key 的点积，得到相关性分数 |
| $\sqrt{d_k}$ | 缩放因子，避免维度大时点积过大 |
| $M$ | 可选 Mask；GPT 中用它屏蔽未来位置 |
| Softmax | 把每一行分数转成总和为 1 的权重 |
| 乘 $V$ | 按权重汇总真正的信息 |

若：

$$
Q,K\in\mathbb{R}^{n\times d_k},\qquad
V\in\mathbb{R}^{n\times d_v}
$$

则：

$$
QK^T\in\mathbb{R}^{n\times n}
$$

这个 $n\times n$ 矩阵表示序列中每个位置对每个位置的注意力分数。

### 为什么除以 $\sqrt{d_k}$？

维度越大，点积的数值通常越容易变大。过大的分数会让 Softmax 过早接近 0 或 1，梯度变小、训练不稳定。除以 $\sqrt{d_k}$ 可以把分数控制在更合适的范围。

---

## 8. 手算一次 Attention

为了看清全过程，设有两个 Token A、B，并直接给出一组简单的 Q、K、V：

$$
Q=K=V=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},\qquad d_k=2
$$

### 第一步：计算相似度 $QK^T$

$$
QK^T=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

解释：A 与自己的匹配分数为 1，与 B 为 0；B 同理。

### 第二步：缩放

因为 $\sqrt{d_k}=\sqrt{2}\approx1.414$：

$$
S=\frac{QK^T}{\sqrt{2}}
\approx
\begin{bmatrix}
0.707&0\\
0&0.707
\end{bmatrix}
$$

### 第三步：逐行做 Softmax

第一行：

$$
\operatorname{softmax}([0.707,0])
=\left[
\frac{e^{0.707}}{e^{0.707}+e^0},
\frac{e^0}{e^{0.707}+e^0}
\right]
\approx[0.67,0.33]
$$

第二行同理：

$$
A\approx
\begin{bmatrix}
0.67&0.33\\
0.33&0.67
\end{bmatrix}
$$

### 第四步：权重乘 V

$$
O=AV
=
\begin{bmatrix}
0.67&0.33\\
0.33&0.67
\end{bmatrix}
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
=
\begin{bmatrix}
0.67&0.33\\
0.33&0.67
\end{bmatrix}
$$

因此：

- A 的新表示为 $0.67V_A+0.33V_B$；
- B 的新表示为 $0.33V_A+0.67V_B$。

这就是 Attention 的本质：**先算“该关注谁”，再按权重混合信息。**

> 在 GPT 中还会加入 Causal Mask。此时第一个 Token 不能看第二个 Token，因此第一行的权重会变成 $[1,0]$；第二个 Token 可以看自己和前面，第二行保持正常计算。

---

## 9. Softmax：把分数变成权重

对一组分数 $z_1,\dots,z_n$，Softmax 定义为：

$$
\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

它有三个重要效果：

- 每个结果都大于 0；
- 所有结果之和等于 1；
- 较大的原始分数会获得更大的相对权重。

注意力中的 Softmax 是对分数矩阵的**每一行**分别计算：每一行代表一个 Query 对所有 Key 的关注分布。

Softmax 不等于“绝对概率真相”，更像是在当前候选集合中的相对分配。温度参数还能控制分布的尖锐或平缓程度。

---

## 10. Multi-Head Attention

单头注意力只有一套投影，表达视角有限。多头注意力使用多组 $W_Q,W_K,W_V$ 并行计算：

$$
\text{head}_i=\operatorname{Attention}(Q_i,K_i,V_i)
$$

$$
\operatorname{MHA}(X)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)W_O
$$

不同 Head 可以学习不同关系，例如语法、指代、距离或语义联系。不过“某个头一定负责某种功能”只是便于理解的类比，不是人工规定。

典型情况下：

$$
d_{head}=\frac{d_{model}}{h}
$$

多个 Head 的结果先拼接，再经输出矩阵 $W_O$ 投影回模型维度。

---

## 11. FFN：每个 Token 独立加工信息

Attention 负责 Token 之间交换信息；FFN（Feed-Forward Network）负责对每个 Token 的表示做更深的非线性加工。

经典形式：

$$
\operatorname{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2
$$

通常先把维度扩大，再压回 $d_{model}$。现代模型常使用 GELU、SiLU、SwiGLU 等激活或门控结构。

关键区别：

- Attention 在序列维度上混合不同 Token 的信息；
- FFN 对每个 Token 独立使用同一套参数，不直接混合不同位置。

好记的比喻：

> Attention 是“和别人交流”，FFN 是“自己消化和思考”。

---

## 12. Residual、LayerNorm 与 RMSNorm

### 12.1 Residual Connection（残差连接）

残差连接把子层的输入直接加回输出：

$$
y=x+F(x)
$$

它让模型可以在原信息上做增量修改，有助于梯度传播和深层网络训练。

### 12.2 LayerNorm

LayerNorm 对单个 Token 的特征维度做归一化，再使用可学习参数缩放和平移：

$$
\operatorname{LayerNorm}(x)
=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

它能稳定数值分布，使训练更可靠。

### 12.3 RMSNorm

RMSNorm 是现代大模型常见的简化归一化方法。它不减均值，主要按均方根缩放：

$$
\operatorname{RMSNorm}(x)
=\gamma\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
$$

二者区别可简记为：

| 方法 | 减均值 | 按方差/均方根缩放 | 常见场景 |
|---|---:|---:|---|
| LayerNorm | 是 | 是 | 经典 Transformer、GPT-2 等 |
| RMSNorm | 否 | 是 | LLaMA、许多现代大模型 |

### 12.4 Pre-Norm 与 Post-Norm

- **Post-Norm**：先子层、再残差、再 Norm；原始 Transformer 使用这种形式。
- **Pre-Norm**：先 Norm、再子层、再残差；许多现代大模型采用，深层训练通常更稳定。

---

## 13. 一个 Transformer Block

现代 GPT 类模型常见的 Pre-Norm Block 可写为：

$$
H'=H+\operatorname{Attention}(\operatorname{Norm}(H))
$$

$$
H''=H'+\operatorname{FFN}(\operatorname{Norm}(H'))
$$

结构示意：

```text
输入 H
  │
  ├───────────────┐
  ↓               │
Norm              │
  ↓               │
Self-Attention    │
  ↓               │
  + ←─────────────┘  残差
  │ H'
  ├───────────────┐
  ↓               │
Norm              │
  ↓               │
FFN               │
  ↓               │
  + ←─────────────┘  残差
  │
输出 H''
```

一个大模型会堆叠几十甚至上百个 Block。浅层可能学习较局部的模式，深层逐步组合成更抽象的上下文表示；但不能简单断言每层只负责某一种固定功能。

---

## 14. GPT：Decoder-only Transformer

GPT 只保留适合自回归生成的 Decoder 风格主体，可概括为：

```text
文本
 ↓
Tokenizer → Token IDs
 ↓
Token Embedding
 ↓
位置处理（如 RoPE 作用于各层 Q、K）
 ↓
Transformer Block × N
 ↓
Final Norm
 ↓
Linear / LM Head
 ↓
词表中每个 Token 的 logits
 ↓
概率分布与采样
 ↓
下一个 Token
```

“Decoder-only”不表示它只能机械解码；它仍通过多层因果自注意力和 FFN 建立复杂的上下文表示。这里的核心限制是：每个位置只能使用它之前及自身的信息。

---

## 15. Causal Mask：不能偷看未来

训练 GPT 时，整段文本可并行送入模型，但位置 $i$ 不能使用位置 $i+1$ 及之后的信息，否则预测下一个 Token 时就泄露答案。

四个 Token 的可见关系如下：

```text
Query \ Key   1   2   3   4
1             ✓   ×   ×   ×
2             ✓   ✓   ×   ×
3             ✓   ✓   ✓   ×
4             ✓   ✓   ✓   ✓
```

实现时，会在未来位置的注意力分数上加一个极小值（概念上为 $-\infty$）：

$$
M_{ij}=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

经过 Softmax 后，被屏蔽位置的权重变为 0。

---

## 16. Next-token Prediction

GPT 的基本任务是：已知前面的 Token，预测下一个 Token。

$$
P(x_t\mid x_1,x_2,\dots,x_{t-1})
$$

整段文本的概率可分解为：

$$
P(x_1,\dots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

例如输入“中国的首都是”，模型可能输出：

```text
北京  0.92
上海  0.03
广州  0.01
其他  0.04
```

选出“北京”后，把它追加到上下文，再预测下一个 Token。生成不是一次完成整篇文章，而是不断重复：

```text
读取已有上下文 → 预测一个 Token → 追加 → 再预测
```

最终选择不一定总取概率最大项，还可使用温度、top-k、top-p 等采样策略控制稳定性和多样性。

---

## 17. 训练流程

### 17.1 构造输入与标签

以 Token 序列为例：

```text
序列： [中国, 的, 首都, 是, 北京]
输入： [中国, 的, 首都, 是]
标签： [的, 首都, 是, 北京]
```

标签相对输入左移一位，这种方式叫 **teacher forcing**：训练时每个位置看到的前文是真实 Token，而不是模型刚刚预测的 Token。

### 17.2 前向传播

输入依次经过 Embedding、多个 Transformer Block 和 LM Head，得到每个位置对整个词表的 logits。

### 17.3 计算损失

通常使用交叉熵损失，使正确下一个 Token 的概率变高：

$$
\mathcal{L}=-\sum_t \log P(x_t\mid x_{<t})
$$

Padding 或不需要学习的位置可通过 loss mask 排除。

### 17.4 反向传播与更新

```text
Loss
 ↓
Backpropagation（计算梯度）
 ↓
Optimizer（如 AdamW）
 ↓
更新 Embedding、QKV、FFN 等参数
```

预训练通常学习广泛的语言与知识模式；之后还可能进行指令微调、偏好优化等阶段，让模型更会遵循指令并符合期望行为。

训练时可并行计算一整段序列的位置；生成时则有前后依赖，通常必须逐 Token 进行。

---

## 18. 推理与 KV Cache

假设模型已经处理：

```text
我 喜欢 学习 人工智能
```

生成下一个 Token 时，旧 Token 在每一层的 K 和 V 不会改变。如果每次都重新计算它们，会浪费大量时间。KV Cache 就是在每层保存历史 Token 的 K、V。

下一步生成时：

- 只为新 Token 计算新的 Q、K、V；
- 把新 K、V 追加到 Cache；
- 用新 Q 查询全部历史 K；
- 对历史 V 加权求和。

### Prefill 与 Decode

- **Prefill（提示词处理）**：一次并行处理整个输入提示，建立 KV Cache。
- **Decode（逐 Token 解码）**：每步处理一个新 Token，并读取、追加 Cache。

KV Cache 用显存换速度。其占用通常随以下因素近似线性增长：

- 层数；
- 上下文长度；
- Batch 大小；
- K/V Head 数；
- 每个 Head 的维度；
- 数据精度。

因此，上下文越长，KV Cache 往往越大。

---

## 19. MHA、GQA 与 MQA

这些方法主要区别在于多少个 Query Head 共享 K/V Head。

| 方法 | Query Heads | K/V Heads | 特点 |
|---|---:|---:|---|
| MHA | $h$ | $h$ | 每个 Q Head 有自己的 K/V，表达力强，Cache 较大 |
| GQA | $h$ | 少于 $h$、多于 1 | 一组 Q Heads 共享一组 K/V，质量与效率折中 |
| MQA | $h$ | 1 | 所有 Q Heads 共享一组 K/V，Cache 最小、读取更省 |

例如 32 个 Query Heads、8 个 K/V Heads，就是 GQA：平均每 4 个 Query Heads 共用一组 K/V。

GQA/MQA 的主要价值是减少 KV Cache 大小和解码阶段的内存带宽压力，从而提高推理效率。它们通常不把 Q Head 数也一起压到 1。

---

## 20. 时间与空间复杂度

设序列长度为 $n$，模型维度为 $d$。

### 20.1 标准 Attention

- 计算 $QK^T$：约 $O(n^2d)$；
- 注意力矩阵空间：约 $O(n^2)$；
- QKV 和输出投影：约 $O(nd^2)$。

因此常说 Attention 对序列长度具有 **平方复杂度 $O(n^2)$**。这句话主要强调注意力矩阵随 $n$ 平方增长，不代表整个 Block 在所有模型尺寸和序列长度下都只由这一项主导。

### 20.2 FFN

FFN 的主要计算量约为 $O(ndd_{ff})$；若 $d_{ff}$ 与 $d$ 成固定倍数，可粗略记为 $O(nd^2)$。

### 20.3 训练与生成的差异

- 训练/Prefill：多个位置可以并行，但长序列的注意力矩阵成本高。
- 自回归 Decode：使用 KV Cache 后不重算全部历史 K/V；单个新 Token 仍要关注所有历史位置，因此每一步成本随当前上下文长度增长。

### 20.4 常见优化

- **FlashAttention**：通过更高效的数据分块与内存访问精确计算标准 Attention，主要降低显存读写和中间内存，并非简单删掉注意力关系。
- **Sliding Window / Local Attention**：只关注一定窗口，降低长序列成本。
- **Sparse Attention**：只计算部分位置关系。
- **GQA/MQA**：减少 K/V Head 和 KV Cache。
- **量化**：用更低精度存储或计算参数、Cache。

---

## 21. 完整流程图

```text
原始文本：“我喜欢人工智能”
            │
            ▼
       Tokenization
            │
            ▼
     Token IDs：[...]
            │
            ▼
     Token Embedding
            │
            ▼
┌───────────────────────────┐
│ Transformer Block × N     │
│                           │
│ Norm                      │
│   ↓                       │
│ 生成 Q、K、V              │
│   ↓                       │
│ RoPE 作用于 Q、K          │
│   ↓                       │
│ Causal Self-Attention     │
│   ↓                       │
│ Residual                  │
│   ↓                       │
│ Norm → FFN → Residual     │
└───────────────────────────┘
            │
            ▼
        Final Norm
            │
            ▼
      LM Head / Linear
            │
            ▼
    全词表 logits → 概率
            │
            ▼
     选择下一个 Token
            │
            ▼
   追加到上下文并重复生成
```

可以用一个“会议室”比喻记住 Block：

- Q：我需要什么信息？
- K：我能提供什么线索？
- V：我的实际内容是什么？
- Attention：大家互相交流；
- Multi-Head：从多个投影视角交流；
- FFN：每个人独立消化信息；
- Residual：保留原观点，在此基础上修改；
- Norm：稳定每轮讨论的数值尺度；
- 多层 Block：反复交流和加工。

---

## 22. 推荐学习路线

### 第一阶段：建立直觉

1. Token 与 Tokenizer
2. Embedding 与向量
3. 点积、矩阵乘法
4. Softmax
5. Q、K、V
6. Self-Attention

目标：能用自己的话解释“Q 找 K，按权重汇总 V”。

### 第二阶段：读懂一个 Block

1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. FFN 与激活函数
4. Residual
5. LayerNorm / RMSNorm
6. Pre-Norm
7. 位置编码 / RoPE

目标：能画出 Block，写出输入输出张量的大致形状。

### 第三阶段：理解 GPT

1. Decoder-only
2. Causal Mask
3. LM Head
4. Next-token Prediction
5. 交叉熵与训练流程
6. Temperature、top-k、top-p

目标：能完整讲明“训练时并行、生成时逐 Token”的原因。

### 第四阶段：理解推理优化

1. Prefill 与 Decode
2. KV Cache
3. MHA / GQA / MQA
4. FlashAttention
5. 量化
6. 长上下文与显存估算

目标：能解释模型参数显存之外，为什么长上下文还会消耗大量显存。

### 第五阶段：继续深入

- SwiGLU 等现代 FFN；
- MoE（Mixture of Experts）；
- 推测解码、连续批处理；
- 分布式训练与推理；
- 阅读并实现一个迷你 GPT。

---

## 23. 一页速记

### 最核心的三个公式

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

$$
P(x_1,\dots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

### 最重要的直觉

- Tokenization：把文本切成模型处理的单位。
- Embedding：把 Token ID 变成向量。
- RoPE：让 Q、K 的匹配感知相对位置。
- Q：我想找什么。
- K：我能否与需求匹配。
- V：我真正提供的信息。
- Attention：Token 之间交流信息。
- Softmax：把相关性分数转成相对权重。
- Multi-Head：在多个投影子空间并行建模关系。
- FFN：对每个 Token 独立做非线性加工。
- Residual：保留输入并学习增量。
- Norm：稳定数值与训练。
- Causal Mask：禁止看到未来 Token。
- GPT：堆叠 Decoder 风格 Block，不断预测下一个 Token。
- KV Cache：缓存历史 K/V，加速逐 Token 生成。
- GQA/MQA：共享 K/V Heads，减少 Cache 与内存带宽压力。

### 最值得记住的一句话

> **Transformer 让每个 Token 通过 Attention 从上下文收集信息，再用 FFN 加工信息，并把这个过程重复很多层；GPT 在此基础上使用因果掩码，不断预测下一个 Token。**

---

## 复习自测

如果能不看答案解释下面问题，就说明已经掌握主线：

1. Token 为什么不一定等于一个字或一个单词？
2. 为什么 Attention 需要位置信息？
3. Q、K、V 分别解决什么问题？
4. 为什么 $QK^T$ 得到的是 $n\times n$ 矩阵？
5. 为什么点积要除以 $\sqrt{d_k}$？
6. Softmax 在 Attention 中沿哪个维度计算？
7. Multi-Head 与单头的区别是什么？
8. Attention 与 FFN 的分工是什么？
9. Residual 和 Norm 分别解决什么问题？
10. Causal Mask 为什么不会妨碍训练时并行计算？
11. GPT 的训练标签为什么要相对输入移动一位？
12. KV Cache 缓存什么，为什么只缓存 K/V？
13. GQA/MQA 如何降低推理成本？
14. 为什么说标准 Attention 对序列长度是平方复杂度？

