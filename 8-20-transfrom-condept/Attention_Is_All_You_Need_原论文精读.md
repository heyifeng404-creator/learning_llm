# 《Attention Is All You Need》原论文精读与研究分析

> 论文：Ashish Vaswani et al., *Attention Is All You Need*, NeurIPS 2017  
> 官方来源：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) ｜ [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
> 阅读视角：计算机研究生 / 大模型方向  
> 说明：本文严格区分“2017 年原论文中的 Transformer”与“后来 GPT、LLaMA 等模型的改造”。

---

## 0. 先给出论文的核心判断

这篇论文真正的贡献，不只是提出了一个 Attention 公式。Scaled Dot-Product Attention、Encoder–Decoder Attention 和 Self-Attention 的相关思想都有前置工作。它的关键突破是：

> 把 Self-Attention 提升为序列建模的主干计算单元，完全移除按时间步递归的 RNN 和用于混合位置的卷积，并构造出可大规模并行训练的完整 Encoder–Decoder 系统。

论文的论证链条是：

1. RNN 的时间步依赖限制训练并行性；
2. 卷积可并行，但建立远距离依赖需要多层传播；
3. 全局 Self-Attention 可在一层内直接连接任意两个位置；
4. Multi-Head 缓解单次加权平均造成的表达瓶颈；
5. 位置编码补回 Attention 本身缺失的顺序信息；
6. Encoder–Decoder、残差、归一化、FFN 等组件共同组成可训练系统；
7. 翻译实验表明该结构兼顾质量、训练速度和可扩展性。

从今天的大模型视角看，最深远的贡献是建立了一个高度并行、易于扩大数据和参数规模的通用序列计算骨架。

---

## 1. 论文问题背景

### 1.1 2017 年之前的主流路线

神经机器翻译通常采用 Seq2Seq Encoder–Decoder：

- Encoder 把源语言序列压成连续表示；
- Decoder 根据 Encoder 表示和已生成目标词逐步生成翻译；
- RNN、LSTM、GRU 是主要序列建模单元；
- Attention 通常作为 RNN 主干之外的辅助对齐机制。

RNN 的状态更新形式可抽象为：

$$
h_t=f(h_{t-1},x_t)
$$

计算 $h_t$ 必须先得到 $h_{t-1}$，所以单个样本内部难以跨时间步并行。序列越长，顺序计算链越长。

### 1.2 论文试图解决的矛盾

研究目标不是简单追求“不要 RNN”，而是同时满足：

- 更少的顺序计算；
- 任意位置间更短的信息路径；
- 足够强的表示能力；
- 训练效率和翻译质量不下降。

卷积模型能够并行，但固定卷积核只能看到局部邻域。连接距离为 $n$ 的两个位置，普通卷积通常需要 $O(n/k)$ 层路径，扩张卷积约需 $O(\log_k n)$。全局 Self-Attention 则让任意两个位置在单层中直接交互。

---

## 2. 整体架构：原论文不是 GPT

原论文 Transformer 是一个完整的 Encoder–Decoder 翻译模型：

```text
源语言 Token
  → Embedding + Positional Encoding
  → Encoder × 6
  → 编码记忆 Z

右移后的目标语言 Token
  → Embedding + Positional Encoding
  → Decoder × 6（Masked Self-Attention + Cross-Attention + FFN）
  → Linear + Softmax
  → 下一个目标 Token
```

Base 模型主要配置：

| 超参数 | 原论文 Base |
|---|---:|
| Encoder 层数 $N$ | 6 |
| Decoder 层数 $N$ | 6 |
| 模型维度 $d_{model}$ | 512 |
| FFN 隐层维度 $d_{ff}$ | 2048 |
| Head 数 $h$ | 8 |
| 每头 $d_k=d_v$ | 64 |
| Dropout | 0.1 |
| Label smoothing $\epsilon_{ls}$ | 0.1 |
| 参数量 | 约 65M |

Transformer-big 使用 $d_{model}=1024$、$d_{ff}=4096$、16 个 Head，约 213M 参数。

### 容易混淆的历史差异

| 组件 | 2017 原论文 | 现代 GPT/LLaMA 常见形式 |
|---|---|---|
| 主体 | Encoder–Decoder | Decoder-only |
| Norm 顺序 | Post-Norm | 多为 Pre-Norm |
| Norm 类型 | LayerNorm | LayerNorm 或 RMSNorm |
| 位置 | 正弦/余弦绝对位置编码 | Learned position、RoPE 等 |
| FFN | ReLU，两层线性 | GELU、SwiGLU 等 |
| 注意力 | MHA | MHA、MQA 或 GQA |
| 训练任务 | 监督机器翻译 | 大规模自回归语言建模为主 |

---

## 3. Encoder 逐层分析

Encoder 由 6 个相同结构但参数不共享的层堆叠而成。每层有两个子层：

1. Multi-Head Self-Attention；
2. Position-wise FFN。

原论文采用 Post-Norm：

$$
\operatorname{Output}=\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))
$$

因此 Encoder 层为：

$$
H_1=\operatorname{LN}(X+\operatorname{MHA}(X,X,X))
$$

$$
H_2=\operatorname{LN}(H_1+\operatorname{FFN}(H_1))
$$

Self-Attention 中 Q、K、V 全部来自上一层 Encoder 输出。每个源语言位置都可以读取所有源语言位置，不使用因果 Mask；实际批处理时通常仍需 Padding Mask。

### Encoder 输出的研究含义

Encoder 输出不是单个句向量，而是长度不变的一组上下文化表示：

$$
Z=(z_1,z_2,\dots,z_n),\qquad Z\in\mathbb{R}^{n\times d_{model}}
$$

这组表示同时承担 Cross-Attention 中的 K 和 V，供 Decoder 在每一步查询。

---

## 4. Decoder 逐层分析

Decoder 同样堆叠 6 层，每层有三个子层：

1. Masked Multi-Head Self-Attention；
2. Encoder–Decoder Multi-Head Attention（Cross-Attention）；
3. Position-wise FFN。

其计算可写为：

$$
S_1=\operatorname{LN}(Y+\operatorname{MaskedMHA}(Y,Y,Y))
$$

$$
S_2=\operatorname{LN}(S_1+\operatorname{MHA}(Q=S_1,K=Z,V=Z))
$$

$$
S_3=\operatorname{LN}(S_2+\operatorname{FFN}(S_2))
$$

### 三种注意力不能混为一谈

| 注意力 | Q 来源 | K/V 来源 | 可见范围 | 功能 |
|---|---|---|---|---|
| Encoder Self-Attention | Encoder | Encoder | 全部源位置 | 理解源序列内部关系 |
| Decoder Masked Self-Attention | Decoder | Decoder | 当前及过去目标位置 | 建模已生成目标前缀 |
| Cross-Attention | Decoder | Encoder 输出 | 全部源位置 | 根据当前生成状态读取源句 |

Cross-Attention 是原始 Encoder–Decoder Transformer 与 GPT Decoder-only 的重要区别。GPT 的标准 Block 不含 Encoder Cross-Attention。

---

## 5. Scaled Dot-Product Attention

论文公式：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

若加入 Mask，可写成：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

### 5.1 维度推导

设 Query 数为 $n_q$，Key/Value 数为 $n_k$：

$$
Q\in\mathbb{R}^{n_q\times d_k},\quad
K\in\mathbb{R}^{n_k\times d_k},\quad
V\in\mathbb{R}^{n_k\times d_v}
$$

则：

$$
QK^T\in\mathbb{R}^{n_q\times n_k}
$$

$$
\operatorname{softmax}(QK^T/\sqrt{d_k})V
\in\mathbb{R}^{n_q\times d_v}
$$

Self-Attention 中通常 $n_q=n_k=n$；Cross-Attention 中目标长度与源长度可以不同。

### 5.2 为什么使用点积

与加性 Attention 相比，点积 Attention 能直接使用高度优化的矩阵乘法，实际速度和空间效率更好。Q/K 点积是一个可学习投影空间内的兼容性函数，不应简单等同于原始词向量余弦相似度。

### 5.3 为什么除以 $\sqrt{d_k}$

论文给出的统计解释是：假设 $q_i,k_i$ 独立、均值为 0、方差为 1，则

$$
q\cdot k=\sum_{i=1}^{d_k}q_ik_i
$$

的方差为 $d_k$，标准差随 $\sqrt{d_k}$ 增长。维度较大时，未缩放点积会把 Softmax 推入饱和区，使梯度很小。除以 $\sqrt{d_k}$ 后，分数尺度更稳定。

### 5.4 Attention 输出是什么

每个 Query 的输出是所有 Value 的凸组合：

$$
o_i=\sum_j \alpha_{ij}v_j,
\qquad \alpha_{ij}\ge 0,
\qquad \sum_j\alpha_{ij}=1
$$

这带来强大的内容寻址能力，也意味着单头输出具有“加权平均”的信息压缩特征。Multi-Head 正是用于在多个子空间中并行保留不同关系。

---

## 6. Multi-Head Attention

第 $i$ 个 Head：

$$
\operatorname{head}_i=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

整体输出：

$$
\operatorname{MultiHead}(Q,K,V)
=\operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)W^O
$$

投影矩阵：

$$
W_i^Q,W_i^K\in\mathbb{R}^{d_{model}\times d_k},\quad
W_i^V\in\mathbb{R}^{d_{model}\times d_v},\quad
W^O\in\mathbb{R}^{hd_v\times d_{model}}
$$

Base 模型中 $h=8$，$d_k=d_v=64$，所以拼接后仍为 $8\times64=512=d_{model}$。

### 为什么多头总计算量仍可控

单头若直接使用 512 维，主要注意力乘法成本与 8 个 64 维 Head 的总和在同一量级。多头不是免费，但通过缩小每头维度，论文使其总成本接近全维单头，同时获得多个投影子空间。

### 消融结果告诉了什么

论文 Table 3 显示：

- 单头比 Base 的 8 头低约 0.9 BLEU；
- 16 头与 8 头接近；
- 32 头反而下降；
- 过小的 $d_k$ 会损害效果。

因此，“Head 越多越好”不成立。Head 数、每头维度和优化难度之间存在折中。

---

## 7. Masked Self-Attention 与输出右移

Decoder 需要满足自回归分解：

$$
P(y_1,\dots,y_m\mid x)
=\prod_{t=1}^{m}P(y_t\mid y_{<t},x)
$$

训练时目标序列整体并行输入，但有两项配合：

1. Decoder 输入相对预测目标右移一位，开头加入起始符；
2. 上三角未来位置在 Softmax 前设为 $-\infty$。

Mask：

$$
M_{ij}=\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

这使位置 $i$ 只能利用 $y_{<i}$。训练仍可一次计算全部位置，因为“并行计算”与“信息是否可见”是不同概念：矩阵可以并行算，Mask 决定依赖图。

---

## 8. Position-wise FFN

论文使用：

$$
\operatorname{FFN}(x)
=\max(0,xW_1+b_1)W_2+b_2
$$

Base 模型的维度变化：

```text
512 → 2048 → ReLU → 512
```

FFN 对每个位置独立、使用相同参数，但不同层之间不共享参数。等价地，可理解为两个核大小为 1 的卷积。

### Attention 与 FFN 的分工

- Attention 主要在 Token 维度混合信息；
- FFN 主要在通道/特征维度变换信息；
- 二者交替堆叠形成“跨位置通信 + 逐位置计算”的基本模式。

从现代 LLM 研究看，FFN 往往占据大量参数和 FLOPs，不能把 Transformer 简化为“只有 Attention”。论文标题是架构主张，不是说其他组件不重要。

---

## 9. Residual 与 LayerNorm

每个子层外使用残差连接，并在相加后 LayerNorm：

$$
y=\operatorname{LN}(x+F(x))
$$

这是 **Post-Norm**。残差要求子层输入输出维度一致，因此各子层输出统一为 $d_{model}=512$。

### 现代视角下的限制

Post-Norm 在网络加深时可能更难优化，后来的大模型普遍转向 Pre-Norm：

$$
y=x+F(\operatorname{Norm}(x))
$$

这不否定原论文设计，而是说明 Transformer 是持续演进的架构族。阅读现代代码时，不能机械套用 Figure 1 的 Norm 顺序。

---

## 10. Embedding、权重共享与输出层

源 Token 和目标 Token 先映射为 $d_{model}$ 维 Embedding。Decoder 最终隐藏状态经过线性变换和 Softmax，得到目标词表概率。

论文共享三处权重矩阵：

- 源语言 Embedding；
- 目标语言 Embedding；
- 输出 Softmax 前线性层。

Embedding 输出额外乘以 $\sqrt{d_{model}}$。权重绑定减少参数，并让输入/输出词表示处于一致空间；这要求词表设计能够支持共享。

---

## 11. 正弦位置编码

没有 RNN 和卷积后，模型本身对输入排列缺少顺序归纳偏置。论文把位置编码加到输入 Embedding：

$$
PE_{(pos,2i)}=\sin\left(pos/10000^{2i/d_{model}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(pos/10000^{2i/d_{model}}\right)
$$

不同维度对应不同波长，波长从 $2\pi$ 到 $10000\cdot2\pi$ 构成几何级数。

作者选择固定正弦编码的理由：对固定偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，因此模型可能更容易学习相对位移关系，并可能外推到训练长度之外。

消融中，把正弦编码换为可学习位置 Embedding，开发集结果几乎相同（25.8 vs 25.7 BLEU）。因此论文并没有证明正弦编码绝对更优，只给出了外推方面的设计动机。

---

## 12. 为什么 Self-Attention：复杂度表的正确解读

论文用三个指标比较层类型：每层复杂度、最少顺序操作数、最大路径长度。

| 层类型 | 每层复杂度 | 顺序操作 | 最大路径长度 |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted Self-Attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

### 12.1 Self-Attention 什么时候更便宜

比较 $n^2d$ 与 $nd^2$：当 $n<d$ 时，Self-Attention 的主项可能小于循环层。在当时机器翻译常见设置中，序列长度通常远小于表示维度。

### 12.2 “顺序操作 $O(1)$”不等于常数计算量

它表示一个 Attention 层内所有位置能并行计算，不表示总 FLOPs 与序列长度无关。标准 Attention 仍需构造 $n\times n$ 分数矩阵。

### 12.3 最大路径长度为何重要

任意两个位置在 Self-Attention 中可一步直接交互；RNN 需要沿时间链传播。较短路径被认为更利于学习长距离依赖，但全局连接也引入平方成本和加权平均的表达问题。

---

## 13. 训练设置复现要点

### 13.1 数据与分词

- WMT 2014 English–German：约 450 万句对，BPE，共享约 37K Token 词表；
- WMT 2014 English–French：约 3600 万句对，约 32K WordPiece 词表；
- 句对按近似长度分批，每个 Batch 约含 25K 源 Token 和 25K 目标 Token。

### 13.2 优化器

使用 Adam：

$$
\beta_1=0.9,\quad \beta_2=0.98,\quad \epsilon=10^{-9}
$$

学习率调度：

$$
lr=d_{model}^{-0.5}
\cdot\min(step^{-0.5},\ step\cdot warmup^{-1.5})
$$

其中 $warmup=4000$。

含义：

- Warmup 阶段学习率近似线性增大；
- 之后按步数的平方根倒数衰减；
- 模型维度越大，基础学习率越小。

### 13.3 正则化

- Residual Dropout：对子层输出与位置编码/Embedding 之和应用；
- Attention 权重上也使用 Dropout；
- Base Dropout 为 0.1，Big 的主要实验配置为 0.3；
- Label smoothing 使用 $\epsilon_{ls}=0.1$。

Label smoothing 会使困惑度看起来变差一些，但提高准确率与 BLEU，说明最大似然的置信度校准与序列生成质量并不完全一致。

### 13.4 硬件与时间

- 单机 8 张 NVIDIA P100；
- Base：约 0.4 秒/step，100K steps，约 12 小时；
- Big：约 1.0 秒/step，300K steps，约 3.5 天。

这些是 2017 年硬件与实现背景下的数据，不能直接与现代 GPU、混合精度和 FlashAttention 训练时间横向比较。

---

## 14. 机器翻译结果

论文最终报告的代表结果：

| 任务 | Transformer-big BLEU | 结论 |
|---|---:|---|
| WMT14 English–German | 28.4 | 超过当时已报告结果，包括集成模型 |
| WMT14 English–French | 41.0 | 超过当时单模型结果，训练成本更低 |

解码采用 Beam Search：

- Beam size = 4；
- Length penalty $\alpha=0.6$；
- 最大输出长度为输入长度 + 50；
- Base 平均最后 5 个 Checkpoint，Big 平均最后 20 个。

### 研究性解读

论文的说服力来自三项同时成立：

1. 质量达到或刷新当时水平；
2. 单样本内部训练并行性显著提高；
3. 训练成本低于强基线。

只展示 BLEU 提升不足以支撑架构替代；效率结果使论文影响力显著增强。

---

## 15. 消融实验精读

Table 3 是理解论文设计选择的核心证据。

### 15.1 Head 数量

- 1 Head：PPL 5.29，BLEU 24.9；
- 4 Heads：PPL 5.00，BLEU 25.5；
- 8 Heads（Base）：PPL 4.92，BLEU 25.8；
- 16 Heads：PPL 4.91，BLEU 25.8；
- 32 Heads：PPL 5.01，BLEU 25.4。

结论：多头优于单头，但过多 Head 会让每头维度过小，不一定继续受益。

### 15.2 Key 维度

减小 $d_k$ 会损害结果，说明学习 Query–Key 兼容性并不简单，需要足够的投影容量。

### 15.3 深度、宽度和 FFN

- 2 层明显较差；4 层接近但仍低于 Base；
- 8 层的 PPL 改善，但 BLEU 未超过 Base；
- 增大 $d_{model}$ 和 $d_{ff}$ 通常改善结果，同时显著增加参数。

这说明当时训练方案下，规模扩展总体有效，但单一指标与最终 BLEU 并非严格单调。

### 15.4 Dropout 与 Label Smoothing

无 Dropout 明显过拟合；适当 Dropout 很重要。Label smoothing 对 BLEU 有帮助，即使 PPL 指标可能受其目标分布改变影响。

### 15.5 位置编码

固定正弦与可学习位置 Embedding 结果近似，支持“架构需要位置信息”，但不支持“只有正弦位置编码可行”。

---

## 16. 英文成分句法分析实验

作者用 4 层 Transformer 测试模型是否能迁移到翻译之外的结构化输出任务：

- 仅 WSJ 约 40K 训练句：F1 91.3；
- 半监督、约 17M 句：F1 92.7。

这个实验规模不如翻译部分系统，但具有概念价值：它说明纯 Attention 架构并非只适用于某个翻译数据集，能够学习带强结构约束、输出比输入更长的任务。

---

## 17. 论文的真正创新、边界与不足

### 17.1 真正创新

- 首次构建完全依靠 Attention 的高性能序列转导主干；
- 将全局 Self-Attention、Multi-Head、位置编码、FFN、残差和归一化组织为可扩展系统；
- 用路径长度与顺序操作分析解释架构优势；
- 用质量和训练成本共同证明替代 RNN/CNN 的可行性。

### 17.2 不能过度宣称的地方

- Attention、Encoder–Decoder、残差、LayerNorm、位置表示都不是从零发明；
- 实验核心是机器翻译，而不是今天意义上的通用大语言模型；
- 论文没有证明 Attention 对所有序列长度和任务都优于循环或卷积；
- $O(n^2)$ 的长序列代价在当时任务上不突出，后来成为核心瓶颈；
- Attention 权重可以提供分析线索，但不能自动等同于因果解释。

### 17.3 原始架构的工程局限

- 全局 Attention 的分数矩阵随长度平方增长；
- Post-Norm 在极深模型中优化困难；
- 固定正弦位置编码不是长上下文的最终方案；
- 标准 MHA 的 KV Cache 对自回归大模型较大；
- Decoder 的生成过程仍然逐 Token 串行，论文也把“减少生成顺序性”列为未来工作。

---

## 18. 从原论文到现代大模型

### 18.1 GPT 路线

GPT 保留 Decoder 的 Masked Self-Attention 和 FFN，移除 Encoder 与 Cross-Attention，使用大规模文本做 Next-token Prediction。因果 Transformer 因而从翻译组件变成通用生成模型。

### 18.2 BERT 路线

BERT 主要保留 Encoder 双向 Self-Attention，通过掩码语言建模学习理解表示。

### 18.3 现代 LLM 常见改造

- Post-Norm → Pre-Norm；
- LayerNorm → RMSNorm；
- ReLU → GELU / SwiGLU；
- 正弦位置编码 → RoPE；
- MHA → GQA / MQA；
- 普通实现 → FlashAttention；
- Dense FFN → 部分模型采用 MoE；
- 较短上下文 → 位置插值、窗口注意力等长上下文方法。

这些变化没有替换 Transformer 的基本骨架：跨 Token 的 Attention、逐 Token 的通道变换、残差路径和层级堆叠仍是主线。

---

## 19. 参数量的粗略推导

忽略 Bias 和 Norm，一个 Base Encoder 层：

- Q/K/V/O 投影约 $4d_{model}^2$；
- FFN 约 $2d_{model}d_{ff}$。

代入 $d_{model}=512,d_{ff}=2048$：

$$
4\times512^2+2\times512\times2048
\approx 3.15\text{M}
$$

一个 Decoder 层多一个 Cross-Attention，约增加 $4d_{model}^2$，总计约 4.19M。6 层 Encoder + 6 层 Decoder 的 Block 参数约：

$$
6\times3.15+6\times4.19\approx44.0\text{M}
$$

再加共享词嵌入/输出矩阵、Bias 与 Norm，接近论文报告的约 65M。这个估算有助于理解：FFN 和投影矩阵是参数主体，Attention 分数矩阵本身不是可训练参数。

---

## 20. 建议的复现与研究任务

### Level 1：最小复现

- 实现 Scaled Dot-Product Attention；
- 验证 Mask 后未来权重为 0；
- 打印每一步张量形状；
- 与框架自带 Attention 做数值对齐。

### Level 2：复现完整 Block

- 实现原论文 Post-Norm Encoder / Decoder；
- 实现 Padding Mask、Causal Mask、Cross-Attention；
- 加入正弦位置编码和权重共享；
- 在小型复制/翻译任务上过拟合一个 Batch。

### Level 3：研究对照

- Post-Norm vs Pre-Norm 的梯度稳定性；
- 单头 vs 多头，固定总投影维度；
- 正弦位置 vs Learned Position vs RoPE；
- ReLU FFN vs SwiGLU；
- MHA vs GQA 的显存和吞吐；
- 标准 Attention vs FlashAttention 的显存峰值。

### Level 4：论文阅读延伸

建议沿两条线继续：

1. **架构演进**：BERT、GPT 系列、T5、LLaMA、Mistral/Mixtral；
2. **机制与效率**：RoPE、FlashAttention、MQA/GQA、长上下文、MoE。

---

## 21. 研究生答辩式自测

1. 为什么论文称 Self-Attention 的顺序操作数为 $O(1)$，但计算复杂度仍为 $O(n^2d)$？
2. Cross-Attention 中 Q、K、V 分别来自哪里，为什么源长与目标长可以不同？
3. 缩放因子为什么是 $\sqrt{d_k}$ 而不是 $d_k$？
4. Multi-Head 的总维度为何通常保持为 $d_{model}$？
5. 论文的 Decoder 如何在训练时并行，又保持自回归约束？
6. 为什么 FFN 是 Transformer 的核心组件，而非可忽略的附属层？
7. Post-Norm 与 Pre-Norm 的计算图有何不同？
8. 论文消融是否证明 Head 越多越好？
9. 固定正弦位置编码的理论动机与实验结论分别是什么？
10. Transformer 相比 RNN 的优势是总 FLOPs 更低，还是并行路径更短？应在什么条件下讨论？
11. 为什么现代 GPT 没有 Cross-Attention，而原始 Transformer Decoder 有？
12. 原论文的哪些结论能外推到现代 LLM，哪些不能直接外推？

---

## 22. 总结

《Attention Is All You Need》的历史意义，不在于宣称其他神经网络组件都不再需要，而在于证明：

> 只用 Attention 负责跨位置的信息交互，就能建立一个质量更高、训练更并行、远距离路径更短的完整序列转导系统。

对于大模型研究者，读完论文后应形成三个层次的认识：

1. **公式层**：能推导 QKV、Mask、Multi-Head 和张量维度；
2. **系统层**：理解 Attention、FFN、Residual、Norm、位置编码如何共同让深层网络可训练；
3. **研究层**：知道论文证据的适用边界，并能解释现代 LLM 为什么对原始结构做 Pre-Norm、RoPE、GQA、SwiGLU 等改造。

---

## 参考资料

- Vaswani et al., 2017, [NeurIPS 官方论文页面](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- Vaswani et al., [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- 原论文公开实现入口：[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
- 配套结构图文档：[Transformer 原论文官方结构图中文重绘](./Transformer_原论文官方结构图中文重绘.md)

