# 从零看懂 Transformer：可运行的最小项目

这个项目用 **Python 标准库**手写一个迷你 Decoder-only Transformer 前向传播。它不训练模型，参数是随机数；目标是让你看清数据怎样经过 Embedding、位置编码、Attention、FFN，最后变成“下一个 token”的概率。

## 1. 运行方法

要求：Python 3.8 或更高版本，无第三方依赖。

```bash
cd transformer_beginner
python main.py
```

你会看到各阶段的张量维度、最后一个位置对词表中各 token 的预测概率，以及第一个注意力头的注意力矩阵。概率没有语义意义，因为参数未经训练。

## 2. 先认识符号

本项目一直使用以下字母：

| 符号 | 含义 | 示例值 |
|---|---|---:|
| `B` | batch size，一次处理几句话 | 1 |
| `T` | sequence length，每句话几个 token | 4 |
| `D` | `d_model`，每个 token 的表示维度 | 8 |
| `H` | 注意力头数 | 2 |
| `Dh` | 每个头的维度，`D / H` | 4 |
| `F` | FFN 中间层维度 | 16 |
| `V` | vocabulary size，词表大小 | 6 |

输入是 `[B,T] = [1,4]`，最终输出 logits 是 `[B,T,V] = [1,4,6]`。

## 3. 整体数据流

```text
token IDs [B,T]
  ↓ Embedding
[B,T,D]
  ↓ 加位置编码 [T,D]（广播到每个 batch）
[B,T,D]
  ↓ Transformer Block
[B,T,D]
  ↓ LayerNorm + LM Head
logits [B,T,V]
  ↓ 对最后位置做 softmax
下一个 token 的概率 [V]
```

## 4. 对照代码逐段学习

### 4.1 Token 与 Embedding

`main()` 把 `我 喜欢 学习 Transformer` 转成整数 ID，形状为 `[B,T]`。`Embedding.__call__()` 用每个 ID 查询一行向量，得到 `[B,T,D]`。

直觉上，Embedding 是一本可学习的“词向量字典”：整数 ID 是页码，向量是该页内容。真实训练会不断更新这些向量；本例只随机初始化。

### 4.2 位置编码

Attention 本身不知道顺序，所以 `sinusoidal_position_encoding()` 为第 0、1、2……个位置产生不同向量 `[T,D]`。`add_position()` 将其加到每个 token 的 Embedding 上，形状仍是 `[B,T,D]`。

本例采用经典正弦/余弦位置编码。现代模型也常用 RoPE，但“把位置信息注入 token 表示”这个目的相同。

### 4.3 Q、K、V

`MultiHeadAttention.__call__()` 通过三个不同的线性投影得到：

```text
Q = X · Wq
K = X · Wk
V = X · Wv
```

它们的整体形状都是 `[B,T,D]`。可以这样记：

- Q（Query）：当前 token 想找什么；
- K（Key）：每个 token 可用什么特征被匹配；
- V（Value）：匹配成功后真正取走的信息。

Q 和 K 决定“看谁”，V 决定“拿什么”。

### 4.4 Scaled Dot-Product Attention

`scaled_dot_product_attention()` 实现核心公式：

```text
Attention(Q,K,V) = softmax(QKᵀ / √Dh) V
```

单个头的 Q、K、V 是 `[B,T,Dh]`。`QKᵀ` 让每个 query 与所有 key 配对，产生分数矩阵 `[B,T,T]`；softmax 后每行和为 1；再对 V 加权求和，输出 `[B,T,Dh]`。

除以 `√Dh` 是为了避免维度较大时点积数值过大，导致 softmax 过于尖锐。

本项目还使用 causal mask：第 `i` 个 token 不能看 `i` 后面的 token。代码把未来位置的分数改成极小值，softmax 后权重接近 0。这正是 GPT 能做“根据前文预测后文”的关键限制。

### 4.5 Multi-Head Attention

`MultiHeadAttention` 将 D 维拆为 H 个头，每头维度 `Dh=D/H`。本例是 `D=8, H=2, Dh=4`。

每个头独立做 Attention，可能学习不同关系，例如一个头关注语法，另一个头关注语义。所有头的 `[B,T,Dh]` 结果拼接回 `[B,T,D]`，再经过输出投影 `Wo`，形状仍为 `[B,T,D]`。

教学代码先生成 `[B,T,D]` 的 Q/K/V，再按最后一维切头。高性能框架通常会 reshape 成 `[B,H,T,Dh]` 后并行计算，数学含义一致。

### 4.6 FFN

`FFN` 对每个 token 独立应用两层线性变换：

```text
[B,T,D] → [B,T,F] → ReLU → [B,T,D]
```

Attention 负责 token 之间交换信息；FFN 负责对每个 token 已经汇总的信息做更深的特征变换。真实大模型常用 GELU、SiLU 或 SwiGLU，本例使用更易读的 ReLU。

### 4.7 Residual 与 LayerNorm

`TransformerBlock` 使用 Pre-Norm 结构：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Residual（残差连接）中的 `x + ...` 给信息和梯度提供直接通道，使深层网络更容易训练。`LayerNorm` 对每个 token 的 D 个特征计算均值和方差，使数值更稳定；输入输出均为 `[B,T,D]`。

原始 Transformer 常画成 Post-Norm，许多现代大模型使用 Pre-Norm。两者放置顺序不同，但都包含相同核心组件。

### 4.8 Transformer Block

`TransformerBlock` 把多头注意力、两个 LayerNorm、FFN 和两次 Residual 组合在一起。真实模型会堆叠很多个 Block；本项目只用一个，以便输出容易观察。

### 4.9 最小语言模型式前向传播

`MiniLanguageModel` 的流程是：

1. token IDs `[B,T]` → Embedding `[B,T,D]`；
2. 加位置编码，仍为 `[B,T,D]`；
3. 经过 Transformer Block，仍为 `[B,T,D]`；
4. 最终 LayerNorm；
5. LM Head 将 D 维投影到词表 V 维，得到 logits `[B,T,V]`；
6. 对最后一个位置的 V 个 logits 做 softmax，得到下一个 token 的概率。

训练时，会用正确的“下一个 token”计算交叉熵损失，再反向传播更新全部参数。本项目只实现前向传播，因此重点是结构，不是生成质量。

## 5. 如何读注意力矩阵

程序最后打印一个 `[T,T]` 矩阵：行代表 query，列代表 key。第 3 行第 1 列可以理解成“第 3 个 token 对第 1 个 token 分配了多少注意力”。

因果遮罩让主对角线右上方全接近 0：第一个 token 只能看自己；第二个能看前两个；最后一个能看全部已有 token。

## 6. 建议学习顺序

1. 先运行程序，只观察每个阶段的形状。
2. 阅读 `Embedding` 和位置编码，确认为什么 `[B,T]` 会变成 `[B,T,D]`。
3. 单独阅读 `scaled_dot_product_attention()`，手画一个 `T×T` 注意力矩阵。
4. 理解单头后，再看 Multi-Head 如何切分与拼接。
5. 看 FFN、Residual、LayerNorm 如何组成 Block。
6. 最后看 `MiniLanguageModel` 如何把隐藏表示变成词表概率。
7. 修改 `tokens`、`d_model`、`num_heads`，再次运行并预测输出形状。

## 7. 推荐的小实验

- 把 `causal=True` 改成 `False`，观察注意力矩阵右上方不再为 0。
- 把头数从 2 改成 4，并计算新的 `Dh`。
- 再创建一个 `TransformerBlock` 并串联，观察形状保持不变。
- 尝试把 ReLU 换成 `max(0.01*v, v)`，实现 Leaky ReLU。
- 下一阶段用 PyTorch 重写，并加入训练循环、交叉熵损失和反向传播。

## 8. 与 PyTorch 名称对照

| 本项目 | PyTorch 常见写法 |
|---|---|
| `Embedding` | `torch.nn.Embedding` |
| `linear` | `torch.nn.Linear` |
| `LayerNorm` | `torch.nn.LayerNorm` |
| `MultiHeadAttention` | `torch.nn.MultiheadAttention` 或手写投影 |
| `FFN` | 两个 `nn.Linear` 加激活函数 |
| `MiniLanguageModel` | Embedding + Blocks + LM Head |

这份实现用普通列表和循环换取可读性，并不适合大规模训练。理解它之后，再迁移到 PyTorch，会更容易看懂张量变形和框架封装。
