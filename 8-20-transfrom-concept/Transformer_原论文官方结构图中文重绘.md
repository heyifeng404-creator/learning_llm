# Transformer 原论文 Figure 1：官方结构中文重绘与逐层图解

> 对照来源：Vaswani et al., *Attention Is All You Need*, Figure 1, NeurIPS 2017。  
> [查看 NeurIPS 官方论文](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)  
> 本图是忠于论文连接关系的**中文可编辑重绘版**，不是从官方 PDF 复制出来的原始图片。原图以论文 Figure 1 为准。

---

## 1. 完整 Encoder–Decoder 结构图

```mermaid
flowchart BT
    SRC[源序列 Inputs] --> SE[输入 Embedding]
    SE --> SPA((+))
    SPE[Positional Encoding<br/>正弦位置编码] --> SPA

    subgraph ENC[Encoder Stack × N = 6]
      direction BT
      ESA[Multi-Head<br/>Self-Attention] --> EA1[Add & Norm]
      EA1 --> EFF[Position-wise<br/>Feed Forward]
      EFF --> EA2[Add & Norm]
      EIN((Encoder 层输入)) -.残差.-> EA1
      EA1 -.残差.-> EA2
    end

    SPA --> EIN
    EIN --> ESA
    EA2 --> EM[Encoder Output / Memory Z]

    TGT[目标序列 Outputs<br/>训练时右移一位] --> TE[输出 Embedding]
    TE --> TPA((+))
    TPE[Positional Encoding<br/>正弦位置编码] --> TPA

    subgraph DEC[Decoder Stack × N = 6]
      direction BT
      DMSA[Masked Multi-Head<br/>Self-Attention] --> DA1[Add & Norm]
      DA1 --> DCA[Multi-Head Attention<br/>Cross-Attention]
      DCA --> DA2[Add & Norm]
      DA2 --> DFF[Position-wise<br/>Feed Forward]
      DFF --> DA3[Add & Norm]
      DIN((Decoder 层输入)) -.残差.-> DA1
      DA1 -.残差.-> DA2
      DA2 -.残差.-> DA3
    end

    TPA --> DIN
    DIN --> DMSA
    EM -- K, V --> DCA
    DA3 --> LIN[Linear]
    LIN --> SM[Softmax]
    SM --> PROB[Output Probabilities<br/>下一个 Token 概率]

    classDef enc fill:#e8f1ff,stroke:#3569a8,color:#102a43,stroke-width:1.5px;
    classDef dec fill:#fff0e5,stroke:#c45d19,color:#4a2511,stroke-width:1.5px;
    classDef norm fill:#f1edff,stroke:#7055a5,color:#30234d;
    classDef io fill:#e8f8ef,stroke:#36805a,color:#153f2b;
    class ESA,EFF,EM,EIN enc;
    class DMSA,DCA,DFF,DIN dec;
    class EA1,EA2,DA1,DA2,DA3 norm;
    class SRC,SE,SPE,SPA,TGT,TE,TPE,TPA,LIN,SM,PROB io;
```

### 图中最关键的三条信息流

1. **源序列路径**：Embedding + Position → 6 层 Encoder → Memory $Z$；
2. **目标序列路径**：右移后的目标 Token → Masked Decoder Self-Attention；
3. **跨序列路径**：Decoder 用当前状态作 Q，用 Encoder Memory 作 K/V。

> Mermaid 中虚线表示残差旁路。原论文每个子层执行的是 `LayerNorm(x + Sublayer(x))`，属于 Post-Norm。

---

## 2. Encoder 单层展开图

```mermaid
flowchart TB
    X[输入 X<br/>n × 512] --> QKV[线性投影得到 Q / K / V]
    QKV --> MHA[8-Head Self-Attention<br/>每头 dk = dv = 64]
    X --> ADD1((Add))
    MHA --> ADD1
    ADD1 --> LN1[LayerNorm]
    LN1 --> FF1[Linear: 512 → 2048]
    FF1 --> RELU[ReLU]
    RELU --> FF2[Linear: 2048 → 512]
    LN1 --> ADD2((Add))
    FF2 --> ADD2
    ADD2 --> LN2[LayerNorm]
    LN2 --> Y[输出 Y<br/>n × 512]
```

Encoder Self-Attention：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

所有源位置彼此可见，因此没有因果 Mask。

---

## 3. Decoder 单层展开图

```mermaid
flowchart TB
    Y[Decoder 输入 Y<br/>m × 512] --> MQKV[Q / K / V 投影]
    MQKV --> MMHA[Masked Multi-Head<br/>Self-Attention]
    Y --> A1((Add))
    MMHA --> A1
    A1 --> N1[LayerNorm]

    N1 -- Q --> CROSS[Multi-Head<br/>Cross-Attention]
    Z[Encoder Memory Z<br/>n × 512] -- K, V --> CROSS
    N1 --> A2((Add))
    CROSS --> A2
    A2 --> N2[LayerNorm]

    N2 --> F1[Linear: 512 → 2048]
    F1 --> R[ReLU]
    R --> F2[Linear: 2048 → 512]
    N2 --> A3((Add))
    F2 --> A3
    A3 --> N3[LayerNorm]
    N3 --> O[Decoder 层输出<br/>m × 512]
```

Decoder 比 Encoder 多出的模块是 Cross-Attention：

$$
Q=H_{decoder}W_Q,qquad
K=ZW_K,qquad
V=ZW_V
$$

它回答的是：“生成当前目标词时，源句哪些位置最重要？”

---

## 4. Scaled Dot-Product Attention：Figure 2 左侧重绘

```mermaid
flowchart BT
    Q[Q] --> MATMUL1[MatMul<br/>QKᵀ]
    K[K] --> MATMUL1
    MATMUL1 --> SCALE[Scale<br/>÷ √dk]
    SCALE --> MASK[Mask<br/>仅 Decoder Self-Attention 使用]
    MASK --> SOFT[Softmax]
    SOFT --> MATMUL2[MatMul<br/>权重 × V]
    V[V] --> MATMUL2
    MATMUL2 --> OUT[Attention Output]
```

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

原论文 Figure 2 在 Mask 模块旁标记 “optional”，因为：

- Encoder Self-Attention 不需要因果 Mask；
- Decoder Self-Attention 必须使用因果 Mask；
- Cross-Attention 不使用目标侧因果 Mask，但实现中可能使用源 Padding Mask。

---

## 5. Multi-Head Attention：Figure 2 右侧重绘

```mermaid
flowchart BT
    Q[Q] --> LQ[Linear × h]
    K[K] --> LK[Linear × h]
    V[V] --> LV[Linear × h]
    LQ --> H1[Scaled Dot-Product<br/>Attention Head 1]
    LK --> H1
    LV --> H1
    LQ --> HH[⋯ Heads 2…h]
    LK --> HH
    LV --> HH
    H1 --> CAT[Concat]
    HH --> CAT
    CAT --> LO[Linear Wᴼ]
    LO --> O[Multi-Head Output]
```

Base 模型：

```text
d_model = 512
h = 8
d_k = d_v = 64
Concat 后维度 = 8 × 64 = 512
```

---

## 6. 三种 Attention 的 Q/K/V 来源图

```mermaid
flowchart LR
    subgraph E[Encoder Self-Attention]
      EX[Encoder X] --> EQ[Q]
      EX --> EK[K]
      EX --> EV[V]
    end

    subgraph D[Decoder Masked Self-Attention]
      DY[Decoder Y] --> DQ[Q]
      DY --> DK[K]
      DY --> DV[V]
    end

    subgraph C[Cross-Attention]
      DH[Decoder Hidden] --> CQ[Q]
      EZ[Encoder Z] --> CK[K]
      EZ --> CV[V]
    end
```

| 模块 | Q | K | V | Mask |
|---|---|---|---|---|
| Encoder Self-Attention | Encoder | Encoder | Encoder | 无因果 Mask |
| Decoder Self-Attention | Decoder | Decoder | Decoder | Causal Mask |
| Cross-Attention | Decoder | Encoder | Encoder | 通常仅 Padding Mask |

---

## 7. Causal Mask 可见性图

目标序列为 $[y_1,y_2,y_3,y_4]$：

```text
             Key 位置
Query        y1   y2   y3   y4
y1           ✓    ×    ×    ×
y2           ✓    ✓    ×    ×
y3           ✓    ✓    ✓    ×
y4           ✓    ✓    ✓    ✓
```

矩阵形式：

$$
M=
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
$$

Softmax 后，所有 $-\infty$ 对应权重变为 0。

---

## 8. 原论文图与 GPT 结构的对应关系

```mermaid
flowchart LR
    ORIGINAL[原始 Transformer] --> ENC[保留 Encoder<br/>得到 BERT 风格主干]
    ORIGINAL --> DEC[保留 Decoder 的<br/>Masked Self-Attention + FFN]
    DEC --> DROP[移除 Encoder 与 Cross-Attention]
    DROP --> GPT[Decoder-only GPT]
```

GPT 不是把 Figure 1 右半边原封不动复制下来。标准 GPT Block 通常：

- 没有 Encoder；
- 没有 Cross-Attention；
- 只保留 Causal Self-Attention 与 FFN；
- 后来常改为 Pre-Norm；
- 现代模型还常用 RoPE、RMSNorm、SwiGLU、GQA。

---

## 9. 看 Figure 1 时最容易犯的错误

1. **把右侧 Decoder 当作 GPT Block**：原图 Decoder 还有 Cross-Attention。
2. **把 Add & Norm 看成先 Norm**：原论文是先子层与残差相加，再 Norm。
3. **认为 Encoder 和 Decoder 共享层参数**：层结构相同，但各层参数不共享。
4. **认为 Position Encoding 每层都加**：原图只在 Encoder/Decoder 堆栈底部加入。
5. **认为所有 Attention 都有 Causal Mask**：只有 Decoder Self-Attention 需要防止看未来。
6. **认为 Cross-Attention 的 Q/K/V 同源**：Q 来自 Decoder，K/V 来自 Encoder。
7. **认为 Softmax 就是 Attention 内唯一的 Softmax**：Attention 内 Softmax 归一化位置权重；模型顶部 Softmax 归一化词表概率。

---

## 10. 图结构速记

```text
Encoder 层：
Self-Attention → Add & Norm → FFN → Add & Norm

Decoder 层：
Masked Self-Attention → Add & Norm
→ Cross-Attention → Add & Norm
→ FFN → Add & Norm

跨模块：
Encoder 输出 → Cross-Attention 的 K、V
Decoder 隐状态 → Cross-Attention 的 Q
```

配套论文精读：[《Attention Is All You Need》原论文精读与研究分析](./Attention_Is_All_You_Need_原论文精读.md)

