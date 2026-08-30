"""零依赖、可运行的迷你 Transformer 教学示例（仅演示前向传播）。"""

import math
import random


random.seed(7)


def shape(x):
    """返回本项目中 1D/2D/3D 列表张量的形状。"""
    dims = []
    while isinstance(x, list):
        dims.append(len(x))
        x = x[0] if x else None
    return tuple(dims)


def linear(x, weight, bias=None):
    """最后一维线性变换：[..., in_dim] -> [..., out_dim]。"""
    if x and isinstance(x[0][0], list):
        return [linear(sample, weight, bias) for sample in x]
    out_dim = len(weight[0])
    return [
        [sum(row[i] * weight[i][j] for i in range(len(row))) +
         (bias[j] if bias else 0.0) for j in range(out_dim)]
        for row in x
    ]


def rand_matrix(rows, cols, scale=0.12):
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def add(a, b):
    return [[[a[n][t][d] + b[n][t][d] for d in range(len(a[0][0]))]
             for t in range(len(a[0]))] for n in range(len(a))]


def relu(x):
    return [[[max(0.0, v) for v in token] for token in sample] for sample in x]


def softmax(values):
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


class Embedding:
    def __init__(self, vocab_size, d_model):
        self.table = rand_matrix(vocab_size, d_model, 0.3)

    def __call__(self, token_ids):
        # token_ids: [B, T] -> [B, T, D]
        return [[self.table[token_id][:] for token_id in sample] for sample in token_ids]


def sinusoidal_position_encoding(seq_len, d_model):
    """经典正弦/余弦位置编码：[T, D]。"""
    pe = [[0.0] * d_model for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            pe[pos][i] = math.sin(angle)
            if i + 1 < d_model:
                pe[pos][i + 1] = math.cos(angle)
    return pe


def add_position(x, pe):
    return [[[x[n][t][d] + pe[t][d] for d in range(len(x[0][0]))]
             for t in range(len(x[0]))] for n in range(len(x))]


def scaled_dot_product_attention(q, k, v, causal=True):
    """单个头：Q/K/V 均为 [B, T, Dh]，返回 context 与注意力权重。"""
    batch, seq_len, head_dim = shape(q)
    contexts, all_weights = [], []
    for n in range(batch):
        sample_context, sample_weights = [], []
        for i in range(seq_len):
            scores = []
            for j in range(seq_len):
                score = sum(q[n][i][d] * k[n][j][d] for d in range(head_dim))
                score /= math.sqrt(head_dim)  # Scaled Dot-Product 的 scaled
                if causal and j > i:
                    score = -1e9  # GPT 式因果遮罩：不能偷看未来 token
                scores.append(score)
            weights = softmax(scores)
            context = [sum(weights[j] * v[n][j][d] for j in range(seq_len))
                       for d in range(head_dim)]
            sample_weights.append(weights)
            sample_context.append(context)
        all_weights.append(sample_weights)
        contexts.append(sample_context)
    return contexts, all_weights


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.wq = rand_matrix(d_model, d_model)
        self.wk = rand_matrix(d_model, d_model)
        self.wv = rand_matrix(d_model, d_model)
        self.wo = rand_matrix(d_model, d_model)

    def __call__(self, x):
        # 三个不同投影产生 Q/K/V：[B,T,D] -> [B,T,D]
        q, k, v = linear(x, self.wq), linear(x, self.wk), linear(x, self.wv)
        head_outputs, head_weights = [], []
        for h in range(self.num_heads):
            start, end = h * self.head_dim, (h + 1) * self.head_dim
            qh = [[token[start:end] for token in sample] for sample in q]
            kh = [[token[start:end] for token in sample] for sample in k]
            vh = [[token[start:end] for token in sample] for sample in v]
            context, weights = scaled_dot_product_attention(qh, kh, vh)
            head_outputs.append(context)
            head_weights.append(weights)

        # 拼接所有头：[B,T,H,Dh] -> [B,T,D]，再做输出投影
        batch, seq_len, _ = shape(x)
        joined = [[[value for h in range(self.num_heads) for value in head_outputs[h][n][t]]
                   for t in range(seq_len)] for n in range(batch)]
        return linear(joined, self.wo), {"q": q, "k": k, "v": v, "weights": head_weights}


class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma, self.beta, self.eps = [1.0] * d_model, [0.0] * d_model, eps

    def __call__(self, x):
        result = []
        for sample in x:
            normalized_sample = []
            for token in sample:
                mean = sum(token) / len(token)
                variance = sum((v - mean) ** 2 for v in token) / len(token)
                normalized_sample.append([
                    (v - mean) / math.sqrt(variance + self.eps) * self.gamma[d] + self.beta[d]
                    for d, v in enumerate(token)
                ])
            result.append(normalized_sample)
        return result


class FFN:
    def __init__(self, d_model, d_ff):
        self.w1, self.b1 = rand_matrix(d_model, d_ff), [0.0] * d_ff
        self.w2, self.b2 = rand_matrix(d_ff, d_model), [0.0] * d_model

    def __call__(self, x):
        return linear(relu(linear(x, self.w1, self.b1)), self.w2, self.b2)


class TransformerBlock:
    """Pre-Norm Block：x + Attention(LN(x))，再 x + FFN(LN(x))。"""
    def __init__(self, d_model, num_heads, d_ff):
        self.norm1, self.attn = LayerNorm(d_model), MultiHeadAttention(d_model, num_heads)
        self.norm2, self.ffn = LayerNorm(d_model), FFN(d_model, d_ff)

    def __call__(self, x):
        attn_out, debug = self.attn(self.norm1(x))
        x = add(x, attn_out)       # Residual 1
        x = add(x, self.ffn(self.norm2(x)))  # Residual 2
        return x, debug


class MiniLanguageModel:
    def __init__(self, vocab_size, d_model=8, num_heads=2, d_ff=16):
        self.embedding = Embedding(vocab_size, d_model)
        self.block = TransformerBlock(d_model, num_heads, d_ff)
        self.final_norm = LayerNorm(d_model)
        self.lm_head = rand_matrix(d_model, vocab_size)
        self.d_model = d_model

    def __call__(self, token_ids):
        x_embed = self.embedding(token_ids)
        pe = sinusoidal_position_encoding(len(token_ids[0]), self.d_model)
        x_positioned = add_position(x_embed, pe)
        hidden, debug = self.block(x_positioned)
        logits = linear(self.final_norm(hidden), self.lm_head)
        return logits, {"embedding": x_embed, "positioned": x_positioned, **debug}


def main():
    vocab = ["<pad>", "我", "喜欢", "学习", "Transformer", "。"]
    token_to_id = {token: i for i, token in enumerate(vocab)}
    tokens = ["我", "喜欢", "学习", "Transformer"]
    token_ids = [[token_to_id[t] for t in tokens]]  # B=1, T=4

    model = MiniLanguageModel(len(vocab))
    logits, debug = model(token_ids)

    print("输入 token:", tokens)
    print("token_ids       [B,T]    =", shape(token_ids))
    print("Embedding       [B,T,D]  =", shape(debug["embedding"]))
    print("加位置编码后    [B,T,D]  =", shape(debug["positioned"]))
    print("Q / K / V       [B,T,D]  =", shape(debug["q"]), shape(debug["k"]), shape(debug["v"]))
    print("注意力权重/头   [B,T,T]  =", shape(debug["weights"][0]))
    print("最终 logits     [B,T,V]  =", shape(logits))

    last_probs = softmax(logits[0][-1])
    ranked = sorted(zip(vocab, last_probs), key=lambda item: item[1], reverse=True)
    print("\n最后位置的下一个 token 概率（随机参数，仅演示流程）：")
    for token, probability in ranked:
        print(f"  {token:12s} {probability:.4f}")

    print("\n第 1 个注意力头的因果注意力矩阵（行=query，列=key）：")
    for row in debug["weights"][0][0]:
        print(" ", " ".join(f"{value:.3f}" for value in row))


if __name__ == "__main__":
    main()
