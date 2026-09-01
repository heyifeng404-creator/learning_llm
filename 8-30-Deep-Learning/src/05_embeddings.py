"""知识点 05：one-hot、Embedding、相似度与位置编码。"""

import math

import torch
from torch import nn

from common import set_seed


def sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError("dim 必须是偶数")
    position = torch.arange(length).float().unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * scale)
    pe[:, 1::2] = torch.cos(position * scale)
    return pe


def main() -> None:
    set_seed(5)
    vocab = {"深": 0, "度": 1, "学": 2, "习": 3, "模": 4, "型": 5}
    ids = torch.tensor([[vocab[c] for c in "深度学习"]])  # [batch=1, seq=4]
    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=8)
    token_vectors = embedding(ids)

    one_hot = nn.functional.one_hot(ids, num_classes=len(vocab)).float()
    # Embedding 查表等价于 one-hot 乘权重矩阵，但更节省存储和计算。
    via_matmul = one_hot @ embedding.weight
    assert torch.allclose(token_vectors, via_matmul)

    position = sinusoidal_position_encoding(length=ids.size(1), dim=8)
    hidden = token_vectors + position.unsqueeze(0)
    cosine = nn.functional.cosine_similarity(hidden[:, :-1], hidden[:, 1:], dim=-1)

    print("token ids shape:", tuple(ids.shape))
    print("embedding shape:", tuple(token_vectors.shape))
    print("with position shape:", tuple(hidden.shape))
    print("相邻位置的余弦相似度:", [round(x, 3) for x in cosine.flatten().tolist()])
    print("\n思考题：只用 token embedding 而不加任何位置信息会丢失什么？")


if __name__ == "__main__":
    main()

