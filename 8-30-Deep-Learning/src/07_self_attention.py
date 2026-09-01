"""知识点 07：缩放点积注意力、多头拆分与因果 mask。"""

import torch

from common import set_seed
from transformer_components import CausalSelfAttention


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if causal:
        length = q.size(-2)
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


def main() -> None:
    set_seed(7)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    output, weights = scaled_dot_product_attention(q, k, v, causal=True)
    print("single-head output shape:", tuple(output.shape))
    print("attention weights:\n", weights[0].round(decimals=3))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 4))
    assert torch.count_nonzero(weights[0].triu(diagonal=1)) == 0

    layer = CausalSelfAttention(n_embd=16, n_heads=4, block_size=8)
    x = torch.randn(2, 6, 16)
    multi_output, multi_weights = layer(x, return_weights=True)
    print("multi-head output shape:", tuple(multi_output.shape))
    print("multi-head weights shape [B,H,T,T]:", tuple(multi_weights.shape))
    assert multi_output.shape == x.shape
    print("\n思考题：如果不除以 sqrt(d_k)，维度增大时 Softmax 和梯度会怎样？")


if __name__ == "__main__":
    main()

