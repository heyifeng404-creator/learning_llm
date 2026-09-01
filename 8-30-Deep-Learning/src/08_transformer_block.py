"""知识点 08：多头注意力、残差连接、LayerNorm 与前馈网络。"""

import torch

from common import grad_norm, set_seed
from transformer_components import TransformerBlock


def main() -> None:
    set_seed(8)
    block = TransformerBlock(n_embd=32, n_heads=4, block_size=16, dropout=0.1)
    x = torch.randn(3, 12, 32, requires_grad=True)
    output = block(x)
    loss = output.pow(2).mean()
    loss.backward()

    print("input/output shape:", tuple(x.shape), "->", tuple(output.shape))
    print("block parameter gradient norm:", round(grad_norm(block), 4))
    print("input gradient norm:", round(x.grad.norm().item(), 4))
    print("残差连接保持形状并给梯度提供短路径；LayerNorm 在特征维归一化。")
    assert output.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("\n思考题：将两个残差连接去掉，堆叠很多层后训练会有什么风险？")


if __name__ == "__main__":
    main()

