"""知识点 01：张量、广播、计算图、链式法则与梯度累积。"""

import torch

from common import set_seed


def main() -> None:
    set_seed(1)

    # 矩阵乘法：(2, 3) @ (3, 1) -> (2, 1)，偏置 (1,) 被广播。
    x = torch.tensor([[1.0, 2.0, -1.0], [0.0, 3.0, 2.0]])
    w = torch.tensor([[0.2], [-0.5], [1.0]], requires_grad=True)
    b = torch.tensor([0.1], requires_grad=True)
    target = torch.tensor([[0.0], [1.0]])

    prediction = x @ w + b
    loss = ((prediction - target) ** 2).mean()
    loss.backward()

    print("x shape:", tuple(x.shape), "prediction shape:", tuple(prediction.shape))
    print("loss:", round(loss.item(), 6))
    print("autograd dw:", w.grad.flatten().tolist(), "db:", b.grad.tolist())

    # MSE 的手工梯度，用于验证 Autograd，而不是把梯度当黑盒。
    n = target.numel()
    d_prediction = 2.0 * (prediction.detach() - target) / n
    manual_dw = x.T @ d_prediction
    manual_db = d_prediction.sum(dim=0)
    print("manual   dw:", manual_dw.flatten().tolist(), "db:", manual_db.tolist())
    assert torch.allclose(w.grad, manual_dw)
    assert torch.allclose(b.grad, manual_db)

    # backward 默认把梯度累加到 .grad；训练循环必须及时清零。
    first_grad = w.grad.clone()
    loss2 = ((x @ w + b - target) ** 2).mean()
    loss2.backward()
    assert torch.allclose(w.grad, 2 * first_grad)
    w.grad.zero_()
    b.grad.zero_()
    print("第二次反传后梯度会累积；zero_ 后范数:", w.grad.norm().item())

    print("\n思考题：若把 mean 改成 sum，梯度会按什么比例变化？")


if __name__ == "__main__":
    main()

