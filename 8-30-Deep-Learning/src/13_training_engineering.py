"""知识点 13：梯度累积、梯度裁剪、混合精度与有效 batch size。"""

import argparse
from contextlib import nullcontext

import torch
from torch import nn

from common import get_device, grad_norm, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=10)
    args = parser.parse_args()
    set_seed(13)
    device = get_device()
    model = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 8)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    micro_batch_size, accumulation_steps = 4, 4
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    last_before = last_after = 0.0

    for _ in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation_steps):
            x = torch.randn(micro_batch_size, 32, device=device)
            y = torch.randint(0, 8, (micro_batch_size,), device=device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            )
            with amp_context:
                logits = model(x)
                # 除以累积步数，使梯度等价于大 batch 上 loss 的平均值。
                loss = nn.functional.cross_entropy(logits, y) / accumulation_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        last_before = grad_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        last_after = grad_norm(model)
        scaler.step(optimizer)
        scaler.update()

    print("device:", device, "AMP enabled:", use_amp)
    print("micro batch:", micro_batch_size)
    print("accumulation steps:", accumulation_steps)
    print("effective batch:", micro_batch_size * accumulation_steps)
    print(f"gradient norm before/after clipping: {last_before:.3f} -> {last_after:.3f}")
    assert last_after <= 0.5001
    print("注意：梯度累积降低激活的单步峰值，但不会减少模型参数本身的显存。")
    print("\n思考题：累积步数改变后，学习率和 BatchNorm 行为可能需要怎样调整？")


if __name__ == "__main__":
    main()
