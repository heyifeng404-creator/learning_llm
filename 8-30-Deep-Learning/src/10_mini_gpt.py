"""知识点 10：组装并训练字符级迷你 GPT，完成自回归生成闭环。"""

import argparse

import torch

from common import count_parameters, get_device
from course_training import train_minigpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--new-tokens", type=int, default=40)
    args = parser.parse_args()
    model, tokenizer, _, losses = train_minigpt(args.steps)
    device = get_device()

    prompt = "深度学习"
    context = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated = model.generate(context, args.new_tokens, temperature=0.8, top_k=12)
    text = tokenizer.decode(generated[0].cpu().tolist())

    print("device:", device)
    print("parameters:", f"{count_parameters(model):,}")
    print(f"training loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    print("generated:", repr(text))
    assert generated.size(1) == len(prompt) + args.new_tokens
    print("\n思考题：训练可并行计算所有位置，为何生成仍需逐 token 迭代？")


if __name__ == "__main__":
    main()

