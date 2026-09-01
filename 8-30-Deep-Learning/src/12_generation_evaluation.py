"""知识点 12：temperature、top-k、top-p、验证 loss 与困惑度。"""

import argparse
import math

import torch

from common import get_device, set_seed
from course_training import token_loss, train_minigpt


def generate_with_policy(model, tokenizer, prompt: str, **policy) -> str:
    device = next(model.parameters()).device
    context = torch.tensor([tokenizer.encode(prompt)], device=device)
    result = model.generate(context, max_new_tokens=32, **policy)
    return tokenizer.decode(result[0].cpu().tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    model, tokenizer, valid_ids, losses = train_minigpt(args.steps, seed=12)
    validation_loss = token_loss(model, valid_ids)
    perplexity = math.exp(min(validation_loss, 20))

    policies = {
        "greedy": {"temperature": 0.0},
        "low-temperature": {"temperature": 0.6, "top_k": 10},
        "top-p": {"temperature": 0.9, "top_p": 0.9},
    }
    print("device:", get_device())
    print(f"last train loss: {losses[-1]:.3f}")
    print(f"held-out token loss: {validation_loss:.3f}; perplexity: {perplexity:.2f}")
    for i, (name, policy) in enumerate(policies.items()):
        set_seed(100 + i)
        print(f"{name:>16}:", repr(generate_with_policy(model, tokenizer, "模型", **policy)))
    print("困惑度衡量 token 预测，不直接等于事实性、有用性或安全性。")
    print("\n思考题：为什么不能用同一训练语料同时训练和报告最终 PPL？")


if __name__ == "__main__":
    main()

