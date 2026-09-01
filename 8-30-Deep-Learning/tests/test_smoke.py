"""关键不变量测试：数据移位、因果性、形状、梯度和 LoRA 冻结。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from lm_utils import CausalTextDataset, CharTokenizer  # noqa: E402
from mini_gpt_model import GPTConfig, MiniGPT, sample_from_logits  # noqa: E402
from transformer_components import CausalSelfAttention, TransformerBlock  # noqa: E402


def load_numbered_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CourseSmokeTests(unittest.TestCase):
    def test_causal_dataset_shift(self) -> None:
        tokenizer = CharTokenizer("研究生学习大模型")
        dataset = CausalTextDataset(tokenizer.encode("研究生学习大模型"), block_size=4)
        x, y = dataset[0]
        self.assertTrue(torch.equal(x[1:], y[:-1]))

    def test_attention_cannot_see_future(self) -> None:
        layer = CausalSelfAttention(n_embd=16, n_heads=4, block_size=8)
        _, weights = layer(torch.randn(2, 6, 16), return_weights=True)
        future = weights.triu(diagonal=1)
        self.assertEqual(torch.count_nonzero(future).item(), 0)

    def test_transformer_shape_and_gradient(self) -> None:
        block = TransformerBlock(16, 4, 8)
        x = torch.randn(2, 7, 16, requires_grad=True)
        y = block(x)
        y.mean().backward()
        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(x.grad)

    def test_minigpt_loss_and_weight_tying(self) -> None:
        model = MiniGPT(GPTConfig(vocab_size=20, block_size=8, n_embd=16, n_heads=4, n_layers=1))
        x = torch.randint(0, 20, (2, 8))
        logits, loss = model(x, x)
        self.assertEqual(logits.shape, (2, 8, 20))
        self.assertTrue(torch.isfinite(loss))
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_sampling_filters(self) -> None:
        logits = torch.tensor([[0.0, 1.0, 4.0, -2.0]])
        greedy = sample_from_logits(logits, temperature=0)
        self.assertEqual(greedy.item(), 2)
        for _ in range(10):
            self.assertEqual(sample_from_logits(logits, top_k=1).item(), 2)

    def test_lora_freezes_base(self) -> None:
        module = load_numbered_module("11_lora_finetune.py", "lora_case")
        layer = module.LoRALinear(torch.nn.Linear(8, 4), rank=2)
        self.assertFalse(layer.base.weight.requires_grad)
        self.assertTrue(layer.lora_a.requires_grad)
        self.assertTrue(layer.lora_b.requires_grad)


if __name__ == "__main__":
    unittest.main()

