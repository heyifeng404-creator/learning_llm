"""知识点 06：循环神经网络语言模型，以及它与 Transformer 的差异。"""

import argparse

import torch
from torch import nn

from common import read_corpus, set_seed
from lm_utils import CharTokenizer, random_batch


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.embedding(token_ids))
        return self.output(hidden)


def train(steps: int = 60) -> tuple[RNNLanguageModel, CharTokenizer, list[float]]:
    set_seed(6)
    text = read_corpus() * 8
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    model = RNNLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses: list[float] = []
    for _ in range(steps):
        x, y = random_batch(data, block_size=24, batch_size=16)
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return model, tokenizer, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    model, tokenizer, losses = train(args.steps)
    probe = torch.tensor([tokenizer.encode("深度学习")])
    with torch.no_grad():
        logits = model(probe)
    print("logits shape [batch, sequence, vocab]:", tuple(logits.shape))
    print(f"loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    print("RNN 按时间步传递状态，训练并行度与长程梯度路径通常不如 Transformer。")
    print("\n思考题：GRU 最后一个位置的状态需要经过多少次递归才能包含首字符信息？")


if __name__ == "__main__":
    main()

