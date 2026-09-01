"""知识点 09：字符分词、词表、编码解码与因果语言建模数据集。"""

from torch.utils.data import DataLoader

from common import read_corpus
from lm_utils import CausalTextDataset, CharTokenizer


def main() -> None:
    text = read_corpus()
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)
    dataset = CausalTextDataset(ids, block_size=16)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y = next(iter(loader))

    sample = "深度学习"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print("vocab size:", tokenizer.vocab_size)
    print("encode:", sample, "->", encoded, "->", decoded)
    print("batch x/y shape:", tuple(x.shape), tuple(y.shape))
    print("x[0]:", tokenizer.decode(x[0].tolist()))
    print("y[0]:", tokenizer.decode(y[0].tolist()))
    assert decoded == sample
    assert (x[:, 1:] == y[:, :-1]).all()
    print("\n思考题：字符分词和 BPE 在词表大小、序列长度、未知词方面有何取舍？")


if __name__ == "__main__":
    main()

