# 面向大模型的深度学习：研究生实践课程

这是一套从深度学习基础逐步走到 Transformer、迷你 GPT、LoRA 微调与生成评估的离线课程。每个知识点都有一个可直接运行的 PyTorch 案例，适合作为计算机相关专业研究生的自学材料、课程实验或组会演示。

## 你将学会什么

- 用张量、计算图和反向传播理解神经网络训练。
- 从线性模型、MLP、正则化逐步过渡到词嵌入和语言模型。
- 从公式与代码两条线理解自注意力、多头注意力和 Transformer Block。
- 完成字符级分词、因果语言建模数据集、迷你 GPT 训练与文本生成。
- 理解参数高效微调（LoRA）、采样策略和困惑度等评估方法。
- 掌握混合精度、梯度累积、梯度裁剪等实际训练技巧。

## 目录结构

```text
8-30-DL/
├── README.md
├── 深度学习总体介绍.md
├── requirements.txt
├── run_all.py
├── data/
│   └── tiny_corpus.txt
├── src/
│   ├── common.py
│   ├── 01_tensor_autograd.py
│   ├── 02_linear_regression.py
│   ├── 03_mlp_classification.py
│   ├── 04_optimization_regularization.py
│   ├── 05_embeddings.py
│   ├── 06_rnn_language_model.py
│   ├── 07_self_attention.py
│   ├── 08_transformer_block.py
│   ├── 09_tokenizer_dataset.py
│   ├── 10_mini_gpt.py
│   ├── 11_lora_finetune.py
│   ├── 12_generation_evaluation.py
│   └── 13_training_engineering.py
└── tests/
    └── test_smoke.py
```

## 快速开始

建议在 PowerShell 中执行：

```powershell
cd C:\Users\Albrt\Desktop\learning\_llm\8-30-DL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\src\01_tensor_autograd.py
```

本机如果已经安装 PyTorch，可直接运行案例。全部快速实验：

```powershell
python .\run_all.py
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 推荐学习顺序

| 阶段 | 案例 | 核心问题 | 建议产出 |
|---|---|---|---|
| 基础 | 01–04 | 梯度从哪里来，参数为什么能学会？ | 手推一次梯度并对照 Autograd |
| 表示与序列 | 05–06 | 离散 token 如何变成连续表示？ | 比较 Embedding 与 one-hot |
| Transformer | 07–08 | 注意力如何混合上下文？ | 画出注意力矩阵并检查 mask |
| LLM 数据与模型 | 09–10 | GPT 如何把“预测下一个 token”变成生成？ | 训练迷你 GPT 并采样 |
| 高效适配与评估 | 11–12 | 如何低成本微调，如何判断生成质量？ | 比较全量训练与 LoRA 参数量 |
| 工程 | 13 | 显存不足、训练不稳怎么办？ | 观察裁剪前后梯度范数 |

## 使用建议

1. 先读 [深度学习总体介绍.md](./深度学习总体介绍.md)，再按编号运行脚本。
2. 每个脚本顶部均写明学习目标，主体含形状断言和关键注释。
3. 先保持默认参数完成 CPU 快速实验，再增加 `--steps`、`--epochs` 等参数。
4. 不要只看最终 loss；记录张量形状、梯度范数、训练/验证差距和生成样本。
5. 将每章的“思考题”写成实验报告，重点解释现象，不只粘贴输出。

## 课程边界

本项目强调可读性与原理验证，不追求生产级吞吐。真正的大模型训练还需要分布式并行、数据治理、检查点容错、监控、安全对齐和系统化评测等基础设施。

