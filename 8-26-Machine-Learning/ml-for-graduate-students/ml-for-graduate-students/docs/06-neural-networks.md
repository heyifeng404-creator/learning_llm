# 06｜神经网络：从计算图到训练

## 1. 多层感知机

一层神经元做仿射变换再通过非线性激活：

\[
h=\phi(Wx+b)
\]

多层网络把简单变换复合：

\[
f(x)=W_L\phi(W_{L-1}\cdots\phi(W_1x+b_1))+b_L
\]

如果没有非线性，多层线性变换仍等价于一层线性变换。激活函数让网络能够表达弯曲、分段和复杂交互。

## 2. 常见激活函数

| 激活 | 形式/特征 | 使用建议 |
|---|---|---|
| Sigmoid | 输出 0–1，易饱和 | 二分类输出概率 |
| Tanh | 零中心，仍会饱和 | 传统 RNN 状态 |
| ReLU | \(\max(0,x)\) | 简单高效，经典默认 |
| Leaky ReLU | 负半轴保留小斜率 | 缓解死亡 ReLU |
| GELU/SiLU | 平滑门控 | Transformer/现代网络常见 |

分类最后一层通常输出 logits；交叉熵函数内部做稳定 softmax。不要在 logits 上手动 softmax 后再传给要求 logits 的损失。

## 3. 前向传播与反向传播

前向传播计算中间变量和损失；反向传播沿计算图反向使用链式法则，复用局部导数，高效获得所有参数的梯度。

以两层网络为例：

\[
Z_1=XW_1+b_1,\quad H=\mathrm{ReLU}(Z_1),\quad Z_2=HW_2+b_2
\]

若上游梯度为 \(G_2=\partial L/\partial Z_2\)，则：

\[
\frac{\partial L}{\partial W_2}=H^TG_2,
\quad
G_1=(G_2W_2^T)\odot\mathbb 1[Z_1>0],
\quad
\frac{\partial L}{\partial W_1}=X^TG_1
\]

自动微分负责机械计算，但研究者仍需理解形状、梯度流与数值稳定性。

## 4. 梯度检查

用有限差分验证手写反向传播：

\[
\frac{\partial L}{\partial\theta_j}\approx
\frac{L(\theta_j+\epsilon)-L(\theta_j-\epsilon)}{2\epsilon}
\]

相对误差应很小。检查时使用双精度、小网络、固定数据，关闭 dropout 等随机操作。有限差分计算昂贵，只用于调试。

## 5. 参数初始化

若所有神经元权重相同，它们会得到相同梯度，无法学习不同特征。随机初始化打破对称性。

- Xavier/Glorot：适合 tanh 等近似对称激活。
- He/Kaiming：适合 ReLU，方差约为 \(2/\text{fan-in}\)。

目标是让前向激活和反向梯度在各层尺度相对稳定。

## 6. 归一化

- BatchNorm：用批次统计量归一化，CNN 常见；训练与推理行为不同。
- LayerNorm：沿特征维归一化，不依赖批大小，Transformer 标准组件。
- RMSNorm：不减均值的简化形式，在大模型中常见。

归一化既改善优化，也改变模型的隐式正则化和参数化行为。

## 7. 正则化

- 权重衰减：限制参数规模。
- Dropout：训练时随机屏蔽神经元，减少共适应。
- 数据增强：注入问题认可的不变性。
- Label smoothing：不让分类目标过度确信。
- 早停：验证集不再改善时停止。

数据增强必须保持标签语义。例如水平翻转可能适合自然图像，却不一定适合文字或医学影像。

## 8. 一个规范训练循环

```python
for inputs, targets in train_loader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

验证时必须调用 `model.eval()` 和 `torch.no_grad()`，否则 dropout 与 BatchNorm 行为不正确且浪费显存。

## 9. 训练故障诊断

| 现象 | 可能原因 | 优先检查 |
|---|---|---|
| 损失不降 | 学习率、标签、梯度、激活 | 尝试过拟合 20 个样本 |
| 损失 NaN | 溢出、学习率过大、坏数据 | 输入范围、梯度范数 |
| 训练好验证差 | 过拟合、泄漏、分布不同 | 数据拆分、正则、增强 |
| 训练极慢 | 数据管线、设备、模型瓶颈 | profiler、批大小 |
| 准确率卡在随机水平 | 标签映射或输出维错误 | 单批次逐项检查 |

“先让模型过拟合一个很小的数据子集”是最有效的调试步骤之一。如果连小数据都记不住，通常是实现或优化有问题，而非泛化问题。

## 10. 代码入口

- `src/neural_network_from_scratch.py`：仅用 NumPy 实现两层网络和反向传播。
- `src/pytorch_training_demo.py`：完整 PyTorch 数据加载、训练、验证、早停和评估。

