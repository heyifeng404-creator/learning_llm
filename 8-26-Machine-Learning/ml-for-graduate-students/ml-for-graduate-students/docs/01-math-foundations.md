# 01｜够用且有直觉的数学基础

## 1. 线性代数：模型的语言

### 向量、矩阵与张量

- 标量：单个数。
- 向量 \(x\in\mathbb R^d\)：一个样本的 \(d\) 个特征。
- 矩阵 \(X\in\mathbb R^{n\times d}\)：\(n\) 个样本组成的数据表。
- 张量：更高维数组，例如图像批次形状为 `batch × channel × height × width`。

内积 \(x^Tw\) 可理解为“输入与模板的匹配程度”。范数度量大小：

\[
\lVert x\rVert_1=\sum_i|x_i|,\qquad
\lVert x\rVert_2=\sqrt{\sum_i x_i^2}
\]

L1 正则倾向产生稀疏参数；L2 正则平滑地收缩所有参数。

### 线性变换与基

矩阵乘法 \(Ax\) 是对向量进行旋转、缩放、剪切和投影。很多“特征提取”都可以理解为变换坐标系，使重要结构更容易被后续模型利用。

### 特征值、奇异值与 PCA

对称矩阵可写为 \(A=Q\Lambda Q^T\)。特征向量给出不改变方向的轴，特征值给出沿轴的缩放。任意矩阵有奇异值分解：

\[
X=U\Sigma V^T
\]

PCA 取中心化数据的前几个右奇异向量作为主方向，从而以最小重构误差进行线性降维。

### 数值稳定性

不要显式求逆来解最小二乘。优先使用 `solve`、QR 或 SVD。矩阵条件数很大时，输入的小扰动会造成解的大变化；标准化和正则化往往能改善问题。

```python
import numpy as np

# 不推荐：w = np.linalg.inv(X.T @ X) @ X.T @ y
w, *_ = np.linalg.lstsq(X, y, rcond=None)
```

## 2. 概率论：处理不确定性

### 基本对象

- 随机变量：结果未知的量。
- 概率分布：不同结果出现的规律。
- 期望 \(\mathbb E[X]\)：长期平均。
- 方差 \(\mathrm{Var}(X)\)：围绕均值的波动。
- 协方差：两个变量是否共同变化；不代表因果。

联合、边缘与条件概率：

\[
P(x,y)=P(y\mid x)P(x),\qquad P(x)=\sum_y P(x,y)
\]

### 贝叶斯公式

\[
P(\theta\mid D)=\frac{P(D\mid\theta)P(\theta)}{P(D)}
\]

- 先验 \(P(\theta)\)：观察数据前的知识。
- 似然 \(P(D\mid\theta)\)：参数对观测数据的解释程度。
- 后验 \(P(\theta\mid D)\)：观察数据后的更新认识。
- 证据 \(P(D)\)：对所有参数解释能力的加权平均。

贝叶斯公式不是“主观相信”，而是一致的不确定性更新规则。

### 常见分布

| 分布 | 适用对象 | 机器学习中的位置 |
|---|---|---|
| Bernoulli | 0/1 结果 | 二分类似然 |
| Categorical | 多类别 | softmax 输出 |
| Gaussian | 连续量 | 回归噪声、潜变量 |
| Binomial | 多次独立成功次数 | 计数概率 |
| Poisson | 固定区间事件数 | 计数回归 |
| Beta/Dirichlet | 概率本身 | Bernoulli/Categorical 共轭先验 |

### 信息论

熵衡量分布的不确定性：

\[
H(P)=-\sum_xP(x)\log P(x)
\]

交叉熵 \(H(P,Q)=-\sum_xP(x)\log Q(x)\) 衡量用 \(Q\) 编码来自 \(P\) 的数据所需代价。KL 散度：

\[
D_{KL}(P\Vert Q)=H(P,Q)-H(P)
\]

它非负但不对称，不是严格的距离。分类交叉熵本质上是负对数似然。

## 3. 微积分：学习变化方向

导数表示局部变化率，梯度 \(\nabla_\theta L\) 收集损失对每个参数的偏导。梯度的反方向是局部下降最快方向：

\[
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
\]

链式法则是反向传播的核心：若 \(z=f(x),L=g(z)\)，则

\[
\frac{dL}{dx}=\frac{dL}{dz}\frac{dz}{dx}
\]

Hessian 矩阵记录二阶曲率。正定意味着局部凸；特征值相差悬殊意味着不同方向尺度不同，梯度下降会曲折缓慢。

## 4. 优化：怎样找到参数

### 凸与非凸

凸函数任意局部极小值都是全局极小值，线性回归和逻辑回归的标准目标属于凸优化。神经网络通常非凸，但大规模随机优化在实践中仍能找到泛化良好的解。

### 常用优化器

- 批量梯度下降：每步用全数据，稳定但昂贵。
- SGD：每步用一个或小批样本，噪声大但可扩展。
- Momentum：累计方向，减少震荡。
- Adam：对每个参数自适应调整步长，容易作为起点。
- AdamW：把权重衰减与梯度更新正确解耦，Transformer 中常见。

### 学习率比优化器名字更重要

学习率过大导致发散，过小导致收敛慢或困在平台。常配合 warmup、余弦衰减或分段衰减。需要同时观察训练损失、验证损失和梯度范数。

## 5. 矩阵微分速记

若 \(L=\frac1n\lVert Xw-y\rVert_2^2\)，则

\[
\nabla_wL=\frac{2}{n}X^T(Xw-y)
\]

若 \(z=Xw+b\)，上游梯度为 \(G=\partial L/\partial z\)，则

\[
\frac{\partial L}{\partial w}=X^TG,\quad
\frac{\partial L}{\partial X}=Gw^T,\quad
\frac{\partial L}{\partial b}=\sum_iG_i
\]

记住形状检查：导数与被求导变量形状一致。

## 6. 数值计算常见坑

- `log(0)`：给概率加小量或使用稳定库函数。
- softmax 溢出：先减去最大 logit。
- 浮点比较：用容差，不直接比较相等。
- 梯度爆炸：裁剪梯度、调整初始化或归一化。
- 梯度消失：合适激活、残差连接、归一化。

稳定 softmax：

```python
def softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)
```

## 7. 最低掌握标准

你不必成为数学专业学生，但应能：

- 检查矩阵形状并推导线性模型梯度。
- 从似然得到负对数似然损失。
- 解释期望、方差、条件概率和 Bayes 公式。
- 解释正则化、凸性、学习率与数值稳定性。
- 看懂论文中的常见目标函数，并能判断每一项的作用。

