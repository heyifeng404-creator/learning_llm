# 机器学习：从全局地图到研究实践

> 面向计算机专业研究生的中文机器学习开源式教程。目标不是背算法，而是建立一套能够迁移到论文阅读、实验设计和工程实践中的认知框架。

## 你会得到什么

- 一张覆盖监督学习、无监督学习、深度学习、概率模型与研究方法的总地图。
- 每个主题都有：直觉、数学表达、关键假设、代码、常见误区和延伸问题。
- 一套可运行的 Python 示例：既有 NumPy 手写算法，也有 scikit-learn / PyTorch 实践。
- 一个端到端实验模板，涵盖数据拆分、预处理、调参、评估、误差分析与复现。
- 面向研究生的论文阅读、选题、消融实验和科研写作指南。

## 机器学习的一句话定义

机器学习是在给定数据、假设空间和评价标准的条件下，通过优化方法寻找一个能在未见样本上表现良好的函数。

可以把整个领域压缩为下面的关系：

```text
现实问题
  ↓ 任务定义（输入 X、输出 y、约束）
数据生成过程 P(X, y)
  ↓ 采样、清洗、表示
训练数据 D
  ↓ 模型 fθ + 损失 L + 优化算法
学到参数 θ*
  ↓ 泛化评估、误差分析、部署监控
对未见数据作出预测或决策
```

研究中的大多数争论，都可以定位到五个问题：

1. **数据**：样本从哪里来，是否独立同分布，是否有偏差或泄漏？
2. **表示**：怎样把对象表示为模型可利用的特征？
3. **模型**：函数族具有什么归纳偏置和表达能力？
4. **学习**：用什么目标函数与优化算法，从有限样本得到参数？
5. **评估**：指标是否真的对应研究目标，结论是否可靠、可复现？

## 项目结构

```text
ml-for-graduate-students/
├── README.md                         # 入口、路线与全局框架
├── requirements.txt                 # 最小运行环境
├── docs/
│   ├── 00-knowledge-map.md           # 知识地图与术语系统
│   ├── 01-math-foundations.md        # 线代、概率、微积分、优化
│   ├── 02-learning-theory.md         # 风险、泛化、偏差-方差、正则化
│   ├── 03-ml-workflow.md             # 完整实验工作流与数据问题
│   ├── 04-supervised-learning.md     # 回归、分类、核方法、树与集成
│   ├── 05-unsupervised-learning.md   # 聚类、降维、密度与异常检测
│   ├── 06-neural-networks.md         # 神经网络、反向传播与训练
│   ├── 07-modern-deep-learning.md    # CNN、序列、Transformer、生成模型
│   ├── 08-probabilistic-ml.md        # MLE、MAP、贝叶斯与潜变量模型
│   ├── 09-evaluation-and-debugging.md# 指标、统计检验、误差分析
│   ├── 10-ml-engineering.md          # 复现、部署、漂移、MLOps
│   ├── 11-research-guide.md          # 论文、选题、实验与写作
│   ├── 12-cheatsheet.md              # 速查表、选型与面试式问答
│   └── 13-exercises-and-projects.md  # 分级练习、综合项目与评分量规
├── src/
│   ├── common.py                     # 通用数据和随机种子工具
│   ├── linear_models_from_scratch.py # NumPy 手写线性/逻辑回归
│   ├── classical_ml_demo.py          # 树、集成、SVM、调参与评估
│   ├── unsupervised_demo.py          # PCA、K-Means、异常检测
│   ├── neural_network_from_scratch.py# NumPy 手写两层神经网络
│   ├── pytorch_training_demo.py      # 规范 PyTorch 训练循环
│   └── end_to_end_experiment.py      # 防泄漏的端到端实验模板
├── tests/
│   └── test_smoke.py                 # 核心实现的冒烟测试
└── data/
    └── README.md                     # 数据管理约定
```

## 文档导航

| 章节 | 主题 | 读完应该能够 |
|---|---|---|
| [00](docs/00-knowledge-map.md) | 知识地图 | 把任意 ML 问题放进统一坐标系 |
| [01](docs/01-math-foundations.md) | 数学基础 | 看懂常见公式、梯度和概率表达 |
| [02](docs/02-learning-theory.md) | 学习理论 | 解释泛化、正则化和归纳偏置 |
| [03](docs/03-ml-workflow.md) | 实验工作流 | 设计无泄漏、可复现的实验 |
| [04](docs/04-supervised-learning.md) | 监督学习 | 比较线性、核、树与集成模型 |
| [05](docs/05-unsupervised-learning.md) | 无监督学习 | 理解聚类、降维与异常检测 |
| [06](docs/06-neural-networks.md) | 神经网络 | 手推反向传播并调试训练过程 |
| [07](docs/07-modern-deep-learning.md) | 现代深度学习 | 理解 CNN、Transformer 与生成模型 |
| [08](docs/08-probabilistic-ml.md) | 概率机器学习 | 区分 MLE、MAP、后验和近似推断 |
| [09](docs/09-evaluation-and-debugging.md) | 评估与调试 | 选指标、做区间估计和误差分析 |
| [10](docs/10-ml-engineering.md) | ML 工程 | 理解部署、监控、漂移和复现 |
| [11](docs/11-research-guide.md) | 科研指南 | 读论文、做消融、复现与写作 |
| [12](docs/12-cheatsheet.md) | 速查表 | 快速回顾公式、选型和自测问题 |
| [13](docs/13-exercises-and-projects.md) | 练习与项目 | 用实验、复现和综合项目形成能力 |

代码入口：[手写线性模型](src/linear_models_from_scratch.py) · [经典模型比较](src/classical_ml_demo.py) · [无监督学习](src/unsupervised_demo.py) · [手写神经网络](src/neural_network_from_scratch.py) · [PyTorch 训练](src/pytorch_training_demo.py) · [端到端实验](src/end_to_end_experiment.py)

## 推荐学习路线

### 路线 A：第一次系统学习（8–12 周）

| 周 | 主题 | 阅读 | 必做实验 |
|---|---|---|---|
| 1 | 全局地图、数学复习 | 00–01 | 推导最小二乘梯度 |
| 2 | 学习理论与实验流程 | 02–03 | 观察过拟合和正则化 |
| 3–4 | 监督学习 | 04 | 比较线性、树、SVM、集成 |
| 5 | 无监督学习 | 05 | PCA + K-Means 可视化 |
| 6 | 神经网络 | 06 | 手写反向传播并做梯度检查 |
| 7–8 | 现代深度学习 | 07 | 用 PyTorch 完成分类实验 |
| 9 | 概率机器学习 | 08 | 比较 MLE 与 MAP |
| 10 | 评估与调试 | 09 | 阈值、校准、置信区间 |
| 11 | 工程与复现 | 10 | 固定环境并生成实验记录 |
| 12 | 研究训练 | 11 | 复现一篇小论文并做消融 |

### 路线 B：已经上过课，准备科研

先读 `00 → 02 → 03 → 09 → 11`，再根据方向选择：

- 视觉：04、06、07 中的 CNN 与自监督学习。
- NLP / 大模型：06、07 中的注意力、Transformer、预训练与对齐。
- 数据挖掘：04、05、09，重点是树模型、表示、异常检测与可靠评估。
- 概率建模：01、02、08，补充图模型、变分推断与 MCMC。
- 系统与工程：03、09、10，重点是数据契约、实验追踪、服务与漂移。

## 快速开始

要求 Python 3.10 或更高版本。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/linear_models_from_scratch.py
python src/classical_ml_demo.py
python src/unsupervised_demo.py
python src/neural_network_from_scratch.py
python src/pytorch_training_demo.py
python src/end_to_end_experiment.py
python -m pytest -q
```

所有示例默认使用库自带或程序生成的数据，不需要联网下载。

## 学习每个算法时固定问八个问题

1. 它解决什么任务？输入与输出是什么？
2. 模型的函数形式是什么，参数是什么？
3. 损失函数为何这样定义？
4. 如何求解，时间与空间复杂度如何？
5. 它隐含什么统计假设和归纳偏置？
6. 哪些超参数最重要，如何选择？
7. 它会以什么方式失败，如何诊断？
8. 与相邻方法相比，何时应该选它？

## 建议的学习方式

不要只“看懂”。每一章至少做四件事：

- 用自己的话写 200 字总结。
- 不看资料推导一个关键公式。
- 修改代码制造一次失败，再解释失败原因。
- 设计一个对照实验，改变一个因素并报告结论。

## 记住这条主线

```text
训练误差低 ≠ 泛化好
指标高 ≠ 问题解决
相关性强 ≠ 因果关系
代码能跑 ≠ 实验可信
模型更大 ≠ 方法更科学
```

本教程的最终目标，是让你能把一个模糊的现实问题，转化为可验证、可复现、可解释的机器学习研究问题。
