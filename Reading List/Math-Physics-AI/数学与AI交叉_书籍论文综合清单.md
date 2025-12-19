# 数学与AI交叉：代数、拓扑、微分几何在深度学习中的应用

**综合清单：书籍与论文**

本清单展示代数、拓扑、微分几何与人工智能、神经网络、深度学习之间的新兴联系，特别关注几何深度学习、拓扑数据分析、流形学习等前沿方向。

---

## 目录

1. [学习路线图](#学习路线图)
2. [几何深度学习 (Geometric Deep Learning)](#几何深度学习)
3. [拓扑数据分析 (Topological Data Analysis)](#拓扑数据分析)
4. [流形学习与降维](#流形学习与降维)
5. [图神经网络的几何理论](#图神经网络的几何理论)
6. [神经网络的拓扑与几何性质](#神经网络的拓扑与几何性质)
7. [等变神经网络与群论](#等变神经网络与群论)
8. [信息几何与优化](#信息几何与优化)
9. [经典论文精选](#经典论文精选)

---

## 学习路线图

### 📍 路径概览

```
基础数学 → 深度学习基础 → 几何深度学习 → 前沿研究
  ↓            ↓                ↓              ↓
3-6个月      3-6个月          6-12个月       研究阶段
```

### 📍 初级阶段
**数学+机器学习基础**

**数学准备：**
- 线性代数（矩阵理论、特征值）
- 微积分与优化
- 概率论与统计
- 图论基础

**机器学习：**
- 传统ML（分类、回归）
- 神经网络基础
- 深度学习入门

**推荐书籍：**
- Goodfellow《Deep Learning》
- Bishop《Pattern Recognition and Machine Learning》

### 📍 中级阶段
**几何与拓扑基础**

**数学深化：**
- 微分几何入门（流形、切空间）
- 代数拓扑初步（同调、同伦）
- 图论与谱方法
- 李群初步

**深度学习进阶：**
- CNN、RNN、Transformer
- 图神经网络
- 生成模型

**推荐书籍：**
- Lee《Introduction to Smooth Manifolds》（选读）
- Bronstein et al.《Geometric Deep Learning》
- Hamilton《Graph Representation Learning》

### 📍 高级阶段
**前沿研究方向**

选择专门方向：
- 拓扑神经网络
- 等变神经网络
- 流形上的深度学习
- 持久同调与TDA

---

## 几何深度学习

### 📚 核心书籍

#### 入门级
- [ ] **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges** - *Bronstein, Bruna, Cohen, Veličković* ⭐🔥
  - **GDL领域的奠基性著作**
  - 统一CNN/GNN/Transformer的几何视角
  - 基于Erlangen纲领
  - 免费在线版：https://geometricdeeplearning.com/
  - 2021

- [ ] **Graph Representation Learning** - *William Hamilton* ⭐🔥
  - 图表示学习综合教材
  - 从传统方法到GNN
  - 免费在线版
  - *Morgan & Claypool*, 2020

#### 中级
- [ ] **Deep Learning on Graphs** - *Yao Ma, Jiliang Tang* 🔥
  - 图深度学习全面介绍
  - 理论与实践结合
  - *Cambridge University Press*, 2021

- [ ] **Equivariant and Coordinate Independent Convolutional Networks** - *Maurice Weiler et al.* 🔥
  - 等变CNN的规范场理论
  - 群论与深度学习
  - 在线草稿

### 📚 相关数学书籍

#### 流形与微分几何
- [ ] **Introduction to Smooth Manifolds** - *John M. Lee* ⭐
  - 现代微分几何标准教材
  - 为理解流形学习做准备
  - *Springer GTM 218*, 2012

- [ ] **Riemannian Geometry** - *Manfredo do Carmo* ⭐
  - Riemannian几何经典
  - 理解流形上的度量学习
  - *Birkhäuser*, 1992

#### 图论与谱方法
- [ ] **Spectral Graph Theory** - *Fan Chung* ⭐
  - 谱图理论
  - GNN的数学基础
  - *AMS*, 1997

- [ ] **A Course in Metric Geometry** - *Burago, Burago, Ivanov* 🔥
  - 度量几何
  - 图的几何性质
  - *AMS GSM 33*, 2001

### 📄 经典论文

#### GDL综述
- [ ] **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges** - *Bronstein et al.* (2021) ⭐🔥
  - GDL的统一框架
  - 150+页综述
  - *arXiv:2104.13478*

- [ ] **Geometric Deep Learning: Going beyond Euclidean data** - *Bronstein et al.* (2017) 🔥
  - 早期GDL综述
  - *IEEE Signal Processing Magazine*

#### 图神经网络
- [ ] **The Graph Neural Network Model** - *Scarselli et al.* (2009) ⭐
  - GNN概念提出
  - *IEEE Transactions on Neural Networks* 20

- [ ] **Semi-Supervised Classification with Graph Convolutional Networks** - *Kipf & Welling* (2017) ⭐🔥
  - GCN
  - 谱方法到空间方法
  - *ICLR 2017*

- [ ] **Inductive Representation Learning on Large Graphs** - *Hamilton, Ying, Leskovec* (2017) 🔥
  - GraphSAGE
  - 归纳式学习
  - *NIPS 2017*

- [ ] **Graph Attention Networks** - *Veličković et al.* (2018) 🔥
  - GAT
  - 注意力机制
  - *ICLR 2018*

- [ ] **How Powerful are Graph Neural Networks?** - *Xu et al.* (2019) 🔥
  - GNN表达能力分析
  - Weisfeiler-Lehman test
  - *ICLR 2019*

---

## 拓扑数据分析

### 📚 核心书籍

#### 入门级
- [ ] **Computational Topology: An Introduction** - *Edelsbrunner & Harer* ⭐🔥
  - 计算拓扑入门
  - 持久同调基础
  - *AMS*, 2010

- [ ] **Topological Data Analysis for Scientific Visualization** - *Julien Tierny (Ed.)* 🔥
  - TDA应用导向
  - *Springer*, 2017

#### 中级
- [ ] **Elementary Applied Topology** - *Robert Ghrist* ⭐
  - 应用拓扑简明教材
  - 免费在线版
  - 2014

- [ ] **Topology for Computing** - *Afra Zomorodian* 🔥
  - 计算拓扑
  - *Cambridge Monographs*, 2005

#### 高级
- [ ] **Algebraic Topology** - *Allen Hatcher* ⭐
  - 代数拓扑标准教材
  - 免费在线
  - *Cambridge University Press*, 2002

- [ ] **Persistent Homology** - *Various Authors* 📊
  - 综述文章合集
  - *Handbook of Discrete and Computational Geometry*

### 📄 经典论文

#### 持久同调奠基
- [ ] **Topological Persistence and Simplification** - *Edelsbrunner, Letscher, Zomorodian* (2002) ⭐
  - 持久同调理论
  - *Discrete & Computational Geometry* 28

- [ ] **Computing Persistent Homology** - *Zomorodian & Carlsson* (2005) ⭐
  - 持久同调算法
  - *Discrete & Computational Geometry* 33

- [ ] **Stability of Persistence Diagrams** - *Cohen-Steiner, Edelsbrunner, Harer* (2007) ⭐🔥
  - 稳定性定理
  - *Discrete & Computational Geometry* 37

#### TDA与机器学习
- [ ] **A roadmap for the computation of persistent homology** - *Nina Otter et al.* (2017) 📊
  - TDA计算方法综述
  - *EPJ Data Science* 6

- [ ] **Topological Data Analysis** - *Larry Wasserman* (2018) 📊
  - 统计学视角
  - *Annual Review of Statistics* 5

- [ ] **Persistence Images: A Stable Vector Representation of Persistent Homology** - *Adams et al.* (2017) 🔥
  - 持久同调向量化
  - 可用于深度学习
  - *JMLR* 18

#### 拓扑神经网络
- [ ] **Topological Deep Learning: Going Beyond Graph Data** - *Hajij et al.* (2023) ⭐🔥
  - 拓扑深度学习综述
  - Cell complexes与neural networks
  - *arXiv:2206.00606*

- [ ] **A Survey on Topological Deep Learning** - *Papillon et al.* (2023) 📊
  - TDL最新综述
  - *arXiv:2304.10031*

---

## 流形学习与降维

### 📚 核心书籍

#### 入门级
- [ ] **The Elements of Statistical Learning** - *Hastie, Tibshirani, Friedman* ⭐
  - 包含降维方法章节
  - 免费在线版
  - *Springer*, 2nd ed. 2009

- [ ] **Manifold Learning Theory and Applications** - *Yunqian Ma, Yun Fu* 🔥
  - 流形学习专著
  - *CRC Press*, 2011

### 📄 经典论文

#### 经典算法
- [ ] **Nonlinear Dimensionality Reduction by Locally Linear Embedding** - *Roweis & Saul* (2000) ⭐🔥
  - LLE算法
  - *Science* 290

- [ ] **A Global Geometric Framework for Nonlinear Dimensionality Reduction** - *Tenenbaum, de Silva, Langford* (2000) ⭐🔥
  - Isomap算法
  - *Science* 290

- [ ] **Laplacian Eigenmaps for Dimensionality Reduction and Data Representation** - *Belkin & Niyogi* (2003) 🔥
  - Laplacian特征映射
  - *Neural Computation* 15

- [ ] **Visualizing Data using t-SNE** - *van der Maaten & Hinton* (2008) ⭐🔥
  - t-SNE
  - 最常用可视化方法
  - *JMLR* 9

- [ ] **UMAP: Uniform Manifold Approximation and Projection** - *McInnes, Healy, Melville* (2018) 🔥
  - UMAP
  - 基于Riemannian几何
  - *arXiv:1802.03426*

#### 理论分析
- [ ] **Manifold Regularization** - *Belkin, Niyogi, Sindhwani* (2006) 🔥
  - 流形正则化
  - *JMLR* 7

- [ ] **Semi-Supervised Learning on Riemannian Manifolds** - *Belkin & Niyogi* (2004) 🔥
  - Riemann流形上的学习
  - *Machine Learning* 56

---

## 图神经网络的几何理论

### 📚 书籍

#### 中级
- [ ] **Graph Neural Networks: Foundations, Frontiers, and Applications** - *Wu, Pan, Chen, Long, Zhang, Yu (Eds.)* 📊
  - GNN综合参考
  - *Springer*, 2022

### 📄 经典论文

#### 几何GNN
- [ ] **E(n) Equivariant Graph Neural Networks** - *Satorras, Hoogeboom, Welling* (2021) ⭐🔥
  - E(n)等变GNN
  - 3D分子建模
  - *ICML 2021*

- [ ] **Geometric Vector Perceptrons** - *Jing et al.* (2021) 🔥
  - GVP
  - 几何向量表示
  - *ICLR 2021*

- [ ] **SE(3)-Transformers** - *Fuchs et al.* (2020) 🔥
  - SE(3)等变Transformer
  - *NeurIPS 2020*

#### 谱方法
- [ ] **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering** - *Defferrard, Bresson, Vandergheynst* (2016) 🔥
  - ChebNet
  - Chebyshev多项式
  - *NIPS 2016*

- [ ] **Spectral Networks and Deep Locally Connected Networks on Graphs** - *Bruna et al.* (2014) ⭐
  - 谱图卷积
  - *ICLR 2014*

#### 消息传递框架
- [ ] **Neural Message Passing for Quantum Chemistry** - *Gilmer et al.* (2017) 🔥
  - MPNN框架
  - *ICML 2017*

- [ ] **Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks** - *Morris et al.* (2019) 🔥
  - k-WL与高阶GNN
  - *AAAI 2019*

---

## 神经网络的拓扑与几何性质

### 📚 书籍

#### 理论书籍
- [ ] **The Principles of Deep Learning Theory** - *Roberts, Yaida, Hanin* ⭐🔥
  - 深度学习理论原理
  - 场论方法
  - *Cambridge University Press*, 2022

- [ ] **High-Dimensional Probability** - *Roman Vershynin* 🔥
  - 高维概率
  - 神经网络分析工具
  - *Cambridge Series*, 2018

### 📄 经典论文

#### 损失landscape几何
- [ ] **Visualizing the Loss Landscape of Neural Nets** - *Li et al.* (2018) 🔥
  - 损失函数可视化
  - *NeurIPS 2018*

- [ ] **The Loss Surfaces of Multilayer Networks** - *Choromanska et al.* (2015) 🔥
  - 损失surface拓扑
  - *AISTATS 2015*

- [ ] **Mode Connectivity in Loss Landscapes** - *Garipov et al.* (2018) 🔥
  - 模式连通性
  - *NeurIPS 2018*

#### 神经网络表达能力
- [ ] **On the Number of Linear Regions of Deep Neural Networks** - *Montúfar et al.* (2014) 🔥
  - ReLU网络的分段线性结构
  - *NIPS 2014*

- [ ] **The Power of Depth for Feedforward Neural Networks** - *Eldan & Shamir* (2016) 🔥
  - 深度的表达能力
  - *COLT 2016*

#### 拓扑分析
- [ ] **Topology and Data** - *Gunnar Carlsson* (2009) ⭐📊
  - TDA在数据科学中的应用
  - *Bulletin of the AMS* 46

- [ ] **A Topological Regularizer for Classifiers via Persistent Homology** - *Chen et al.* (2019) 🔥
  - 拓扑正则化
  - *AISTATS 2019*

---

## 等变神经网络与群论

### 📚 核心书籍

#### 群论基础
- [ ] **Lie Groups, Lie Algebras, and Representations** - *Brian Hall* ⭐
  - 李群入门
  - 物理与数学双重视角
  - *Springer GTM 222*, 2015

- [ ] **Group Theory in Physics** - *Wu-Ki Tung* ⭐
  - 物理中的群论
  - *World Scientific*, 1985

#### 等变学习
- [ ] **Equivariant Neural Networks** - *Maurice Weiler* (PhD Thesis) ⭐🔥
  - 等变神经网络理论
  - 规范场视角
  - University of Amsterdam, 2021

### 📄 经典论文

#### 等变CNN
- [ ] **Group Equivariant Convolutional Networks** - *Cohen & Welling* (2016) ⭐🔥
  - G-CNN
  - 群等变卷积
  - *ICML 2016*

- [ ] **Steerable CNNs** - *Cohen & Welling* (2017) 🔥
  - 可操纵CNN
  - *ICLR 2017*

- [ ] **3D Steerable CNNs** - *Weiler, Geiger, Welling* (2018) 🔥
  - 3D等变CNN
  - *NeurIPS 2018*

- [ ] **A General Theory of Equivariant CNNs on Homogeneous Spaces** - *Cohen, Weiler, Kicanaoglu, Welling* (2019) ⭐
  - 齐性空间上的等变CNN
  - *NeurIPS 2019*

#### 等变Transformer
- [ ] **Equivariant Transformers for Neural Network based Molecular Potentials** - *Thölke & De Fabritiis* (2022) 🔥
  - 等变Transformer
  - *ICLR 2022*

#### 对称性与泛化
- [ ] **A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels** - *Kondor & Trivedi* (2018) 🔥
  - Wigner-Eckart定理
  - *ICML 2018*

---

## 信息几何与优化

### 📚 核心书籍

#### 信息几何
- [ ] **Information Geometry and Its Applications** - *Shun-ichi Amari* ⭐
  - 信息几何大师之作
  - *Springer AMS 194*, 2016

- [ ] **Methods of Information Geometry** - *Amari & Nagaoka* ⭐
  - 信息几何方法
  - *AMS/Oxford*, 2000

#### 优化理论
- [ ] **Convex Optimization** - *Boyd & Vandenberghe* ⭐
  - 凸优化经典
  - 免费在线版
  - *Cambridge University Press*, 2004

- [ ] **Optimization for Machine Learning** - *Sra, Nowozin, Wright (Eds.)* 🔥
  - ML优化综述
  - *MIT Press*, 2012

### 📄 经典论文

#### 自然梯度
- [ ] **Natural Gradient Works Efficiently in Learning** - *Shun-ichi Amari* (1998) ⭐🔥
  - 自然梯度方法
  - 信息几何视角
  - *Neural Computation* 10

- [ ] **Practical Riemannian Neural Networks** - *Lezcano-Casado* (2019) 🔥
  - Riemann流形上的优化
  - *arXiv:1906.05485*

#### Adam与变种
- [ ] **Adam: A Method for Stochastic Optimization** - *Kingma & Ba* (2014) ⭐🔥
  - Adam优化器
  - *ICLR 2015*

---

## 经典论文精选

### 🌟 必读综述

#### 几何深度学习
- [ ] **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges** - *Bronstein et al.* (2021) ⭐🔥
  - **最重要的GDL综述**
  - 统一框架
  - *arXiv:2104.13478*

#### 图神经网络
- [ ] **A Comprehensive Survey on Graph Neural Networks** - *Wu et al.* (2020) 📊
  - GNN全面综述
  - *IEEE TNNLS* 32

- [ ] **Graph Neural Networks: A Review of Methods and Applications** - *Zhou et al.* (2020) 📊
  - GNN方法与应用
  - *AI Open* 1

#### 拓扑深度学习
- [ ] **Topological Deep Learning: Going Beyond Graph Data** - *Hajij et al.* (2023) ⭐🔥
  - TDL综述
  - *arXiv:2206.00606*

### 🔬 前沿研究方向

#### Transformer的几何
- [ ] **Attention is All you Need** - *Vaswani et al.* (2017) ⭐🔥
  - Transformer奠基
  - *NIPS 2017*

- [ ] **Do Transformers Really Perform Bad for Graph Representation?** - *Ying et al.* (2021) 🔥
  - Graphormer
  - *NeurIPS 2021*

#### 持久同调与深度学习
- [ ] **A Topology Layer for Machine Learning** - *Gabrielsson & Carlsson* (2019) 🔥
  - 拓扑层
  - *AISTATS 2019*

- [ ] **Neural Persistence** - *Hofer et al.* (2019) 🔥
  - 可微持久同调
  - *ICML 2019*

#### Clifford代数与神经网络
- [ ] **Clifford Neural Layers** - *Brandstetter et al.* (2023) 🔥
  - Clifford代数神经网络
  - *ICML 2023*

- [ ] **Geometric Clifford Algebra Networks** - *Ruhe et al.* (2023) 🔥
  - 几何Clifford网络
  - *ICML 2023*

---

## 软件与工具

### 🛠️ GNN库
- **PyTorch Geometric (PyG)** ⭐
  - 最流行的GNN库
  - https://pytorch-geometric.readthedocs.io/

- **DGL (Deep Graph Library)** ⭐
  - 另一个主流GNN库
  - https://www.dgl.ai/

- **Jraph** 🔥
  - JAX的GNN库
  - https://github.com/deepmind/jraph

### 🛠️ 拓扑数据分析
- **Giotto-TDA** ⭐
  - 机器学习的TDA
  - https://giotto-ai.github.io/gtda-docs/

- **Ripser** 🔥
  - 快速持久同调计算
  - https://github.com/Ripser/ripser

- **TopoNetX** 🔥
  - 拓扑神经网络
  - https://github.com/pyt-team/TopoNetX

### 🛠️ 等变神经网络
- **e3nn** ⭐
  - E(3)等变神经网络
  - https://e3nn.org/

- **escnn** 🔥
  - 等变可操纵CNN
  - https://github.com/QUVA-Lab/escnn

---

## 学习资源

### 📚 在线课程
- **Stanford CS224W: Machine Learning with Graphs** (Jure Leskovec)
- **Cambridge: Geometric Deep Learning** (Bronstein, Veličković)
- **MIT: Machine Learning for Molecules and Materials** (Rafael Gomez-Bombarelli)

### 📖 博客与教程
- **Distill.pub** - 机器学习可视化
- **Geometric Deep Learning Blog** - geometricdeeplearning.com
- **PyG Tutorials** - 官方教程

### 🎥 视频资源
- **ICML/NeurIPS Tutorial Videos**
- **Geometric Deep Learning Course (YouTube)** - Michael Bronstein
- **Graph Neural Networks (YouTube)** - Petar Veličković

---

## 研究前沿

### 🔥 热门方向

#### 1. 拓扑神经网络 (TNN)
**核心问题：**
- Cell complexes上的消息传递
- 高阶交互建模
- 拓扑不变量学习

**关键论文：**
- Hajij et al. (2023) - TDL综述
- Bodnar et al. (2021) - Cell Attention Networks

#### 2. 等变深度学习
**核心问题：**
- 更一般群的等变性
- 等变Transformer
- 物理对称性融入

**关键论文：**
- Weiler (2021) - PhD thesis
- Fuchs et al. (2020) - SE(3)-Transformers

#### 3. 几何Transformer
**核心问题：**
- 将Transformer推广到非欧空间
- 注意力机制的几何解释
- 位置编码的几何意义

**关键论文：**
- Graphormer (2021)
- GPS++ (2023)

#### 4. 持久同调与深度学习
**核心问题：**
- 可微持久同调
- 拓扑正则化
- 拓扑损失函数

**关键论文：**
- Hofer et al. (2019) - Neural Persistence
- Chen et al. (2019) - Topological Regularizer

#### 5. 流形上的深度学习
**核心问题：**
- Riemannian流形上的优化
- 测地线注意力
- 流形生成模型

**关键论文：**
- Lou et al. (2020) - Neural Manifold ODEs
- Bose et al. (2020) - Latent Variable Modelling with Hyperbolic Normalizin Flows

---

## 核心概念对应表

| 数学概念 | 深度学习对应 | 应用 |
|---------|-------------|------|
| 流形 | 数据空间 | 流形学习 |
| 切空间 | 特征空间 | 特征表示 |
| 测地线 | 最短路径 | 图最短路 |
| 曲率 | 数据复杂度 | 数据分析 |
| 同调群 | 拓扑特征 | TDA |
| 持久同调 | 多尺度特征 | 特征提取 |
| 纤维丛 | 层级结构 | 深度架构 |
| 李群 | 对称群 | 等变网络 |
| 群作用 | 数据增强 | 等变性 |
| 图Laplacian | 图卷积 | GCN |
| 谱分解 | 特征分解 | 谱方法 |
| Riemann度量 | 信息度量 | 信息几何 |
| Fisher信息矩阵 | Hessian | 自然梯度 |

---

## 学习建议

### 📖 学习策略

#### 路径1：从机器学习出发
**适合AI/CS背景**

1. **深度学习基础** (3个月)
   - Goodfellow《Deep Learning》
   - 实现基本模型

2. **图神经网络** (3个月)
   - Hamilton《Graph Representation Learning》
   - PyG教程
   - 实现GCN/GAT

3. **几何视角** (6个月)
   - Bronstein et al. GDL综述
   - 学习群论、流形基础
   - 实现等变网络

4. **拓扑方法** (6个月)
   - Edelsbrunner & Harer
   - TDA Python库
   - 拓扑特征提取

#### 路径2：从数学出发
**适合数学背景**

1. **数学基础** (已有)
   - 微分几何
   - 代数拓扑
   - 李群

2. **机器学习** (3个月)
   - Bishop《PRML》
   - 基本算法实现

3. **深度学习** (3个月)
   - 神经网络基础
   - 反向传播
   - PyTorch

4. **几何深度学习** (6-12个月)
   - 直接学习GDL
   - 阅读前沿论文
   - 参与研究项目

### 🎯 实践建议

1. **编程实践**
   - 使用PyG实现GNN
   - 复现经典论文
   - 参与Kaggle竞赛

2. **理论学习**
   - 并行阅读数学和AI论文
   - 做笔记建立联系
   - 参加seminar

3. **项目导向**
   - 选择感兴趣的应用
   - 分子性质预测
   - 社交网络分析
   - 3D物体识别

---

## 会议与期刊

### 📊 主要会议
- **NeurIPS** - ML顶会，大量GDL论文
- **ICML** - ML理论强
- **ICLR** - 表示学习
- **CVPR** - 计算机视觉中的几何
- **KDD** - 图挖掘
- **AAAI/IJCAI** - AI综合

### 📖 相关期刊
- **JMLR** - 机器学习
- **IEEE TPAMI** - 模式识别
- **Neural Computation** - 神经计算
- **Foundations and Trends in ML** - 综述

---

**最后更新：** 2025年12月

**总结：**
数学与AI的交叉是当前最活跃的研究前沿之一。几何深度学习提供了统一的框架，拓扑数据分析带来新的工具，等变神经网络融入物理先验。这个领域需要扎实的数学基础和深度学习实践经验，但回报丰厚，充满机遇！
