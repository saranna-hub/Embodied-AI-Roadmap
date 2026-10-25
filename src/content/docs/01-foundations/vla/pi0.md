---
title: π₀：从视觉语言到连续机器人动作（深度教程）
description: π₀ 双专家架构、Token、Attention Mask、Flow Matching 数学推导、PyTorch 示例、源码和实机迁移
sidebar:
  order: 1
---

> **本篇沿用此前详细教学版，保留公式、张量和工程解释；页面通过 Astro Starlight 渲染，示意图为教学图，未完成真实机器人复现实验。**

> **源码级补充（2026-10-25）**：新增 [π₀ × π₀.₅ 联合注意力、Mask、KV Cache、AdaRMS 专题](./pi0-joint-attention-code-audit/)，逐行核查官方 JAX 代码、头维度、训练/推理差异与动作块注意力的双向性。

## 学习前先补这些基础

如果你还不熟悉矩阵、反向传播、Attention 或 Flow Matching，推荐依次阅读：[张量与反向传播](../../../00-prerequisites/tensors-training/) → [Transformer 与 QKV](../../../00-prerequisites/transformer/) → [视觉 Backbone 与 VLM](../../../00-prerequisites/vision/) → [Flow Matching](../../../00-prerequisites/flow-matching/)。先搞清这些机制，再看 π₀ 的两组专家与连续动作输出，会容易得多。

## 可视化导读：沿着数据流理解 π₀

> **阅读路线**：先明确动作块是什么 → 追踪图像与语言如何形成 Prefix → 观察 State、Noisy Action、Time 如何形成 Suffix → 推导 Flow Matching 的监督 → 阅读 openpi 源码 → 设计机器人实验。下面三图是教学重绘，不是网络实现的逐层精确拓扑图。

![π₀ 模型整体架构：图像、指令、状态、带噪动作的详细输入输出，以及训练和推理的数据流](/Embodied-AI-Roadmap/diagrams/pi0-flow.svg)

*图 1｜π₀ 的完整数据流、双专家计算、训练监督与推理采样。建议在桌面端放大查看；下表按编号逐项解释。*

### 如何读总图：每条数据具体经过什么处理？

这一版把**前向计算主干、训练时的监督支路、推理时的采样支路**分开画。先读上半部分（①到④），再根据当前任务选择训练（⑤）或推理（⑥），不能把训练用的真实动作当成推理输入。

| 编号 | 输入 | 处理模块 | 输出与形状（示例） | 哪一步使用 |
| --- | --- | --- | --- | --- |
| ① 图像 | 多路 RGB `[B,N,3,H,W]` | 逐相机预处理 → SigLIP 视觉编码 → patch/token embedding | 视觉 token 序列（token 数由编码器与图像尺寸决定） | 训练/推理 |
| ① 语言 | 指令字符串 | tokenizer → token embedding | `[B,L,d_v]` 示意 | 训练/推理 |
| ① 状态 | Robot state `[B,D_s]` | 线性投影 | action-side state embedding | 训练/推理 |
| ② Prefix | 视觉 token + 语言 token | PaliGemma / Gemma 视觉语言 Transformer | 视觉语言条件上下文（可复用 KV） | 训练/推理 |
| ③ Suffix | state + 带噪动作 `x_t [B,H,D]` + `t` | action projection + time embedding，构造动作侧 token | Suffix token 序列 | 训练/推理 |
| ④ 双专家 | Prefix 与 Suffix token | 多层联合注意力，利用 block attention mask 约束访问 | 动作专家输出 `vθ [B,H,D]` | 训练/推理 |
| ⑤ 训练监督 | 示范真值 `A`、采样噪声 `ε`、时间 `t` | 构造 `x_t=(1-t)A+tε` 与目标 `ε-A`，计算 MSE | 标量损失并反向传播 | **仅训练** |
| ⑥ 推理采样 | 随机噪声 `x_1` + 当前观测 | 重复 K 次动作专家前向和反向时间积分 | 归一化动作块 `Â [B,H,D]`，再适配实际控制器 | **仅推理** |

**理解重点一：** SigLIP 编码的是图像，不是直接预测机器人动作；PaliGemma 提供视觉语言条件，而动作专家才输出用于 Flow Matching 的速度场。

**理解重点二：** 这里的“双专家”是两套不同的 Transformer 权重参与一个受控注意力计算体系，**不是**先让 VLM 独立生成一段文字，再交给一个完全独立的控制网络；也不是通过 Router 做 top-k 专家选择。

**理解重点三：** 教学图把双专家注意力画为单个概念模块，方便理解信息流。真实 openpi 的 prefix/suffix 是在多个 Transformer 层内协同计算，不能把图中的“联合注意力”理解成额外插入的一层网络。

**理解重点四：** 实际机器人执行前必须完成动作空间映射与反归一化；`[B,50,32]` 仅表示默认配置容器，未必是 32 个真实关节；最终控制频率也不能从 50 步 horizon 推导出来。

### 为什么必须区分 Prefix 与 Suffix？

- **Prefix（观测上下文）**：图像经视觉编码器与语言嵌入进入视觉语言 Transformer。它的任务是描述“现在看到了什么、要做什么”。
- **Suffix（动作建模）**：机器人自身状态、被噪声污染的未来动作块以及时间参数进入动作专家。它的任务是预测去噪方向。
- **联系**：动作侧通过注意力查询前缀语义与视觉上下文，但前缀不应反向依赖每一步重新采样的噪声动作。这样才能避免观测条件在去噪迭代中被随机动作不断污染，并为缓存 Prefix 计算创造条件。

### Token 具体从哪里来？先做张量账本

| 阶段 | 输入（示意） | 输出 | 检查点 |
| --- | --- | --- | --- |
| 多相机输入 | `[B,N,3,H_i,W_i]` | 视觉 patch/token 序列 | 不要把 N 路图像误作 N 个 token；预处理、patch 数量由视觉主干决定 |
| 语言输入 | `[B,L]` | 语言 token embedding | padding、tokenizer、指令模板 |
| 机器人状态 | `[B,D_s]` | Suffix 状态嵌入 | state schema、单位、关节顺序 |
| 未来动作块 | `[B,50,32]` | Suffix 动作 token | 50 是 horizon，32 是容器上限/配置，并非机器人真实自由度 |
| 时间参数 | `[B]` 或可广播形状 | 时间条件嵌入 | 当前使用的噪声比例及方向约定 |
| 模型输出 | 与动作块同形状 | `[B,50,32]` 速度场 | 每次输出整块速度而非一条末端轨迹点 |

**具体例子**：如果机器人每步输出双臂末端位置/姿态增量及两个夹爪指令，那么原始动作可能只有 14 维。映射到模型 32 维容器后，其余维度的填充和训练权重需要由适配器与 loss mask 管理，不能让填充数值伪装成真实控制量。

### Attention Mask：到底谁能看到谁？

![图 2：Prefix-LM 注意力可见性矩阵](/Embodied-AI-Roadmap/diagrams/pi0-attention.svg)

*图 2｜按 Query 行、Key 列理解遮蔽矩阵；图是简化示意，真实实现还必须叠加 padding、token 分组以及具体版本的 mask 规则。*

| Query \\ Key | Prefix（图像/语言） | Suffix（state/动作） |
| --- | --- | --- |
| Prefix | 允许 | 禁止 |
| Suffix | 允许 | 按实现的分组规则允许 |

**为什么这个设计重要？** 同一条指令和同一帧图像在多次 flow 积分过程中通常不变；随机动作却每次迭代都在改变。若 Prefix 可以读 Suffix，缓存的 Prefix KV 就会随动作更新而失效。

**代码核验**：阅读 [openpi/pi0.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)，先找 `embed_prefix`、`embed_suffix` 和 `make_attn_mask`，再沿着 `mask_ar` 或该版本实际采用的块规则追踪最终矩阵。教学图只解释核心访问关系，不能代替该实现检查。

### 训练和推理是两条不同的数据路径

![图 3：π₀ Flow Matching 训练与推理对比](/Embodied-AI-Roadmap/diagrams/pi0-training.svg)

*图 3｜训练用真值和噪声直接构造中间点/监督目标；推理没有真值，从纯噪声反复积分得到动作块。*

**训练**：真实动作 \(A\) + 高斯噪声 \(\epsilon\) + 时间 \(t\) → 构造 \(x_t\) → 前向预测 \(v_\theta\) → 监督 \(\epsilon-A\)。一次训练样本的损失计算不要求先运行完整 ODE 积分。

**推理**：固定图像/语言/state 条件，从随机 \(x_1\) 出发，反复调用动作专家预测速度并用数值积分更新整个动作块，直到得到 \(x_0\)。最终的动作仍需要机器人控制接口解释和执行。


> **阅读定位**：适合已经理解 Transformer、基本模仿学习和机器人状态/动作表示的读者。目标是看懂 π₀ 的核心计算链路，以及官方 openpi 代码如何实现；本篇不是已验证的复现报告。
>
> 原始论文：[π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)（2024-10-31） · [官方代码](https://github.com/Physical-Intelligence/openpi) · [官方博客](https://www.pi.website/blog/pi0) · [系列目录](./README.md)

## 1. 用三分钟理解它解决了什么

机器人操作同时需要语义和精细运动。“把毛巾叠起来”中的物体类别、目标和任务含义属于视觉语言理解；夹爪怎么接近布料、怎样维持接触并折起毛巾，则依赖连续动作预测。

以前一些 VLA 把动作离散成 token，再使用语言模型的自回归目标来预测。问题在于：如果机器人动作包含多个连续自由度，还要输出几十步动作，逐 token 解码会使动作精度、序列长度和推理延迟互相牵制。

π₀ 的选择是：**保留预训练视觉语言模型提供的语义表征，另加一个面向连续动作的 action expert，并使用条件 flow matching 一次生成整段动作**。论文覆盖单臂、双臂和移动操作平台，并测试零样本任务、语言控制与针对新任务微调；并非一项已证明对所有机器人通用的部署方案。

重点区分：
- **VLM**：理解观测与指令，产生上下文 token。
- **Action expert**：读取观测、机器人状态、带噪动作和时间参数，预测如何把动作噪声转成可执行动作。
- **Robot runtime**：把动作转换成实际设备指令；它不属于 flow matching 损失本身。

## 2. 输入输出、张量与数据约定

设观察为 $o_t=(I_t^1,\ldots,I_t^N,\ell_t,q_t)$，分别表示多相机图像、语言指令和机器人自身状态；目标为未来 $H$ 步的动作块：

$$
A_t=[a_t,a_{t+1},\dots,a_{t+H-1}]\in\mathbb{R}^{H\times D_a}.
$$

在 **openpi 官方默认 Pi0Config** 中，$D_a=32$、$H=50$。这是为多种机器人提供的**通用填充上限/配置默认值**，不是说每台机器人都有 32 个有效执行维度。实际训练与推理必须通过数据适配器映射状态/动作，并做归一化。不同机器人可能只用其中一部分维度。

一个帮助理解的示意 batch（**仅是形状说明，不等同于所有预训练数据**）：

| 变量 | 示例形状 | 含义 |
| --- | --- | --- |
| 相机图像 | `[B, N, 3, 224, 224]` | N 路 RGB；具体预处理及相机 key 依配置 |
| 语言 token | `[B, L]` | 指令，经分词器编码 |
| 机器人状态 | `[B, D_s]` | 机器人当前关节/末端/夹爪等状态 |
| 训练动作块 | `[B, 50, 32]` | 时间长度 × 动作维度 |
| 带噪动作 $x_\tau$ | `[B, 50, 32]` | flow matching 的网络输入 |
| 预测速度场 | `[B, 50, 32]` | 与动作块同形状 |

这里最容易搞错两点。第一，$H=50$ 是**预测的动作步数**，不等于 flow matching 的求解迭代步数。第二，动作块的数值语义由机器人数据定义（关节、末端相对位置等），不是模型架构自动保证的。

## 3. 为什么是“两组专家权重”，而不是两个互不相干的模型

π₀ 使用预训练 PaliGemma 视觉语言主干及独立初始化的较小 action expert。官方 openpi 默认配置中，视觉语言侧对应 `gemma_2b` 变体，动作侧对应 `gemma_300m` 变体；视觉编码还包括 PaliGemma 的视觉组件。

可以把核心结构理解为：

![π₀ 双专家内部关系：Prefix 和 Suffix 采用不同权重并进行受控联合注意力](/Embodied-AI-Roadmap/diagrams/pi0-experts.svg)

*图｜双专家计算路径（教学示意）。蓝色表示视觉语言 Prefix，绿色表示动作 Suffix，橙色表示速度场与生成动作；两侧在多层 Transformer 中协同计算，并非在模型末端才合并。*

这里的“专家”不是典型的 top-k gating MoE。两部分负责不同 token 的处理，并通过联合注意力交换信息；不是把 VLM 的结果简单拼到另一个独立网络最后一层。最值得读的代码是 [`pi0.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py) 与 [`gemma.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py)。

**读代码时的三个问题：**观测 token 与动作 token 分别在哪里构造？attention mask 如何限制不同 token 的访问？训练损失到底施加在哪一侧的输出？不要仅凭网络示意图判断梯度路径，要结合 forward 和 `compute_loss` 检查。

## 4. Flow matching 到底训练什么

这部分是 π₀ 与动作 token 自回归模型的关键差别。

设真实动作块为 $A$，高斯噪声为 $\epsilon\sim\mathcal N(0,I)$。为了和当前 openpi 实现保持一致，采用 **$\tau=0$ 是数据、$\tau=1$ 是噪声** 的记号（论文与不同代码可能使用相反时间约定）：

$$
x_\tau=(1-\tau)A+\tau\epsilon,\qquad
u_\tau=\epsilon-A.
$$

网络预测速度场 $v_\theta(x_\tau,\tau,o_t)$，优化：

$$
\mathcal L_{FM} =
\mathbb E_{A,\epsilon,\tau}
\left[\left\|v_\theta(x_\tau,\tau,o_t)-(\epsilon-A)\right\|_2^2\right].
$$

直觉上，你给网络“一段被混入噪声的未来动作”和“噪声比例”，网络要告诉你从真实动作走向噪声的局部方向。反过来沿这个方向积分，就能从随机噪声走回动作分布。

**维度检查：**$A$、$\epsilon$、$x_\tau$、$v_\theta$ 都是 `[B,H,D_a]`，时间 $\tau$ 通常按 batch 采样，再广播到动作维度。openpi 的训练实现使用偏向一端的 Beta 时间采样；这属于具体代码策略，不是 flow matching 的定义条件。

以下是可单独运行的 PyTorch **目标构造示例**，用于检验张量与符号；**不是 openpi 完整训练代码**：

```python
import torch

B, H, D = 2, 50, 32
actions = torch.randn(B, H, D)
noise = torch.randn_like(actions)
tau = torch.rand(B, 1, 1)

noisy_actions = (1 - tau) * actions + tau * noise
target_velocity = noise - actions

assert noisy_actions.shape == target_velocity.shape == (B, H, D)
# 真实训练还需要模型预测 velocity，然后对比 target_velocity 计算 MSE。
```

为什么不直接回归 $A$？从数学上并非绝对不行，而是条件 flow matching 可以利用连续生成建模表达多模态动作分布，训练时有明确的回归目标。它也有成本：推理需要多次网络求解，并不是“单次 forward 直接输出动作”。

## 5. 推理：十次积分与五十步动作，不是一回事

从噪声 $x_1\sim\mathcal N(0,I)$ 出发，用负时间步长沿模型预测向量场积分。在上述时间约定下，Euler 更新为：

$$
x_{\tau-\Delta}=x_\tau-\Delta\,v_\theta(x_\tau,\tau,o_t).
$$

例如设置 10 个积分步，每一步更新的是**整块 `[H,D_a]` 动作**，最后得到预测的 50 步动作块。模型不需要为 50 个控制时刻分别进行 50 次语言模型自回归调用。

控制侧又是另一层：实际只执行其中若干步、何时重新观测并规划、是否使用 action chunk 平滑或异步推理，都会影响闭环效果。**动作 horizon、采样求解步数、设备控制频率三个量不能混为一谈。**

推理耗时不只取决于专家大小，还受相机数量、图像编码、KV 缓存、采样步数和机器人通信链路影响。不能从论文的某个延迟数字直接推断在其他设备上的真实控制频率。

## 6. 训练链：哪些参数更新、哪些被遮蔽

阅读 [`Pi0.compute_loss`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py) 时，应追踪以下路径：

1. 原始机器人数据经过适配器，产出图像、语言、状态和动作块。
2. 图像与文本转换成 VLM token；状态与带噪动作转换成 action expert 需要的 token。
3. 合并注意力计算，让动作 token 从观测 token 获取条件信息。
4. 速度预测头输出与目标动作块匹配的速度张量。
5. 对动作维度计算 flow matching 回归损失，并按配置处理无效维度和填充位置。

要区分**论文的架构设计**和**某个开源训练配置的参数冻结方式**。不能笼统写成“π₀ 总是冻结 VLM”或者“π₀ 只训练动作头”；训练阶段和配置可能不同。实际应在训练脚本确认 trainable parameters、optimizer、LoRA/全量微调设置。

## 7. 实验到底证明了什么

原始论文展示了在多种机器人上执行复杂任务，包含零样本行为、语言指令跟随以及经过新任务数据微调后的能力。这支持“预训练视觉语言知识 + 多机器人动作数据 + 连续动作生成”这一设计在作者实验设置下可行。

但需要谨慎解读：

- **多机器人有效**不代表动作空间天然一致；数据适配与归一化不可省略。
- **能折毛巾**不代表具备通用接触力学模型；不同任务的失败模式很不一样。
- **开环轨迹误差低**不必然意味着闭环成功率高；部署要测真实成功率、耗时与恢复能力。
- 原始 π₀ 与后续 π₀.₅、π₀-FAST 的实验不能混合成一个统一成绩单。

建议阅读原论文时重点对照基线、数据量、是否新增任务示范、评测次数与成功判据。

## 8. 若要迁移到工业轮式双臂，哪些地方需要改

这是**工程推演**，不是论文已验证的“帕西尼手套 + 轮式双臂”结果。

**第一步：先锁定 action definition。** 例如双臂相对末端位姿与夹爪命令，明确 base frame / camera frame、轴旋转表达、动作间隔、归一化方式。轮式底盘导航建议先保持确定性控制，不与高频接触动作一起盲目端到端训练。

**第二步：保持观测一致。** Head/Wrist 的相机语义、分辨率、内外参、同步、缺帧处理在训练与部署端一致；给缺失模态明确 mask 或适配策略。

**第三步：不要把手套信号直接伪装成 robot action。** 手套关节角和压力首先可以用于接触阶段、抓握类别与失败事件监督；是否能进一步提供 robot-compatible action，需要单独的 retarget 与验证。

**第四步：做消融。** 固定相同 robot demonstrations，比较 (A) 原始训练，(B) 加人类 EGO 数据，(C) 加手套派生接触阶段标签。除任务成功率外，还统计碰撞、插接到位、异常恢复与控制时延。没有这些实验，不能宣称手套一定提升了 π₀。

## 9. 读官方代码的顺序

| 顺序 | 文件/目录 | 看什么 |
| --- | --- | --- |
| 1 | [`pi0_config.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py) | 专家配置、horizon、action dim、π₀/π₀.₅ 差异开关 |
| 2 | [`pi0.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py) | prefix/suffix、loss、动作采样 |
| 3 | [`gemma.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py) | Transformer 和联合注意力实现 |
| 4 | [`training/config.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) | 数据、参数、训练配置 |
| 5 | [openpi README](https://github.com/Physical-Intelligence/openpi) | 安装、checkpoint、推理与微调入口 |

官方仓库提供 π₀ 相关代码与权重，但**仓库随时间会变化**，教程引用的是仓库路径而不是固定 commit。本文尚未执行其训练流程，因此不标为“成功复现”。

## 10. 读完应该能回答的四个问题

1. π₀ 的 action expert 到底预测动作还是预测**动作上的速度场**？
2. 为什么 `action_horizon=50` 不代表推理需要运行 50 次网络？
3. 人类 EGO 只有 RGB，没有机器人 action 时，为什么不能直接使用 π₀ 的监督损失进行标准 BC？
4. 如果同一个模型接入 6DoF 机械臂与双臂夹爪，最先需要重新定义的是架构，还是数据与 action mapping？

## 参考资料

- Black et al., [π₀ 原论文](https://arxiv.org/abs/2410.24164)
- Physical Intelligence, [官方 π₀ 介绍](https://www.pi.website/blog/pi0)
- Physical Intelligence, [openpi](https://github.com/Physical-Intelligence/openpi)
- 后续文章将专门比较 [π₀-FAST](https://www.pi.website/blog/fast) 与 [π₀.₅](https://www.pi.website/blog/pi05) 的设计，不在本文混写版本结论。

## 工程补充：从代码实现 Flow Matching 的最小闭环

下面代码**仅复现数学监督与采样循环**，不假装实现了 PaliGemma、Gemma 双专家及其联合 attention。它能够验证时间方向、张量维度与积分过程：

```python
import torch
import torch.nn.functional as F

B, H, D = 4, 50, 32
real_actions = torch.randn(B, H, D)
noise = torch.randn_like(real_actions)
t = torch.rand(B, 1, 1)

# t=0 -> 数据；t=1 -> 纯噪声
x_t = (1 - t) * real_actions + t * noise
target_velocity = noise - real_actions

# 真实训练时由 action expert 接受 x_t、t 和观测条件，输出 predicted_velocity
predicted_velocity = torch.zeros_like(x_t, requires_grad=True)
loss = F.mse_loss(predicted_velocity, target_velocity)
loss.backward()
print("train", x_t.shape, loss.item())

# 简化的 Euler 积分：这里用已知的理想常速度只核对数值方向
steps = 10
x = noise.clone()
for k in range(steps, 0, -1):
    x = x - target_velocity / steps
print("max reconstruction error", (x - real_actions).abs().max().item())
```

注意第二段是**数值单元测试**，用了已知监督速度，因此不是模型生成表现。实战必须把 `target_velocity` 替换为网络在当前 `x,t,o` 上预测的速度；在每个积分步重新计算，不能把一次预测结果重复当常数速度场。

### 进一步的源码阅读路线

1. `pi0_config.py`：确认 expert 变体、`action_dim`、`action_horizon`、模型版本。
2. `embed_prefix`：记录图像每路 token 数、语言长度及 padding。
3. `embed_suffix`：打印状态投影后形状，记录动作块和时间嵌入的组合方式。
4. `make_attn_mask`：用 2 个 Prefix token 和 3 个动作 token 造最小测试，打印 Query×Key 布尔矩阵。
5. `compute_loss`：追踪噪声分布、时间采样、目标速度和无效动作维度屏蔽。
6. `sample_actions`：记录采样步数、Prefix KV 复用情况和完整动作块时延。
7. 部署适配器：核对动作单位、控制帧率、机器人限位及超时回退。

### 迁移到实际机器人必须回答的八个问题

| 核查维度 | 必须拿到的答案 | 典型失败 |
| --- | --- | --- |
| 动作语义 | 关节目标、关节增量、末端 delta pose 还是末端绝对 pose | 预测看似合理，控制器却做了错误动作 |
| 坐标系 | base/world/tool/camera 哪个为准 | 双臂轨迹方向相反 |
| 单位 | 米/毫米、弧度/角度、控制频率 | 运动幅度异常 |
| 数据同步 | 图像、state、action 的时间戳及延迟 | 动作对不上视觉状态 |
| Action chunk | 预测 50 步但每次执行多少步、何时重规划 | 接触任务开环失败 |
| 有效维度 | 32D 容器中哪些维度合法 | loss 被 padding 干扰 |
| 安全 | 碰撞监控、速度限制、急停和失败恢复 | 模型输出直接造成风险 |
| 闭环评测 | 成功率、时延、接触错误、失败恢复 | 仅凭离线 loss 判断可部署 |

