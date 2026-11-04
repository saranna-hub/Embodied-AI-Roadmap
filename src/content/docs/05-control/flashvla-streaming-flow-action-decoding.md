---
title: FlashVLA 深度教程：从 π₀.₅ 串行去噪到流式动作解码
description: CoRL 2026 官方开源代码逐函数追踪，解释 Buffer、分块因果 Mask、训练打包、Flow Matching、异步执行、复现基准与真机风险
sidebar:
  order: 3
---

> **证据与版本（2026-11-04 核查）**：Zekai Li、Jiaming Tang、Zhijian Liu，《FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference》，[arXiv 2608.27384（2026-08-27 初次发布）](https://arxiv.org/abs/2608.27384)，[Z Lab 官方项目（CoRL 2026）](https://z-lab.ai/projects/flashvla/)，[作者源码 z-lab/flashvla](https://github.com/z-lab/flashvla)，[官方模型索引](https://huggingface.co/collections/z-lab/flashvla)。**以下“官方源码”专指这些公开文件；“教学等价式”和“CPU 契约脚本”是本站为理解原理建立的最小实验，绝不表示已经复现完整 GPU 模型或真机实验。**

# FlashVLA：为什么把去噪变成流水线，就能同时改善延迟与动作连续性？

## 0. 学习路线：先懂什么，再验证什么？

先读 [π₀.₅ 源码动作契约](/Embodied-AI-Roadmap/01-foundations/vla/pi05-libero-data-loss-lora-source-guide/) 和 [Flow Matching 数学基础](/Embodied-AI-Roadmap/00-prerequisites/flow-matching/)，再读 [PACT-VLA](/Embodied-AI-Roadmap/05-control/pact-vla-precommit-flowres/)。本文不是“训练一个全新 VLA 从零识别物体”，而是**重构 Flow Matching 动作专家的训练与解码时间安排**。

![FlashVLA 总览：动作缓冲槽、数据与噪声时间、解码路径](/Embodied-AI-Roadmap/diagrams/flashvla-streaming-system.svg)

*图 1：本站基于官方论文与源码重绘。每次 forward 为所有 slot 同时做一步去噪，最干净的一段出队，尾端补入新噪声；此图是概念时序，不是原论文图片。*

**本章要回答的面试问题**：传统 π₀.₅ 做 10 次积分才能吐出一段动作，FlashVLA 为什么“一次 forward 吐一段”而不是偷工减料成一步去噪？跨 chunk 注意力为什么是单向的？为什么冷启动必须特殊处理？异步预测过时的问题真的被完全解决了吗？怎样用一致的评测协议证明加速而非只测 GPU 函数？

## 1. 痛点：一段动作经历独立的 N 次去噪

普通基于 Flow Matching 的动作生成器输入观测 $o_t$、语言 $\ell$、状态 $s_t$，输出长度 $H$ 的机器人动作：

$$
A_t=(a_t,\ldots,a_{t+H-1})\in\mathbb R^{H\times D_a}.
$$

常规做法从独立噪声 $\epsilon\sim\mathcal N(0,I)$ 出发，多次调用同一动作专家 $v_\theta$，最后得到整个 chunk。假设每次前向耗时 $T_f$，有 $N$ 个积分步，忽略额外开销后，**单段动作解码延迟**近似：

$$
T_{\rm decode}^{\rm blocking}\approx N T_f.
$$

但是机器人物理控制器必须每个 $\Delta t_c$ tick 发出动作。同步模式可能出现“执行 K 步 → 等待解码 → 执行下一段”的停顿；异步模式提前在旧图像上生成动作，若物体已移动便出现 **state-observation mismatch**。两种问题的共同成因是**每个动作块都被当成互不相关的一次独立生成任务**。[项目 Motivation](https://z-lab.ai/projects/flashvla/)。

特别注意：$H$（预测的物理动作长度）、$K$（每次执行的低层步数）、$N$（积分步数/流式槽数）、$d$（提前发起推理的重叠步数）是不同概念。$H=40,N=4,K=5,d=1$ 完全可同时成立。

## 2. 核心直觉：把“一个 chunk 的四次去噪”摊开到四个时刻

传统 $N=4$：

\`\`\`text
观测 o₀ ──[decode chunk A: t=1→.75→.50→.25→0]──> A执行
观测 o₁ ──[decode chunk B: t=1→.75→.50→.25→0]──> B执行
                         每段都等待4个完整前向
\`\`\`

FlashVLA $N=4$ 的**稳态**（从近端到远端，槽内每次只有一步去噪）：

\`\`\`text
第 k 次 forward 输入： [slot0@t=.25 | slot1@t=.50 | slot2@t=.75 | slot3@t=1.00]
   全部并行再走一步     [slot0@t=0    | slot1@t=.25 | slot2@t=.50 | slot3@t=.75]
   发出 slot0           → [可执行动作 C 个] 
   向左移位+尾部新噪声    [旧slot1      | 旧slot2      | 旧slot3      | 新slot@t=1]
\`\`\`

每段新动作**仍经过 $N$ 次积分更新**，但是同时在途有 $N$ 段，流水线填满之后**一次 forward 完成一个输出**。这叫 amortization（摊销），并不是把完整速度场直接缩成一轮近似。它与把 Action Horizon 由 50 改成 5、或者把 ODE 积分步数强行从 10 改为 1，都不是同一件事。

需要明确代价：每次 forward 同时处理更多的 suffix token，而且启用了跨段注意力。不能因此断言计算量天然降低为原来的 $1/N$；加速幅度必须由 GPU profile、批量、图像视角数与实现优化共同决定。

## 3. 数学：Flow 时间方向及逐槽 Euler 更新

### 3.1 先统一时间参数

作者代码采用**数据端为 $t=0$、噪声端为 $t=1$**：

$$
x_t=t\epsilon+(1-t)A,\qquad
u_t=\epsilon-A.
$$

训练时用预测速度 $\hat u_\theta=v_\theta(x_t,t,o,\ell,s)$ 拟合真值：

$$
\mathcal L_{\rm FM}=
\mathbb E_{A,\epsilon,t}\left[\|\hat u_\theta-u_t\|_2^2\right].
$$

推理从 $t=1$ **向 $0$ 走**，所以 Euler 更新必须是：

$$
x_{t-\Delta t}=x_t-\Delta t\,v_\theta(x_t,t,o,\ell,s),\quad
\Delta t=\frac1N.
$$

为什么不能写成加号？在完美速度场 $v_\theta=\epsilon-A$ 时，$x_1=\epsilon$，减去 $(\epsilon-A)$ 才能到 $A$；加号会朝反方向发散。源码 [\`forward_shared_observation\`](https://github.com/z-lab/flashvla/blob/main/flashvla/policies/pi05/modeling_pi05_flashvla.py) 使用 \`x_t = time*noise + (1-time)*actions\`、\`u_t = noise-actions\`；[\`_steady_streaming\`](https://github.com/z-lab/flashvla/blob/main/flashvla/policies/pi05/modeling_pi05_flashvla.py) 用 \`buffer = buffer - dt * v_t\`。

### 3.2 把整个缓冲区写成一个张量

设每次发出 $C$ 个低层动作，$N$ 个槽，动作维 $D_a$，那么：

$$
B\in \mathbb R^{b\times (NC)\times D_a}
\quad\text{或}\quad
B\in \mathbb R^{b\times N\times C\times D_a}.
$$

$\mathrm{slot}_j$ 的每个动作 token 都共享去噪时刻 $t_j=(j+1)/N$，因此每次的时间输入不是单个标量，而是：

$$
T\in\mathbb R^{b\times NC},\qquad
T_{:,jC:(j+1)C}=\frac{j+1}{N}.
$$

对于视觉语言前缀 $P$，注意力中的每个后缀动作 token 有自己的 time embedding，代码中 \`PI05SuffixEmbedder.forward\` 的 \`time\` 接收 \`[B,L]\`，通过 per-token AdaRMS 条件注入层。**每个槽内部的 C 个动作共用一个 t，不代表不同槽的 t 相同**。

### 3.3 一个完整的 steady-step

- 为当前图像/任务创建 Prefix embedding；对每层的 Prefix KV 进行 prefill。
- 使用 buffer 与逐槽 time 做一次动作专家 forward，得到 $V\in\mathbb R^{b\times NC\times D_a}$。
- 一次 Euler 更新：$B\leftarrow B-V/N$。
- 取头部 $C$ 个动作发给异步动作队列；剩余槽向前移动，并从尾部补 $C$ 个高斯噪声 token。

**源码事实**：在当前公开 π₀.₅ 实现中，Prefix 会在每次 steady forward 依据当前图像重新编码/构造本次 KV；不是“整个 episode 冻结一份 Prefix”，否则模型无法响应画面变化。

## 4. 真正关键的不是 Buffer，而是 Chunk-wise Causal Attention

如果后面的高噪声 future chunk 能反过来影响更早的低噪声、即将执行的 chunk，早段动作就会被未来尚不可靠的猜测干扰；如果完全不跨 chunk 交互，就重回孤立生成。FlashVLA 选择：

$$
M_{ij}=
\begin{cases}
1,&\text{query chunk }i\text{ 可读 key chunk }j,\ j\le i,\\
0,&j>i.
\end{cases}
$$

并允许槽内的 $C$ 个位置**双向互看**，使一个 chunk 的动作内部保持整体一致。多个位置可以同时观察到**同一份视觉语言前缀**。

举例 $N=4$（这里只画 slot 级，非真实 token 数）：

![FlashVLA 块级注意力、训练配置隔离与异步时序](/Embodied-AI-Roadmap/diagrams/flashvla-mask-timeline.svg)

*图 2：深色实线表示允许读取；禁止 future→past 影响是为了避免高噪声尾槽污染即将执行的近端。训练中不同 cold-start 配置也是互相不可见的。*

\`\`\`text
query\key    Prefix   slot0   slot1   slot2   slot3
Prefix           ✓       ×       ×       ×       ×
slot0            ✓       ✓       ×       ×       ×
slot1            ✓       ✓       ✓       ×       ×
slot2            ✓       ✓       ✓       ✓       ×
slot3            ✓       ✓       ✓       ✓       ✓
\`\`\`

为什么“Prefix 不能反向读后缀”？如果 Prefix 在某层提前读取未来动作 token，再被另一个后缀 token 读取，就会形成**多层间接泄漏**。这种掩码与 π₀ 的 Prefix/Suffix 注意力机制一脉相承，但本文进一步将 suffix 分为有方向的 action chunks。

源码的实际构造不是硬编码上述 5×5：\`PI05SuffixEmbedder\` 将每槽第一个 token 的 \`att_row[:,0]\` 设为 1，利用分组累计和建立同槽双向、跨槽单向的 block-causal mask（[源码](https://github.com/z-lab/flashvla/blob/main/flashvla/policies/pi05/modeling_pi05_flashvla.py)）。工业开发时需要同时考虑真实 \`pad_mask\` 与语义上的 block-causal mask，不能只测可视化。

## 5. 训练：为什么同一张观测要打包 N 种 Cold-start 配置？

训练数据原本包含目标轨迹 $A_{0:NC}$ 和当前观测；真实运行有两种状态：Episode 刚开始 Buffer 里只有一段真实 noise，稳态时所有槽都已经填满。只训练稳态、推理时直接上冷启动，会形成训练与推理的严重不一致。

因此 \`FlashVLADataset.__getitem__\` 构造 $N$ 种训练配置 $k=1,\ldots,N$，其中**前 $k$ 个 slot 为真实动作目标**、余下 $N-k$ 个 slot 为 padding。每种配置长度 $H=NC$，将 $N$ 种配置摊平：

$$
A_{\rm packed}\in\mathbb R^{b\times (N^2C)\times D_a},
\qquad
M_{\rm pad}\in\{0,1\}^{b\times N^2C}.
$$

**不要误读**：这个 $N^2C$ 只表示一次训练样本打包后的动作 token 总数；推理缓冲仍只有 $NC$ 个 token，论文没有让机器人一次执行 $N^2C$ 步。

以官方 LIBERO 训练配置为例，$N=4,C=10$：

| 配置 | 真实动作 token | Padding token | 配置总长 |
| --- | ---: | ---: | ---: |
| k=1 | 10 | 30 | 40 |
| k=2 | 20 | 20 | 40 |
| k=3 | 30 | 10 | 40 |
| k=4 | 40 | 0 | 40 |
| 拼接 | **100** | **60** | **160** |

批次 $b=8$ 时输入训练动作形状为 \`[8,160,D_a]\`，前缀输入图像/文本仍只需要一份，起到**共享视觉计算**的作用。

### 5.1 打包还会产生一类更隐蔽的泄漏

对于相同视觉 Prefix 的不同 cold-start 配置，**配置 k 的 suffix 不能读到配置 k' 的真实动作**。否则训练 loss 会通过其它配置的真值“作弊”。\`_build_shared_obs_mask\` 在 suffix×suffix 区域构造 \`cross_offset_mask\`，只允许同一配置内的 action tokens 互相注意；它们共享视觉 Prefix，但**不能跨配置通信**。推理阶段不打包 $N$ 套配置。

### 5.2 时间采样比固定时间点更细

源码 \`_build_per_chunk_time\` 用 Beta 分布独立采样每个 (配置，槽) 的时间偏移，然后加上槽对应的时间起点，并对 padding 设置 $t=1$。所以训练不是每次在 $t=.25,.5,.75,1$ 上死板拟合，而是在各时间子区间内连续采样，增强噪声时间泛化能力。

代码中两类屏蔽都必须检查：动作 padding mask 与 cross-configuration attention mask。\`forward_shared_observation\` 对 padding 剔除后计算动作速度 MSE，最终 \`losses.mean()\`。目标维度由 \`max_action_dim\` 决定，不应臆测 7D LIBERO 的无效维度必然由某个 action-dim mask 自动清除；如要更严格审计，需要和预处理器、模型配置逐层核对。

## 6. 冷启动：为什么前几个输出应该“保持静止”？

源码 \`sample_actions\` 在 \`step_counter < N-1\` 的前三次调用仅填充缓冲，不输出可执行策略 chunk。第四次才发出第一段动作。于是推理时会遇到 **\`actions is None\`**。

\`PI05FlashVLAPolicy.select_action\` 选择安全的 Cold-start 替代动作：对 Delta-action 的环境使用 \`zero_delta\`；对绝对关节控制可采用 \`current_state\`。**绝对关节角动作空间不能用全零向量假装保持当前位置**；下发 0 可能导致手臂回零或撞击！

官方异步管理器 \`AsyncStreamingActionManager\` 在冷启动也会构造可执行的保持动作，并在每个 episode 边界重置 Buffer。真实工厂中应把末端位姿、夹爪、急停联锁和控制器约束分别审查，不可将研究仿真的 hold action 直接作为安全证明。

## 7. 异步管理器：提前 d 步触发 ≠ 输出自动对齐

官方管理器定义 \`overlap_steps=d\`：在当前执行段结束前 $d$ 个低层动作时发起下段推理，$d=0$ 代表同步。另一选项 \`skip_stale_actions\` 采用 RTC 风格的时间重新对齐：提升下一段时跳过它的前 $d$ 步，仅消费 \`[d,d+K)\`，必须满足：

$$
0\le d\le K,\qquad d+K\le C.
$$

举例 $C=10,K=5,d=2$：下一个预测 chunk 为 \`[a0,...,a9]\`，旧 chunk 在算力重叠期间已经执行 2 步，若选择启用跳过旧动作，就从新 chunk 的 \`a2\` 开始执行 \`a2..a6\`。这里的约束是**manager 的可选重新对齐策略**，不是 FlashVLA 模型中的 block causal mask，也不是所有异步场景都自动跳步。

另一个很容易造成“机器人莫名卡住”的工程陷阱：初次进入 \`torch.compile / CUDA Graph\` 可能需要大量编译时间。作者管理器提供 \`warmup(processed_obs)\`，在正式 rollout 前编译和捕获，然后清空状态再启动。未预热的首个 chunk 延迟不能与稳态推理延迟混为一谈。

## 8. 真实复现：从官方代码而不是想象的接口开始

原始仓库：<https://github.com/z-lab/flashvla>。核心路径逐个对应：

| 需求 | 官方源码入口 | 应看到什么 |
| --- | --- | --- |
| 数据打包 | [\`flashvla/datasets/flashvla_dataset.py\`](https://github.com/z-lab/flashvla/blob/main/flashvla/datasets/flashvla_dataset.py) | $N$ 种 padding 配置、\`action_is_pad\` |
| Prefix/Suffix | [\`modeling_pi05_flashvla.py\`](https://github.com/z-lab/flashvla/blob/main/flashvla/policies/pi05/modeling_pi05_flashvla.py) | token 时间、shared obs、Mask |
| 训练参数 | [\`train/configs/pi05/libero/pi05_flashvla.yaml\`](https://github.com/z-lab/flashvla/blob/main/train/configs/pi05/libero/pi05_flashvla.yaml) | \`num_buffer_slots:4\`、\`chunk_size:10\` |
| 流式前向 | 同上 \`_cold_start/_steady_streaming\` | 更新、发出、移位、补噪声 |
| 异步与预热 | [\`flashvla/async_manager.py\`](https://github.com/z-lab/flashvla/blob/main/flashvla/async_manager.py) | \`overlap_steps\`、reset、warmup |
| LIBERO 评测 | [\`sim_eval/libero/README.md\`](https://github.com/z-lab/flashvla/blob/main/sim_eval/libero/README.md) | 单 suite、种子 sweep、输出 JSON |
| 延迟 Profile | [\`benchmarks/benchmark_latency.py\`](https://github.com/z-lab/flashvla/blob/main/benchmarks/benchmark_latency.py) | 每 policy invocation，非 episode 总耗时 |

**硬件与依赖边界**：官方要求 conda/CUDA 与相应的模拟器环境。本站 CPU 小实验仅用于测试数据形状、Mask、数学符号和时序；没有 GPU，也能先发现 80% 的 tensor/mask 设计级错误，但不可能据此验证官方 latency 和成功率。

### 8.1 环境与权重（官方 README 命令）

\`\`\`bash
git clone https://github.com/z-lab/flashvla.git
cd flashvla
conda env create -f environment.yml
conda activate flashvla

# 官方开源 LIBERO pi0.5-FlashVLA 权重：
# https://huggingface.co/z-lab/flashvla-pi05-libero
# 评测前先阅读 sim_eval/libero/README.md 配置 LIBERO 环境
bash sim_eval/libero/eval.sh

# 单个 suite / 50 episodes 的最小验证入口
python sim_eval/libero/eval.py \
  --policy.path=z-lab/flashvla-pi05-libero \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=50 --policy.n_action_steps=5 \
  --inference_overlap_steps=1 --policy.compile_model=true

# 完整重训练（GPU 资源需求与实际可行性要先检查）
bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml

# 单次推理延迟测试
python benchmarks/benchmark_latency.py \
  --config_path=benchmarks/configs/latency_flashvla.yaml
\`\`\`

**注意**：上面的命令来自官方仓库，不表示本站已安装或运行。单 suite 50 episodes 只是 smoke test，不能宣称复现 4 suite × 5 seed 的论文汇总成绩。

### 8.2 真实实验记录必须固定哪些量？

使用至少三组：原始 π₀.₅ 同步、FlashVLA 同步 $d=0$、FlashVLA 异步 $d=1$；如果计算资源允许，再对照 RTC、PACT-VLA 或另一个 streaming 实现。每组固定**模型权重、任务集合、episode seeds、成功率计分、动作后处理、相机视角数量、控制步频与 K**。还应记录 camera-to-inference 和 inference-to-controller 的实际延迟，而不仅是 GPU kernel 耗时。

作者 LIBERO 报告（均来自官方项目，**不是本站测量**）：

| 方法 | 平均成功率 | Time/Step | 主要含义 |
| --- | ---: | ---: | --- |
| π₀.₅ 同步 | 96.9% | 53.8 ms | 基线 |
| FlashVLA 同步，d=0 | 97.9% | 未列 | 去除异步干扰看模型效果 |
| FlashVLA 异步，d=1 | 97.8% | 22.1 ms | 2.43× 端到端步级加速 |
| FlashVLA 异步，d=2 | 98.3% | 20.6 ms | 2.62×（对应论文设置） |

上述数据的评测配置为 4 个 LIBERO suite，作者 README 标注五次运行均值；不要将“每次动作解码阶段最多 20×”误称为“整机控制 20×”。实际测的 \`Time/Step\` 是 episode wall time / 执行低层控制步数，包含仿真渲染与通信。

### 8.3 推理频率、控制频率、样本产率

定义 $T_{\rm policy}$ 为一次 forward 调用时间，$T_{\rm ctrl}=1/f_{\rm ctrl}$ 为机器人驱动周期，$T_{\rm wall}$ 为 episode wall time：

$$
f_{\rm policy}\approx\frac1{T_{\rm policy}},\quad
f_{\rm ctrl}=\frac1{T_{\rm ctrl}},\quad
{\rm Time/Step}=\frac{T_{\rm wall}}{N_{\rm executed}}.
$$

作者在真实 Franka + RTX A4000 测得约 **67.3 ms 的策略推理**，在 30 Hz 的控制周期下采用约两个 tick 的动作重叠来实现连续控制。**不要误写成模型在该 A4000 上每 33.3ms 完成一次 forward**：控制器可以在模型预测时继续消费动作队列，两种频率本来就不同。

作者报告的 LIBERO 数据与实际工业真机数据分布不同；尤其频繁接触、双臂碰撞、突发人员干预时，无法依靠 block causal mask 本身保证安全。

## 9. 无 GPU 的数学与调度契约：先让错误变得可见

本站随文提供 [CPU 小实验](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/flashvla_streaming_contract_cpu_demo.py)，标准 Python 运行：

\`\`\`bash
python3 examples/flashvla_streaming_contract_cpu_demo.py
\`\`\`

该实验逐项断言：$N=4,C=10$ 的 packed 训练数据确实有 160 tokens / 100 有效 / 60 padding；流式缓冲用完美速度场 Euler 积分每个 chunk 恰好四次，冷启动时不发出策略动作；块级注意力阻止 future→past 和跨配置读取；$K+d\le C$；对同一个候选策略，比对同步等待时间与异步时间的模型化预算。**这不是运行真实模型，也不是论文复杂系统加速的实验数据**。

如果同类数学契约失败，就应先修动作编码、mask、时间方向、episode reset，再上 GPU。这种“先证伪，再算力”的思路对于数据采集规模化很重要。

## 10. 与 PACT-VLA、RTC、传统 π₀.₅ 比较

| 方法 | 主要计算位置 | 关注的错误 | 是否需要额外训练 | 对新画面突变反应 |
| --- | --- | --- | --- | --- |
| π₀.₅ Blocking | 每段完整 ODE | 模型本体动作学习 | 原始训练 | 等完整 decode |
| RTC | 已承诺动作与新段拼接/对齐 | chunk 接缝和状态不对齐 | 取决于实现 | 对中途突变仍受约束 |
| PACT-VLA | 提交前最新观测评分 | 已生成动作的场景失配 | 论文版本可冻结基座 | 新观测可重选候选 |
| **FlashVLA** | 一次动作 Buffer forward | 过长串行解码、跨段连续性 | **训练为 streaming 结构** | 依赖新观测，但没有显式提交前候选验分 |

**不要混淆这两篇的“提前”**：PACT-VLA 用“旧观测生成多个候选，在提交前新观测评分”；FlashVLA 用“动作段同时在不同噪声层级上更新，让后续段看到前面段”。一个在**候选选择**层，一个在**流式动作生成**层。二者不保证即插即用组合，尤其 Prefix、推理时序、批量与算力预算可能互相冲突。

## 11. 工业双臂/UMI 平台实践建议

一个能落地的最小闭环可以是：

\`\`\`text
UMI/EGO 规范数据 → 转换 robot obs/state/action
        ↓                     (先审查 Delta/Absolute、时间戳、Z-up)
π₀.₅ 原始 BC 基线训练 → 固定成功率/时延/碰撞的 Benchmark
        ↓
FlashVLA streaming 结构微调 → 版本锁定的 Checkpoint
        ↓
离线验证 (每槽数量、掩码、绝对/增量控制) → LIBERO/RoboTwin 异步评测
        ↓
固定工作站低速真机 → 重复执行/失败统计/急停与干预记录
        ↓
回采接管与失败事件 → 数据 QA → 下一轮训练
\`\`\`

对于双臂夹爪示教，**K 与 N 的工程意义**不同：$K$ 影响反应频率与 chunk 接缝，$N$ 影响训练噪声层级和单次缓冲尺寸；动作空间是末端 SE(3)+夹爪还是关节控制，则决定 cold-start hold 如何下发与约束。UMI 录制的末端轨迹必须经过可执行性过滤；把不满足关节限位的示范当成可执行 Action 会让模型学到危险动作。

优先选择“固定托盘、两种零件、双臂交替抓放”与“开盖后装入”两种代表性场景。以每个任务 20～50 个**预注册**初始状态跑 3 个模型条件，报告成功率、动作平均跃变、运行时超时、模型每次推理 P50/P95、控制指令间隔 P95、抓取失败率、接触停机率。多任务评估按物体大小、光照、位置偏移分别汇总，不把同一轨迹切帧重复算作独立成功样本。

## 12. 失败现象及定位路径

| 现象 | 优先核查 | 容易忽视的原因 |
| --- | --- | --- |
| 第一个动作块之前“停了几帧” | cold-start/hold 是否符合 Action 语义 | 正常的 N−1 次填槽，或首次 CUDA Graph 编译 |
| 机器人在每个 chunk 边界跳动 | K、d、skip_stale、归一化、state timestamp | chunk 切换消费了过期 index |
| 训练 loss 异常低、真机无效 | N 种配置之间是否 attention 泄漏 | shared Prefix + suffix 跨 offset 漏洞 |
| 执行进入限位/碰撞 | Delta/Absolute 与关节工作空间 | 模型无内生硬安全保障 |
| 少相机有效，多相机极慢 | Prefix encode 占比、图像数量、GPU 内存 | streaming 只摊薄去噪并不消除视觉成本 |
| 标称30Hz，但 FPS 忽高忽低 | GPU forward、网络传输、控制周期分别打点 | 控制/策略/显示频率混为一谈 |

## 13. 对论文结论的边界判断

作者在 LIBERO、RoboTwin 2.0、Franka 实验中获得效果，但这些成绩**不能推广为任何 GPU、所有双臂与所有工厂零件上的保证**；真实控制与安全需要现场测量。论文结果还依赖专用 streaming 联合训练，而不是“拿原版 π₀.₅ checkpoint 改一行 for 循环就获得作者的结果”。官方提供的公开模型检查点能让复现从评测开始；如要迁移到不同 EGO/UMI 传感器和双臂执行器，还需重新制作数据、校验动作空间并调整微调策略。

建议在团队内部将研究结论写成可证伪假设：**“相同训练数据与硬件条件下，streaming 相比 π₀.₅ blocking 在预先约定成功率容差内降低 Time/Step 与动作块边界位置/速度跳变。”** 如果实验不支持它，就要查数据契约、代码时序或任务分布，而不是单凭论文标题判定“已经实时”。

## 参考与复核路径

原论文 [arXiv 2608.27384](https://arxiv.org/abs/2608.27384) ｜ [作者项目与图示](https://z-lab.ai/projects/flashvla/) ｜ [作者源码](https://github.com/z-lab/flashvla) ｜ [LIBERO 权重](https://huggingface.co/z-lab/flashvla-pi05-libero) ｜ [RoboTwin 权重](https://huggingface.co/z-lab/flashvla-pi05-robotwin) ｜ [CPU 数学契约](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/flashvla_streaming_contract_cpu_demo.py)。

**阅读重点：** 把第 2 节的 slot 调度、第 4/5 节的 attention/packed dataset 和第 7/8 节的实际执行时序连成一个脑内可跑的训练→推理完整闭环。
