---
title: "MINERVA 深度教程：54 万参数解决 LIBERO，为什么不等于 VLA 泛化？"
description: "CNN、FiLM、Spatial Softmax、MLP-Mixer、FM与L1、动作块长度、任务 ID 置换、LIBERO-Plus、评测协议及可执行 CPU 契约实验"
sidebar:
  order: 2
---

> **证据边界（2026-10-31）**：[MINERVA 原论文（2026-09-03）](https://arxiv.org/html/2609.03715)；[LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)；[LeRobot 官方 LIBERO 文档](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx)；[openpi 的 LIBERO 运行示例](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero)。截至本次核查，没有从论文找到可验证的完整 MINERVA 官方训练源码与可下载 checkpoint 地址。本篇细分**论文原始结果**、**教学推导**和**研究性替代复现**；不声称已复现论文成功率、完成 GPU 训练或真机执行。

# 为什么一个不到 1M 参数的小模型能挑战大型 VLA？

论文的主题不是「发现了通用 VLA 的最优架构」，而是测量**封闭 LIBERO 任务集合的策略容量下限**。四个标准 suite 一共 40 个已知任务，在训练与测试同样的场景类型、任务指令下，任务语义可以被一个 40 路索引替代。作者把通用语言能力去掉，再问：只为了在这 40 个已知技能里正确选择和闭环执行，需要多少参数？

![MINERVA 端到端数据流](/Embodied-AI-Roadmap/diagrams/minerva-libero-system.svg)

*图 1｜根据论文 §III-B 重绘：双相机、任务 ID、共享轻量视觉编码器、历史与机器人状态、动作生成/回归两条头；这是信息流教学图，未公开的中间特征维度不臆造。*

**阅读路线**：先看 [π₀.₅ 的 LIBERO 数据与 Loss 契约](/Embodied-AI-Roadmap/01-foundations/vla/pi05-libero-data-loss-lora-source-guide/)，再看本文 §2 架构 → §3 动作建模 → §4 闭环 → §5 泛化与统计 → §6 复现。对比 [PACT-VLA 推理与观测延迟](/Embodied-AI-Roadmap/05-control/pact-vla-precommit-flowres/) 时，要区分大型 VLA 和小策略的计算预算。

## 1. 读论文的第一道陷阱：95.05% vs 97.50% 并非严格同协议

| 原论文 Table I | 参数 | Spatial | Object | Goal | Long | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| MINERVA-0.5M：FM | 0.54M | 94.4 | 99.6 | 96.4 | 89.8 | **95.05** |
| 同架构：L1 回归 | 0.54M | 96.8 | 99.6 | 97.4 | 89.2 | 95.75 |
| MINERVA-1M | 0.99M | 97.0 | 99.8 | 97.4 | 92.8 | 96.75 |
| MINERVA-5M | 4.89M | 99.0 | 99.2 | 98.0 | 92.4 | 97.15 |
| MINERVA-10M | 9.66M | 98.4 | 99.0 | 98.4 | 94.0 | 97.45 |
| π₀.₅（LeRobot 参考） | 4.1B | 97.0 | 99.0 | 98.0 | 96.0 | 97.50 |

作者用 **50 episodes/task × 40 tasks = 2,000 rollouts** 测 MINERVA，而 LeRobot π₀.₅ 行是公开 **10 episodes/task × 40 = 400 rollouts** 的参考值。二者还存在实现、checkpoint 与训练种子差异；**2.45 个百分点差距不能当成严格配对实验结论**。论文确实证明了：在*标准 LIBERO 的封闭任务分布*里，训练好一个仅约 54 万参数的网络就可以得到 95% 左右的成功率。

对自动驾驶可类比：一个只在 40 条已知道路和指定车道上测试的规划器，得高分不代表它具备开放道路中的通用语义理解。实验协议决定“高分”究竟证明了什么。

## 2. 逐层架构：把两路 RGB 变成 16 步动作

### 2.1 张量账本与数据含义

论文 §III/IV 可确认：两路图像 agent-view / wrist-view，原始 $256\times256$，缩至 $144\times144$，训练随机裁到 $128\times128$、评测中心裁剪；最近 **2** 个观测时刻；机器人 proprioception **8D**；动作 chunk 长度 **16**。以下 batch 轴顺序是**本站的示意数据契约**，不代表作者源代码 API：

| 名称 | 张量示例 | 应检查的语义 |
|---|---|---|
| RGB | $[B,2,2,3,128,128]$ | batch / obs history / view / channel / height / width |
| State | $[B,2,8]$ | 8 维机器人状态，单位与关节/末端含义需核对版本 |
| Task ID | $[B]$ | 40 个已知任务的查表编号，**不是语言 token** |
| Future actions | $[B,16,D_a]$ | 16 个物理控制时刻的目标动作 |
| Condition $c$ | $[B,D_c]$ | 两路视觉、两帧历史、状态与任务向量的拼接 |
| Flow velocity / L1 output | $[B,16,D_a]$ | 对整块动作生成或回归 |

标准 LIBERO 常见机器人动作接口为末端 6D delta 加夹爪 1D，即 **$D_a=7$**，但必须以实际 adapter 的排列、单位、旋转和夹爪符号为准；**不能**直接把 openpi 的 32 维 padding 容器当成 32 个有效关节。

### 2.2 为什么共享 CNN 加 view embedding？

两个相机通过同一套从零训练的 CNN，避免给每个相机复制一套视觉参数；learned view embedding 告诉模型这是全局相机还是腕部相机。骨干由四级 stride-2 depthwise-separable residual blocks 组成，0.54M 版本的通道宽度依次是 **32/64/112/160**，1M 版本是 **40/80/160/224**。若仅按 stride-2 估算，128 大小的空间分辨率经过四级是 **64→32→16→8**，但具体残差分支内部 kernel、norm 次序不能从论文摘要臆造。

普通卷积核主要参数量 $k^2C_{\rm in}C_{\rm out}$；Depthwise + Pointwise 为：

$$
P_{\rm sep}\simeq k^2C_{\rm in}+C_{\rm in}C_{\rm out}.
$$

后者在通道数大时大幅节约参数。**轻量不等于只减少 action head**：原论文的一个主要发现是，预算优先给“眼睛”（视觉编码器）更有效。

### 2.3 FiLM 如何让任务影响视觉？

task ID 经 embedding 得到 $e$；FiLM 根据 $e$ 对 CNN 特征做逐通道仿射调制：

$$
\operatorname{FiLM}(F,e)=\gamma(e)\odot F+\beta(e).
$$

这里 $F\in\mathbb R^{B\times C\times h\times w}$，$\gamma,\beta\in\mathbb R^{B\times C}$ 在空间维度广播。相同桌面输入下，任务 0 可以更关心黑碗，任务 1 可以更关心篮子边缘。但是 **40 个任务 ID 只能选择训练时见过的条件**；不会自动理解此前未出现的语言、目标或新技能。

### 2.4 Spatial Softmax 为什么省计算？

让每个 CNN 特征通道在空间上形成归一化的概率分布：

$$
p_{cij}=\frac{\exp(F_{cij}/T)}
{\sum_{u,v}\exp(F_{cuv}/T)}.
$$

提取期望坐标作为连续关键点：

$$
x_c=\sum_{ij}p_{cij}x_{ij},\qquad
y_c=\sum_{ij}p_{cij}y_{ij}.
$$

这样每个通道用 2 个坐标承载与任务有关的位置，而不是把全部像素扁平展开到动作头。原论文 0.54M 用 **48 个 learned keypoints**，1M 用 **64 个**；它们不是人工标的手部 21 个关键点，也不是物理上预定义的物体特征点。关键点位置对背景和光照仍然可能极其敏感。

### 2.5 从历史到动作专家：MLP-Mixer 取代 Self-Attention

8D state 经过两层 MLP；两路特征、状态、task embedding 与两帧时间历史拼成条件向量 $c$。动作头基于 **4 层 DiT 风格块**，但对 16 个 action tokens 的交互使用 token-mixing MLP，而不是 QK softmax Attention。AdaLN-Zero 的调制投影参数在各块之间共享，降低额外参数。

对 $X\in\mathbb R^{B\times H\times D}$，token-mixing 的一种教学式是把 $X$ 沿 $H$ 轴先变换：

$$
Y=X+\left[W_2\phi(W_1\,\operatorname{Norm}(X^\top))\right]^\top.
$$

上式仅解释“沿 16 个动作时刻混合信息”的机制，不代表论文披露了精确的投影维度或全部 forward 顺序。消融显示将 MLP mixer 换回 self-attention，参数量增加约 **26%**，单 seed 成功率变化仅 **+0.05 个百分点**，不足以断言 self-attention 有实质优势。

## 3. Flow Matching 与直接 L1：两者到底训练了什么？

**注意流时间方向**：本文根据 MINERVA 论文采用 **噪声端 $\tau=0$、动作端 $\tau=1$**。不要直接复制采用相反时间约定的 openpi 实现。

训练取真实动作块 $A$ 和标准噪声 $\epsilon$：

$$
x_\tau=(1-\tau)\epsilon+\tau A,\quad
v^*=A-\epsilon,\quad
\epsilon\sim\mathcal N(0,I).
$$

速度场监督：

$$
\mathcal L_{\rm FM}=
\mathbb E_{A,\epsilon,\tau}
\left\|v_\theta(x_\tau,\tau,c)-(A-\epsilon)\right\|_2^2.
$$

推理从 $x_0=\epsilon$ 开始，对 **10 次 Euler 采样步**进行积分：

$$
x_{\tau+\Delta\tau}=x_\tau+\Delta\tau\,v_\theta(x_\tau,\tau,c).
$$

**直接 L1** 不引入流时间，输出一个完整动作块 $\hat A_\theta(c)$，目标是：

$$
\mathcal L_{\rm L1}=\mathbb E_{A,c}
\frac1{HD_a}\sum_{h=1}^H\sum_{d=1}^{D_a}
|\hat A_{\theta,h,d}(c)-A_{h,d}|.
$$

在固定任务、动作分歧较少的分布里，L1 的单一条件输出可能已经足够；当接触操作存在不同合理模式时，FM 仍可能更适合建模多峰分布。论文对 1M 模型 **3 个训练 seed** 的消融：FM **95.63±1.08%**、L1 **95.97±0.80%**，差 **+0.34 个百分点**，在 seed 波动内。**这是“未检测到优越性”，不是“数学证明 FM 无用”**。

在同硬件、0.54M 规模，FM 推理 GPU **8.2ms** / CPU **8.9ms**，L1 GPU **2.2ms** / CPU **5.1ms**。论文报告的 3.8 倍主要是对应 GPU 单次动作块耗时；真实机器人还需加图像传输、控制器延迟和动作同步。

## 4. 为什么 Action Chunk=16，实际每 tick 重规划？

必须定义三个独立时间尺度：

$$
H=\text{一次预测的动作步数},\quad
K=\text{Flow 数值积分次数},\quad
E=\text{一次提交后开环执行的动作步数}.
$$

论文默认 $H=16,\ K=10$（FM 时），最佳部署实验是 $E=1$：**预测 16 步，但先执行第一个控制动作，下一 tick 再观察并重新规划**。不是生成 16 步就开环执行 16 步。

| Table II（3 seeds） | 成功率（%） | vs H=16 |
|---|---:|---:|
| H=16 | 95.63±1.08 | — |
| H=8 | 92.47±0.84 | -3.16 |
| H=32 | 93.43±0.71 | -2.20 |
| 视觉预算由 0.49M 减到 0.13M | 94.22±0.20 | -1.41 |

论文分析短 chunk 在 **Goal** 任务容易徘徊在不同目标，长 chunk 在 **Long** 多阶段任务容易反应慢；$H=16$ 是**该数据/控制频率/策略族**的经验最佳点，不可照搬到 30Hz 工业双臂。

### Temporal Ensembling：连续预测如何平滑？

若时刻 $s$ 生成 chunk $\hat A_s$，在当前 $t$ 时刻可以找到过去多个 chunk 对“现在”的预测 $\hat a_{s,t-s}$，平均：

$$
a^{\rm ens}_t=
\frac{\sum_{s:0\le t-s<H}w_{s,t}\,\hat a_{s,t-s}}
{\sum_{s:0\le t-s<H}w_{s,t}}.
$$

$w$ 是时间权重，一种**教学实现**可取 $w_{s,t}=\exp[-\lambda(t-s)]$（近期计划权重大），原论文是否采用完全相同权重须看作者代码，不作断言。

论文筛选实验在同一 0.54M checkpoint 上报告：ensemble **96.3%**、BID **95.0%**、plain **94.1%**、RTC-style **91.8%**。这四个数是**25 episodes/task 的方法筛选**，不是最终 2,000 rollout 的同一张实验表。作用机制：重规划有利于及时响应新观测，ensemble 减轻每次生成的随机抖动；但旧轨迹权重过高又会增加时延。

![标准 LIBERO 与泛化评测隔离](/Embodied-AI-Roadmap/diagrams/minerva-libero-protocol.svg)

*图 2｜固定任务高分、任务 ID 置换与 LIBERO-Plus 数据分布扰动，必须分别报表。*

## 5. 任务 ID 置换的因果探针，比单看成功率更重要

冻结 1M 网络、完全不改参数，只更换 task→ID 映射：

| 测试 | 成功率 |
|---|---:|
| 原任务 ID | **96.75%** |
| 同 suite 内 ID 轮换 | **6.5%** |
| 全部任务强制映射到 suite 任务 0 | **16.2%** |

因此高分的一大组成是“查表选择哪一种熟悉的策略”；任务编号错误时，机器人大概率执行错技能。作者也指出 Object suite 的视觉上下文能够部分补偿错误编号，不能极端解释为“网络仅依赖 ID、完全不看图像”。

建议对真实 VLA 分成两个 probe：**task-ID permutation**（适合 MINERVA）和 **instruction paraphrase / contradictory instruction**（适合真语言 VLA）。没有语言编码器的模型不应该被硬拿去测试零样本自然语言理解。

## 6. LIBERO-Plus：95% 为什么在相机和光照变化下崩溃？

论文 Table IV，0.54M 模型对各种扰动的成功率：

| 评测条件 | 成功率（%） |
|---|---:|
| 语言改写（视觉不变、作基线） | 96.7 |
| 新物体 | 69.7 |
| 传感器噪声 | 68.5 |
| 机器人初始状态 | 57.1 |
| 物体布局 | 50.0 |
| 相机变化 | 11.5 |
| 背景变化 | 9.3 |
| **光照变化** | **1.1** |
| 七类扰动平均 | **46.7** |

更大的 0.99M 仅 **46.0%**，4.89M 为 **55.7%**。这是按扰动因子分层抽样的评价，不是标准 2,000 次 rollout 重跑。证明 standard LIBERO 能解好，并不意味着模型具备面对真实工厂灯光、相机重装、反光金属零件的鲁棒性。现实 EGO/UMI 数据可采集不同曝光、不同背景与不同机位作为独立 holdout，绝不能把同一个工位的帧随机切分就声称泛化。

## 7. 统计与可重复性：小于 1 个百分点的提升别急着宣布有效

论文在关键消融里重新跑了 3 个训练 seed，观察到大约 **±1 个百分点**的平均成绩波动。有些论文展示的 +0.3% 可能只是训练种子的偶然差别，不能当成稳定增益。

对单独一组任务，成功次数 $k$、总数 $N$ 的点估计：

$$
\hat p=\frac{k}{N}.
$$

粗略 95% Wilson 区间：

$$
\frac{\hat p+z^2/(2N)\pm
z\sqrt{\hat p(1-\hat p)/N+z^2/(4N^2)}}
{1+z^2/N},\qquad z=1.96.
$$

但是多任务 episode 不是严格 i.i.d.，所以**不能只报合并 2,000 episode 的置信区间**。同时报告每任务/每 suite 成功率、训练 seed 方差、任务分层区间；模型 A/B 最好在相同初始状态与种子下成对评测。

论文另一组数据：同一小模型家族在 **LIBERO-90 的 89 个有示范任务**达到 **94.6%**，但该试验是单 seed、每任务 10 episode，仍然是一个扩大后的封闭任务集，而不是新任务零样本。

## 8. 从“读完”到“能做”的四级复现实验

**Level 0：无 GPU 的数学与数据契约。** 获取本站 [CPU 合成验证脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/minerva_eval_contract_demo.py)，运行：

~~~bash
python3 examples/minerva_eval_contract_demo.py
~~~

脚本检查 **H=16 的 chunk 重叠和每步重规划、Wilson 区间、Task ID 映射打乱、H/K/E 三时间尺度**；只验证定义，**不是**训练 MINERVA，也不是复现实验成绩。

**Level 1：读取真实数据并检查同步。** 固定 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) / [LeRobot](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx) 的代码/数据 revision。论文使用约 **1693 段示范、27.3 万帧、40 任务**；训练设定 **batch 128、90–120k steps、AdamW lr 1e-4、cosine decay**，sub-1M 的 Long 数据样本 loss 上调 3 倍。将每类摄像机视频、8D 状态和动作在统一时钟可视化，确保各个 episode 末尾的未来 chunk 不跨 episode 拼接。

**Level 2：研究性重实现（无官方完整源码可比对）。** 先用少量片段验证 2 图像输入 → 共享 CNN（FiLM）→ Spatial Softmax → 2 帧历史 + state MLP → 16 步 L1 动作输出的前向、shape、loss 和 10 条轨迹过拟合。之后增加速度场与 Euler 采样用于 FM 消融。论文报告单模型约 2 小时 GH200/RTX 5080 GPU 训练，不能将此估计套到 16GB 内存的无 GPU 笔记本。

**Level 3：正式闭环与泛化。** 四 suite 每任务至少 50 episode；分别统计 Long 子任务失败、实时动作 replan 耗时。再做 task-ID permutation 与 LIBERO-Plus 相机、光照、布局扰动。固定初始条件对比 H=8/16/32、FM/L1、视觉预算；每组至少 3 个训练 seed。保存每个 episode 的 policy checkpoint hash、norm stats、success、steps、latency、failed reason、视频。

### 8.1 工业场景如何用它指导部署？

对于固定工位、固定工序的双臂装配，可以让高层 VLA/Agent 负责**语言理解、技能选择、失败恢复**，小网络负责**单一技能的高频动作控制**。第一期只选固定料盒抓取技能，从 UMI/遥操数据训练轻量 imitation baseline，与 π₀.₅ 同任务比较，重点测 **新零件/新灯光/新相机位姿的成功率、接管率、吞吐与安全**；此工业架构是本站提出的工程建议，不是 MINERVA 作者已在真机完成的实验。

## 9. 最终验收口径：什么可以写 PASS？

| 项目 | PASS 必须有的证据 | 不可替代为 |
|---|---|---|
| 源码/教程审阅 | 论文原文、官方 LIBERO/LeRobot 接口 | 社区文章引用 |
| CPU 契约实验 | 实际标准输出有断言全部通过 | “脚本写好了” |
| 训练复现 | GPU 日志、配置、weights、loss、seed | 合成脚本通过 |
| 仿真成功率 | 40 tasks×50、固定初始状态、完整报告 | 单场景 10 次 |
| 泛化能力 | LIBERO-Plus 等分布外单轴扰动 | 标准 LIBERO 95% |
| 真机部署 | 多场景多次试验、急停/碰撞记录 | 仿真视频 |

---

**参考（2026-10-31 校核）**：[MINERVA §III–VI、原表 I–IV](https://arxiv.org/html/2609.03715) · [LIBERO 官方](https://github.com/Lifelong-Robot-Learning/LIBERO) · [LeRobot LIBERO](https://github.com/huggingface/lerobot/blob/main/docs/source/libero.mdx) · [openpi 官方 LIBERO](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero)。**复现状态：静态技术调研与研究性实验方案，暂无 MINERVA GPU 训练、2000 rollout 或真机实测证据。**