---
title: "FlowDPG 深度教程：无 ODE 反传的 Flow Matching 策略梯度与工业双臂后训练"
description: "解构 CoRL 2026 FlowDPG：单步干净动作投影、IQL 双 Critic、梯度蒸馏、阶段奖励、离线到在线闭环、数学推导及 CPU 验证"
sidebar:
  order: 9
---

> **一手证据核查｜2026-11-08**：依据 [CoRL 2026 论文 PDF（作者项目镜像，17 页）](https://flowdpg.github.io/static/images/main.pdf)、[arXiv 2606.22303](https://arxiv.org/abs/2606.22303)、[FlowDPG 作者项目页](https://flowdpg.github.io/)以及[项目网站源码仓库](https://github.com/FlowDPG/flowdpg.github.io)交叉核查。作者项目仓库主要是网站、论文、视频及图表，**本次未找到作者公开的可运行 FlowDPG 训练算法仓库、checkpoint 或复现配置**；因此本文的公式解读来源于论文，CPU 脚本为本站依据原式独立编写的合成契约，**不是原论文代码复现**。论文效果数值仅代表作者报告，未独立重复。

# FlowDPG：把 Critic 对动作的梯度，教给 Flow Matching 速度场

## 0. 先回答三个工程问题

**FlowDPG 是什么？** 它是一种让 *已经训练好的 Flow Matching 机器人策略* 继续利用离线示教、失败轨迹、真机执行反馈进行价值优化的 actor–critic 后训练方法。它不是从 EGO 视频直接训练策略的 H2R 算法，也不是把 VLA 换成另一套单步动作头。

**它为什么有用？** 假设机器人执行 8 段连续工序，每段 95% 成功，独立近似下端到端概率只有 $0.95^8\approx 66.3\%$。复杂工业任务的瓶颈常常不是基本动作不会做，而是接触、重抓、误位姿恢复导致误差累积。**长程表现来自一个可继续感知和恢复的策略、可评价的阶段反馈与在线数据，而不是 FlowDPG 自带子任务规划器。**

**它和 π₀.₅ 什么关系？** π₀.₅ 使用连续动作 Flow Matching 专家；理论上可尝试在该动作专家上叠加 FlowDPG 式 Critic 目标。但论文的实证对象是作者自己的 flow 策略，**并未展示直接对开源 openpi π₀.₅ 的训练结果**。接入 π₀.₅ 必须重新验证 prefix 条件、动作归一化、损失掩码、冻结范围和优化稳定性。

相关阅读：[Flow Matching 基础](/Embodied-AI-Roadmap/00-prerequisites/flow-matching/) → [π₀.₅ 数据、LoRA 与 Flow Loss](/Embodied-AI-Roadmap/01-foundations/vla/pi05-libero-data-loss-lora-source-guide/) → 本文 → [PACT-VLA 异步执行](/Embodied-AI-Roadmap/05-control/pact-vla-precommit-flowres/)。

![FlowDPG 离线与在线一体化训练结构](/Embodied-AI-Roadmap/diagrams/flowdpg-system.svg)

*图 1：本站重绘的论文结构概念图。离线用示教启动，在线部署真机并追加轨迹；Critic 在训练时作用于速度场，推理时不需要 Critic。*

## 1. 核心科学问题：常规 DPG 为什么难用于多步 Flow Policy？

令条件状态 $s$ 汇总当前相机、机器人状态和必要的语言条件；策略把高斯噪声 $x_0$ 连续地积分到 $x_1$：

$$
x_0\sim\mathcal N(0,I),\qquad
\frac{dx_t}{dt}=v_\theta(x_t,t,s),\qquad
x_1=\operatorname{ODEsolve}(v_\theta,x_0,s).
$$

注意 $t\in[0,1]$ 是 **flow 数值时间**，不是机器人真实时间；动作块长度记为 $H$、总展平维度 $D=H D_a$。普通确定性策略梯度若取 $-\mathbb E[Q_\phi(s,x_1)]$ 作 actor 损失，必然需要

$$
\nabla_\theta Q_\phi(s,x_1)
=\left(\nabla_{x_1}Q_\phi(s,x_1)\right)^\top
\nabla_\theta x_1.
$$

最后一项穿过每一步 ODE 更新的 Jacobian 乘积。在离散 Euler 求解器里，$x_{t+\Delta t}=x_t+\Delta t\,v_\theta(x_t,t,s)$，出现

$$
J_t=I+\Delta t\,\frac{\partial v_\theta}{\partial x_t},\qquad
\nabla_\theta x_1\ \text{含有}\ \prod_t J_t.
$$

数值上这会出现梯度爆炸、消失，显存与求解步数相关。**FlowDPG 的目标不是让 Flow 推理变成一步，而是让训练的价值梯度不必穿过完整的 ODE 解算器。**

### 1.1 为什么不直接用 AWR、RECAP 或单步残差？

| 技术路线 | 学习信号 | 能否改变原多步 Flow 速度场 | 代价与边界 |
| --- | --- | --- | --- |
| BC / 行为克隆 | 已示教动作 | 是（追随示教） | 难超出数据行为支持 |
| AWR / RECAP 等 | 标量优势、条件 | 通常仍受已有动作分布限制 | 很难直接合成新的高价值接触路径 |
| 独立残差 / adapter | 额外小策略 | 原 Flow 可冻结 | 接口简单，但上限受小模块约束 |
| 全 ODE DPG | 终端动作的 Critic 梯度 | 是 | ODE BPTT 昂贵且不稳定 |
| QAM 等 adjoint 路线 | 沿路径价值梯度 | 是 | 对价值误差敏感 |
| **FlowDPG** | 在干净动作投影处查询一次 Critic 梯度 | **是，直接蒸馏回速度场** | 依赖 Critic 及投影准确性 |

上述是作者论文 §2、§3 的对比思路，不是本站统一任务下的 benchmark。

## 2. 输入输出、张量与动作语义

| 张量 / 模块 | 教学示例形状 | 含义 |
| --- | --- | --- |
| RGB images | $[B,V,3,h,w]$ | $V$ 台相机当前帧；Franka 实验用前、上、双腕，共 4 视角 |
| Proprio state | $[B,D_s]$ | 当前关节/末端/夹爪等状态（具体字段依本体） |
| Action chunk $a=x_1$ | $[B,H,D_a]$ | 真实示教或回放的未来 $H$ 步，必须采用统一 action contract |
| Noise $x_0$ | $[B,H,D_a]$ | 独立高斯噪声，**不是传感器观测噪声** |
| Flow time $t$ | $[B,1,1]$ | 每个样本采样一项随机 flow 时间，按动作维广播 |
| Velocity $v_\theta$ | $[B,H,D_a]$ | 预测噪声向数据移动的速度 |
| Critic $Q_{\phi_i}(s,a)$ | $[B,1]$ | 完整动作块的预期回报，双 Critic 抑制高估 |
| Critic gradient $g$ | $[B,H,D_a]$ | 对连续动作的梯度，**不是对图片像素的梯度** |
| One-step projection $\hat a$ | $[B,H,D_a]$ | 在当前 flow 时间估计的干净动作块 |
| Target $u_t^*$ | $[B,H,D_a]$ | 含 Critic 改善方向的速度学习目标 |

这里的 $H,D_a,D_s,h,w$ 是**教学符号，不是假称论文发布了统一固定张量配置**。Franka 和 YAM 的关节及控制接口不同；开放的 π₀.₅ 容器可能包含 padding 和 action validity mask，先核对解码器定义再复现。

尤其要区分：**action chunk** 是一段真实机器人控制时间上的目标序列；**flow time** 则是在同一个动作块的随机变量上进行的计算时间。

## 3. 从普通 Flow Matching 推到 FlowDPG：逐式推导

### 3.1 行为克隆速度：Eq. (4) 之前的起点

取示教动作 $x_1$、高斯 $x_0$，按线性路径插值：

$$
x_t=(1-t)x_0+t x_1,\qquad u_t=x_1-x_0.
$$

普通 Flow Matching 的损失为

$$
\mathcal L_{\mathrm{FM}}
=\mathbb E_{s,x_0,x_1,t}
\left[\lVert v_\theta(x_t,t,s)-u_t\rVert_2^2\right].
$$

在 $(x_t,t,s)$ 附近，网络知道的是局部速度，不一定知道完整解算到 $t=1$ 后的真正落点。要在动作空间优化价值，需要一种便宜、贴近示教的落点估计。

### 3.2 单步投影：Eq. (4)

论文构造：

$$
\boxed{\hat a=x_t+(1-t)v_\theta(x_t,t,s)}.
$$

解释：假设剩余 $1-t$ 的 flow 路径速度近似不变，当前点沿该速度走完剩余时间。在完美直线速度的理想情形 $v_\theta=u_t$，代入可得

$$
\hat a
=(1-t)x_0+t x_1+(1-t)(x_1-x_0)
=x_1.
$$

但真实流场是非线性的，$\hat a$ 不一定等于多步 ODE 终点；它只是**训练时的一步近似**，而且近似误差会影响 Critic 梯度是否可信。

### 3.3 双 Critic 的动作梯度：Eq. (5)

$$
\boxed{
g=\nabla_{\hat a}
\left[\min_{i\in\{1,2\}} Q_{\phi_i}(s,\hat a)\right].
}
$$

把状态及 Critic 权重视为固定，只对输入动作 $\hat a$ 求梯度。**实现中应 detach 投影到 Critic 的传递路径、冻结 Critic 参数梯度，仍允许对 $\hat a$ 求 $\partial Q/\partial a$**。如果把整个 Critic 前向包进无梯度上下文，就连 $g$ 也算不出来。

采用最小值的原因：双 Critic 常见地降低过高估计；它并不保证在分布外动作上正确，取最小值的切换点也可能造成非光滑梯度。

### 3.4 自适应步长：Eq. (6)

$$
\boxed{
\Delta=\alpha\frac{\lVert u_t\rVert_2}{\lVert g\rVert_2+\varepsilon}g
},
\qquad\alpha>0.
$$

如果简单用 $\alpha g$，各样本不同 Q 数值刻度会使动作调整幅度忽大忽小。归一化后在 $g$ 非零且 $\varepsilon$ 很小时，

$$
\lVert\Delta\rVert_2\approx\alpha\,\lVert u_t\rVert_2.
$$

这个动作尺度是 **速度幅值的相对比例**，并不是毫米、弧度或牛顿的物理安全限幅。真实部署还必须在 **归一化前后** 统一位置/转角/夹爪的量纲与限幅策略；如梯度趋近零，应允许 $\Delta$ 趋近零而非强制放大噪声。

### 3.5 蒸馏新的动作与速度：Eq. (7)–(8)

$$
a^*=\hat a+\Delta,\qquad u_t^*=a^*-x_0.
$$

FlowDPG 用 $a^*$ 构造一条新的直线参考速度，训练本来的 $v_\theta$：

$$
\boxed{
\mathcal L_{\mathrm{FlowDPG}}
=\lambda\lVert v_\theta-u_t^*\rVert_2^2
+(1-\lambda)\lVert v_\theta-u_t\rVert_2^2
}.
$$

第一个是 **Critic 引导蒸馏**，第二个是 **行为克隆锚**。论文采用从 $\lambda=0$ 起线性 warmup，避免在 Critic 还不可靠时就让策略偏离示教。这里的 $\lambda$ 是损失混合权重，不是双臂权重或离线/在线数据占比。

> **最易犯的实现错误**：直接把 $-\!Q(s,\operatorname{ODEsolve}(v_\theta))$ 反传到 actor，结果又走回 BPTT；另一种错误是让 $a^*$ 随 actor 更新一起反向传播，产生非预期二阶/目标漂移。教学实现应将 $a^*,u_t^*$ 当成构建后的 *stop-gradient 监督目标*，再更新当前 $v_\theta$。

### 3.6 为什么还需要一致性损失：Eq. (9)

$$
\boxed{
\mathcal L_{\mathrm{cons}}
=\mathbb E[\lVert \hat a-x_1\rVert_2^2]}.
$$

代入线性 $x_t$ 得到精确代数恒等式：

$$
\hat a-x_1=(1-t)(v_\theta-u_t)
\Rightarrow
\mathcal L_{\mathrm{cons}}
=\mathbb E[(1-t)^2\lVert v_\theta-u_t\rVert_2^2].
$$

这不是再添加一个神秘的预测器，而是**在较小 $t$、单步投影误差最敏感的区域增加速度监督**。否则 $\hat a$ 可能偏离示教动作流形，让 Critic 在远离真实动作的 OOD 区域“指挥”策略。

![单步投影、Critic 梯度与动作蒸馏的数值链路](/Embodied-AI-Roadmap/diagrams/flowdpg-gradient-detail.svg)

*图 2：真实网络计算的接口示意；蓝线为示教速度，绿色为局部高价值方向。注意 $a^*$、$u_t^*$ 是训练目标，不是额外的真机直接指令。*

### 3.7 与经典 DPG 的关系：**三个近似**，不是严格等价

论文 §3.2 给出完整 ODE 上的 DPG 梯度：

$$
\nabla_\theta\mathcal L_{\mathrm{DPG}}
=-\int_0^1
g^\top\Phi(1,t)\,
\nabla_\theta v_\theta(x_t,t,s)\,dt,
$$

其中 $\Phi(1,t)=\partial x_1/\partial x_t$ 是 ODE 的灵敏度矩阵，包含沿路径的 Jacobian 乘积。FlowDPG 以

$$
[\nabla_\theta\mathcal L_{\mathrm{FlowDPG}}]_g
\ \propto\
-2\lambda\alpha\lVert u_t\rVert\,
\widehat g(\hat a)^\top \nabla_\theta v_\theta(x_t,t,s)
$$

代替完整梯度。作者显式承认三个近似：(1) 用 $\hat a$ 近似终点，且令 $\Phi$ 近似单位矩阵；(2) 用随机一个 $t$ 的 Monte Carlo 点近似路径积分；(3) 以自适应速度尺度代替 $g$ 的原始幅值。这解释了为什么可以大幅简化，却**不能宣称已无偏恢复标准 DPG**。

## 4. 为什么是双 Critic + IQL，而不是直接拟合失败次数？

**IQL（Implicit Q-Learning）** 避免在 value bootstrap 时用未见过的最优动作估值，核心是针对数据中动作 $a$ 的两阶段学习。记 $\overline Q_i$ 为 EMA 目标 Critic，状态值为 $V_\psi$：

$$
\rho_\tau(u)=\big|\tau-\mathbf1[u<0]\big|u^2,
\qquad \tau>0.5,
$$

$$
\mathcal L_V=
\mathbb E_{\mathcal D}
\left[\rho_\tau\left(
\min_i \overline Q_i(s,a)-V_\psi(s)
\right)\right].
$$

$\tau>0.5$ 更重视高回报尾部，但**expectile 不是取硬最大值**，也不能保证建模噪声消失。随后用动作块结束状态：

$$
y_Q=r_t+\gamma^H V_\psi(s_{t+H}),\qquad
\mathcal L_Q=\mathbb E_{\mathcal D}[(Q_{\phi_i}(s_t,a_t)-y_Q)^2].
$$

这是 chunk-level bootstrap：$H$ 为执行到下一次评价的控制步数，$\gamma^H$ 反映这段实际时间长度。若提前终止，应屏蔽 bootstrap，且**不能跨 Episode 读下一个任务的状态**。论文采用双 Critic 与其 EMA target 降低过高估计和时序漂移。

### 4.1 SARM 阶段奖励如何把长任务变得可训练？

论文借助独立的 **Stage-Aware Reward Model（SARM）**。它从观测估计阶段 $z(o)$ 以及进度 $\varphi(o)\in[0,1]$，定义：

$$
\boxed{
r_t=
\underbrace{\varphi(o_{t+H})-\varphi(o_t)}_{\text{阶段内进度}}
+\underbrace{\beta_{\mathrm{stage}}\mathbf1[z(o_{t+H})>z(o_t)]}_{\text{阶段跃迁奖励}}
+\underbrace{r_t^{\mathrm{term}}}_{\text{最终完成奖励}}
}.
$$

例子：工业装配流程包含“夹取→对准→插接→复位”。仅有最终成败奖励时，中间阶段失误没有足够信用分配信号；引入进度和阶段切换，让 Critic 在局部连续动作之间形成较可用的排序。

但阶段标签是**奖励模型学习出来的**，不是机器人真实物理状态的自动真值；它可能错判“已插好”或遮挡状态。更重要的是，**这篇论文确实需要阶段语义和一定量的人工标注**，并不是“无标注数据直接做 RL”。

### 4.2 实验里的长程是怎么来的？

作者使用 GELLO 收集真机演示、构建一个覆盖全流程的单策略、用 SARM 得到阶段反馈，再进行离线训练与在线执行回流。AirPods 示例包含夹盒、开盖、左右耳机抓插、合盖、放回共 8 阶段；蛋料理包含 14 阶段和工具使用。

**不等价于**“先训练八个独立 Diffusion Policy 再调用任务规划器”。文中说的是一个策略从当前图像持续重规划、必要时再抓取与重试；它仍需要已覆盖基本技能的演示分布和真实反馈。**无需事先训练 DP 才能使用 FlowDPG**，但必须先有能输出有效动作块的 Flow Matching 行为策略。

## 5. 离线→在线完整训练管线（论文附录 B Algorithm 1）

| 阶段 | 数据与更新 | 必须记录的质检字段 |
| --- | --- | --- |
| 0 示教与对齐 | GELLO/机器人遥操示教，统一相机、关节、动作 chunk | episode_id、timestamp、task_id、相机同步、action 语义 |
| 1 阶段奖励 | 少量阶段标注训练 SARM，输出 $z,\varphi$ | 标注一致性、进度单调性误差、阶段跳跃错误 |
| 2 离线值学习 | IQL $V,Q_1,Q_2$；含目标 Critic EMA | TD 误差、Q disagreement、terminal mask |
| 3 离线 actor | 每样本 $x_0,t,x_t,u_t$，做一次 $\hat a$ 与 $g$，蒸馏含 BC 锚的速度 | $\lVert \hat a-a\rVert$、$\lVert g\rVert$、$\lVert \Delta\rVert$、loss 分项 |
| 4 在线部署 | 用原来的多步 Flow ODE 输出动作块；真机保存成功和失败 | 成败、接管、阶段停止位置、碰撞/超时 |
| 5 异步回流 | 离线/在线轨迹按大致 **1:1** 采样更新 | 样本来源与版本、reward 置信度、数据泄漏检测 |
| 6 再发布 | 定期同步新 actor 到推理端，并滚动评价 | checkpoint SHA、成功率、重试、插入精度、回滚开关 |

论文要求：在线阶段冻结共享视觉 backbone 与 SARM，更新 actor 和 critic；**BC 锚只在示教与成功 rollout 上计算**，不是对所有失败动作再做监督克隆。

### 5.1 必须分清训练与推理的计算图

训练时：$\text{camera, proprio, action, reward}\rightarrow\text{feature}\rightarrow(Q,V)\rightarrow g\rightarrow u_t^*\rightarrow \mathcal L$，并执行新参数的优化；推理时：$\text{camera, proprio}\rightarrow v_\theta\rightarrow\text{ODE integration}\rightarrow \text{action chunk}$。**不需要在机器人推理端调用 Q 网络或生成奖励。**

伪代码重点不是单行 torch 写法，而是梯度流：

1. 从示教/在线 buffer 构造有因果性的状态 $s_t$、动作块 $a_t$、下一状态和 reward，保证每块不跨 episode。
2. 采 $x_0,t$，计算插值 $x_t$ 与 $u_t$；在当前 actor 得到 $v_\theta$。
3. 求 $\hat a$；**阻断 actor 到 Critic 评分路径的二阶梯度**，但在动作输入位置计算 $g=\nabla_{\hat a}\min Q$。
4. 用每个样本自己的 $\lVert u_t\rVert,\lVert g\rVert$ 构造 $\Delta$，构造 detached 的 $u_t^*$。
5. 混合蒸馏损失、BC 锚与一致性损失，单独更新 actor；按 IQL 目标更新 $V,Q_1,Q_2$，更新 EMA。
6. 在线阶段：reward/进度失败检查、固定量数据回流、周期性对照旧策略；随时可用旧 checkpoint 回滚。

## 6. 结果与分母：必须按原论文口径解释

论文表 1 使用 **25 次 AirPods 整体装配尝试**；整体成功要按顺序完成双耳机插接与收尾。BC 基线整体成功率 $64\%$、QAM $80\%$、FlowDPG 离线 $88\%$、FlowDPG 加在线 $92\%$。这是作者的同任务比较，**不是 92% 的通用工业场景成功率**。

论文另外进行跨 5 checkpoint、每个 checkpoint 10 次的重复评测，总 50 rollout/cell。此时 FlowDPG 的离线 / 在线 AirPods 整体成功率约为 $86\%/90\%$，**不能把表 1 的 92% 和重复评估的 90% 混写为同一次验证**。蛋料理只执行离线比较，文中报告阶段 rubric 约从 $64.2\%$ 升到 $85.8\%$，不是端到端“86% 成功率”。

| 作者评测指标 | 离线 FlowDPG | 在线 FlowDPG | 备注 |
| --- | ---: | ---: | --- |
| AirPods 整体成功（表 1） | 88% | **92%** | 25 次，成功需全流程完成 |
| AirPods 整体成功（图 4 重复评测） | 86% | 90% | 50 次 / 单元格，不同评估协议 |
| 蛋料理 rubric（图 4） | 约 85.8% | 未报告 | 14 阶段，最多 42 分 |
| 打乱物件中途调整抓取后的恢复率 | 70% | 88% | 50 次扰动试验 |

作者的消融显示：移除一致性约束后整体成功由 92% 降到 80%；移除自适应梯度长度后降到 76%。只给最终奖励、不做阶段进度/切换奖励时，累计覆盖到最后阶段约为 60%。这证明**奖励质量与可解释的目标投影是整套训练成败的基础**，不能只把“公式 (6)”从论文中复制进模型就宣称复现。

## 7. 可运行 CPU 最小验证：先证明数学契约，再谈模型训练

本站附：<code>examples/flowdpg_cpu_contract.py</code>，只依赖 Python 标准库。运行：

<pre>python3 examples/flowdpg_cpu_contract.py</pre>

逐条断言：线性 Flow 下的一步投影等于真值动作；一致性损失与 $(1-t)^2$ 加权速度误差同值；简单二次 Q 的梯度局部提高 Q；自适应 $\lVert\Delta\rVert$ 符合比例上限；零梯度时安全归零；expectile 正负误差权重不对称；SARM 奖励包含阶段变化；$\lambda=0/1$ 分别回到 BC/蒸馏目标。

**这些测试只检验论文公式的数值逻辑**。它们不证明 SARM 标签正确、Critic 训练稳定、收敛有效，更没有使用作者权重或 GPU 仿真。没有作者的训练代码时，不能凭合成测试宣称“官方复现完成”。

### 7.1 真正想复现，需要依次补什么？

- **阶段 A：可观察的 base policy。** 固定任务（比如双臂盒体装配）、统一 3/4 路相机、动作块定义、初始 BC/Flow 监督数据，先测成功率与单次失败位置。
- **阶段 B：独立评测 Reward。** 明确阶段定义、实测插接判据、相机遮挡情况，SARM 置信度与人工抽检；不要对未来测试标签泄漏训练。
- **阶段 C：单独训练 IQL。** 先检查 held-out 动作局部梯度与实际进度变化 Spearman 相关性，再放行 actor RL；否则梯度可能朝错误方向。
- **阶段 D：只在仿真做 FlowDPG。** 与纯 BC / AWR / LoRA-SFT / 固定 actor + Q 后处理做相同任务、相同数据预算和相同 seed 的 A/B；画 $Q$ 偏差、动作步长、成功率与耗时曲线。
- **阶段 E：真实机器人。** 限力、速度、工作空间、安全停机、operator 接管与旧权重可回滚；先低速、再逐渐扩大场景范围，不以单次 92% 为验收标准。

### 7.2 与 π₀.₅ 的迁移实验：这是**研究方案，不是论文已实现功能**

首选对照：先固定 π₀.₅ 的视觉骨干、语言前缀与原 action mask，LoRA 微调获得稳定 BC，再把小型 $Q,V$ 接到**归一化且 mask 后的 action chunk**。第一阶段仅训练 Critic 和值函数，第二阶段在 action expert 的 adapter/LoRA 上尝试 $u_t^*$ 蒸馏，保持标准 Flow Loss 锚定。每一组记录 reward 误判、梯度分歧、真实 success、接管率与非训练任务遗忘。

**注意**：EGO/UMI 人类演示可给前期预训练提供视觉与动作先验，但标准 Critic 的 $Q(s,a)$ 需要与**真实机器人动作和反馈**语义一致；不能直接拿未映射到机械臂动作空间的人手位姿来估机器人执行价值。H2R 重定向与 Replay 质量先过门槛，否则学到的是数据配准误差。

## 8. 风险边界、应用价值与复现可信度

| 风险 | 原因 | 实施对策 |
| --- | --- | --- |
| 错误阶段奖励 | SARM 对遮挡、反光、任务中断判错 | 阶段审计、奖励模型与物理接触信号交叉校验 |
| Critic OOD 梯度 | Q 在未访问动作上可能任意外推 | $\hat a$ 一致性约束、BC 锚、梯度限幅、离线相关性检查 |
| 随机动作块不连续 | 换噪声可能突然跳到另一轨迹模式 | 区分实时 chunk 切换与 flow 求解步；部署端单独做平滑/接管逻辑 |
| 动作量纲耦合 | 关节角、夹爪、末端平移在统一范数内尺度不同 | 用清晰的 action normalization、mask、每组维度物理限制 |
| 线上训练误伤 | 错误改进可能放大接触风险 | 先仿真、双通道 shadow eval、严格回滚及安全阈值 |
| 视觉表征漂移 | 小数据在线更新破坏原有视觉能力 | 冻结 backbone、监测未训练任务遗忘、周期性留出评估 |

**论文已经证明的范围**：作者在指定 Franka AirPods 装配和 YAM 蛋料理两组长流程真机任务上，报告 FlowDPG 取得较高成功率或 rubric；Critic 梯度蒸馏作为一种合理训练路径得到消融支持。**论文尚未证明的范围**：所有具身大模型、全部工业双臂工位、无阶段标注任务、零真机反馈场景都能获得同幅度提升。真实落地至少要补 Reward 的可靠性和连续动作的安全约束。

## 9. 原始来源和复现资产清单

- **[论文：FlowDPG，CoRL 2026（PDF）](https://flowdpg.github.io/static/images/main.pdf)**：§3.1 IQL、§3.2 式 (4)–(11)、§3.3 离线转在线；§4 真机结果；附录 A 梯度推导、附录 B Algorithm 1。
- **[作者项目页](https://flowdpg.github.io/)**：AirPods 25 次连续完整装配录像、蛋料理、失败恢复与评测 rubrics；目前未见代码下载按钮。
- **[arXiv:2606.22303](https://arxiv.org/abs/2606.22303)**：核查论文标题与更新记录；以作者项目页 PDF 最新实验版本为本文结果口径，勿混用旧摘要。
- **[作者项目站点仓库](https://github.com/FlowDPG/flowdpg.github.io)**：已核实包含网站构建源码、论文 PDF 和视频资源，**不是 FlowDPG 训练代码发布**。
- **本站 CPU 合成契约**：<code>examples/flowdpg_cpu_contract.py</code>，验证最小数学行为，下一步仍需真实动作策略、Critic、SARM 与 rollout。

**一句话复盘**：FlowDPG 不靠让机器人“看懂阶段文字”直接提升动作，而是通过阶段奖励先学出有一定可信度的价值函数，再把动作空间的价值梯度压成无需 ODE 反传的 Flow 速度监督。可落地性取决于奖励、Critic、动作契约和真机闭环是否同时成立。