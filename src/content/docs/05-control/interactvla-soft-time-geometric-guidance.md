---
title: "InteractVLA：冻结策略的几何引导、软时间匹配与可审计失败恢复"
description: "从 Flow Matching 输入梯度到动作空间适配、状态版本协议与纠正数据回流，核对 2026-11-17 论文和公开 PyTorch/JAX 工具接口"
sidebar:
  order: 8
---

> 核查日期：2026-11-24。原始来源：[论文 v1](https://arxiv.org/html/2611.22584v1)、[官方仓库](https://github.com/LingwuPro/InteractVLA)、[项目演示](https://lingwupro.github.io/InteractVLA/)。本文是独立教学推导和源码审计；文中 CPU 示例是合成合同测试，并非官方模型复现。所有任务成绩来自作者。

## 1. 为什么仅靠语言重试不够

VLA 的动作块是条件概率分布的一次采样。失败后原样重复指令，可能仍落在同一高概率错误模式。换一个随机种子也不能保证修正空间偏差。一个更直接的接口是由高层视觉语言智能体给出目标位姿，再在低层动作生成过程中引入可微几何约束；生成结束后先检查轨迹，再决定是否执行。

与 [RTC](/Embodied-AI-Roadmap/05-control/rtc-inference-training-time-action-conditioning/) 的旧动作前缀一致性不同，本方法约束的是空间交互目标；与 [FlowDPG](/Embodied-AI-Roadmap/01-foundations/vla/flowdpg-bptt-free-critic-gradient-posttraining/) 的参数优化不同，本方法冻结策略权重，仅在测试时修改生成状态。可独立评测，也可将成功纠正记录供后训练使用。

![几何目标、动作生成、反馈与回流闭环](/Embodied-AI-Roadmap/diagrams/interactvla-system.svg)

图1：本站原创。虚线是候选生成及视觉反馈，不表示已执行；蓝色是模型路径，绿色是实际状态，橙色是显式几何约束。

## 2. 输入输出及三种时间

观测包含图像、语言、当前状态及标定信息；目标包含末端位置、方向、夹爪开度及其有效分量。原生动作张量为 $x_t\in\mathbb R^{B\times H\times D}$。B 是批次，H 是预测步数，D 是 checkpoint 原生动作维数，不是末端位姿维数。一个双臂目标也不是两次独立生成。

| 时间量 | 含义 | 不应混淆 |
| --- | --- | --- |
| t | Flow 生成时间，无量纲，噪声 0 到动作 1 | 不是执行秒数 |
| h | 动作块内执行索引 | 不是去噪迭代索引 |
| k | 新观测/决策周期 | 不是固定的控制步数 |

官方 README 的平台缺省值如下（不是通用推荐参数）：

| 环境 | H | 最多执行前缀 | 控制率 | 最长前缀物理时长 |
| --- | ---: | ---: | ---: | ---: |
| LIBERO | 10 | 10 | 20 Hz | 0.5 s |
| RoboDojo | 50 | 16 | 20 Hz | 0.8 s |
| RoboCasa365 | 50 | 16 | 20 Hz | 0.8 s |
| Dexmal | 32 | 8 | 10 Hz | 0.8 s |

预测出的完整动作块不应全部写入执行日志；只有控制器确认的前缀才属于已执行事实。软匹配的目标索引也可能位于前缀之外，此时不能宣称本周期已经到达目标。

## 3. Flow Matching：为什么需要输入梯度而非优化权重

采用论文的正向生成时间约定，令噪声为 $\epsilon$、干净动作为 $a$：

$$
x_t=(1-t)\epsilon+ta,\qquad v^*=a-\epsilon.
$$

冻结的动作专家给出 $v_\theta(x_t,t;c)$，其中 c 为观测和任务条件。线性路径下的一步终点估计是

$$
m_t=x_t+(1-t)v_\theta(x_t,t;c).
$$

注意：模型只近似速度场，估计的终点不等于多步积分后的真实动作。设可微解码器 C 将原生动作反归一化并映射成命令位姿，目标误差为 L，则链式法则为

$$
\nabla_{x_t}L=
[I+(1-t)J_v]^\top J_C^\top\nabla_{C(m_t)}L.
$$

这里的 $J_v$ 是当前 action expert 关于输入的 Jacobian，不必显式构造。PyTorch 的向量-雅可比积即可完成计算；若把 v detach 后再算几何梯度，就错误地丢掉了动作块内部的耦合项。参数冻结不等于整段 inference_mode；引导步仍需对输入启用自动微分。

在正向时间下，Euler 更新为

$$
x_{t+\Delta t}=\operatorname{sg}
[x_t+\Delta t(v_\theta-b_t)].
$$

b 是受限的几何纠正量，sg 将相邻迭代的图分离。因此只保留当前步的反传图，不通过整个 ODE 展开反传。相较于纯推理，会有额外激活显存、梯度计算和同步开销。

**接入 openpi 的符号陷阱**：[公开 π₀ PyTorch 实现](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py) 使用噪声时间 s，从 1 积分到 0。令 $s=1-t$，则

$$
v_{\rm gen}(x,t)=-v_{\rm noise}(x,1-t),\qquad
m_t=x-sv_{\rm noise}.
$$

正向式的减梯度不能直接复制到负步长积分里。必须明确转换速度方向、时间条件和步长；单独构造常速度 oracle，要求无引导时恢复原采样轨迹。对 π₀.₅ 权重也须按其原始 checkpoint 的时间接口实现，不应假定所有 flow 模型相同。

## 4. 从 native action 到几何误差：这是复现最容易出错的地方

官方 [geometry.py](https://github.com/LingwuPro/InteractVLA/blob/main/code/interactvla/geometry.py) 的 decode 将关节、绝对位姿或增量命令分别解析，不能把归一化动作的前三维直接视作米制 XYZ。

以逐维分位数归一化为例（教学公式，具体归一化类型以 manifest 为准）：

$$
a_d=q_{01,d}+\frac{x_d+1}{2}(q_{99,d}-q_{01,d}).
$$

进一步还需 controller 的 input/output 范围转换、限幅、有效维度以及基座外参。关节型用 FK；绝对型直接解码位置和方向；增量型则依赖当前状态。旋转向量与欧拉角不可混用，四元数必须核对 xyzw 顺序。

特别值得注意的是源码 delta 分支的跟踪近似：用估计内部位置 $\bar p_h$ 和旋转 $\bar R_h$，先输出命令位置，再按 tracking 系数更新内部状态。对于 body-frame 增量，

$$
d_h=\bar R_h\Delta p_h,\quad p_h^{cmd}=\bar p_h+d_h,
\quad\bar p_{h+1}=\bar p_h+\eta d_h.
$$

$$
R_h^{cmd}=\bar R_h\exp([\Delta r_h]_\times),\quad
\bar R_{h+1}=\bar R_h\exp([\eta\Delta r_h]_\times).
$$

world/base 分支将增量变换到统一参考系并左乘旋转。源码缺省 tracking 为 0.25；它是控制器跟踪的近似，不是测量真值。三步各 1 cm 的平移，初始为零、eta=0.25 时，命令位置为 1、1.25、1.5 cm，而不是累加成 1、2、3 cm。真实设备如采用不同跟踪行为，必须重新识别模型。

源码的 complete-pose 能量包含平移、旋转矩阵差和夹爪宽度的归一化误差。单个末端的教学展开：

$$
E_h=m_p\frac{\|p_h-p^*\|^2}{0.05^2}
+m_R\frac{\|R_h-R^*\|_F^2}{8\sin^2(\pi/12)}
+m_w\frac{\mathbb E[(w_h-w^*)^2]}{(w_{\max}-w_{\min})^2}.
$$

这里位置参考尺度为 5 cm，旋转分母对应 30 度的弦距离；这不是硬安全公差。双臂在同一个 h 上对有效分量求和，再进行时间匹配。对于二值夹爪，源码用宽度的一阶和二阶矩构造期望平方误差：

$$
\mathbb E[(w-w^*)^2]=\mathbb E[w^2]-2w^*\mathbb E[w]+(w^*)^2.
$$

只算均值差会漏掉方差。例如 0/1 等概率夹爪、目标 0.5，均值误差是零，但期望平方误差为 0.25。几何分量 mask 与原生 action mask 是不同层级：目标只约束某只手的某个分量，不意味着所有相关 action channel 都可以简单独立冻结。源码按有效末端选择运动及夹爪通道，再与有效维度相交。

## 5. Soft-time：时间匹配不是强迫最后一步到目标

如果每一步都被拉向抓取点，原有绕行和预接近轨迹会被压缩；只约束最后一步则强行假定到达时间。论文将到达索引视作隐变量，用同一个时间权重匹配完整位姿。

定义 $\tau=1-t$ 与 $\beta=t^2/\tau^2$：

$$
L_{\rm soft}=-\tau^2\log\left[
\frac1H\sum_{h=1}^H\exp(-\beta E_h)\right].
$$

对 $E_h$ 求导：$\partial L_{\rm soft}/\partial E_h=t^2q_h$，其中

$$
q_h=\frac{\exp(-\beta E_h)}{\sum_j\exp(-\beta E_j)}.
$$

论文再乘 $\tau/t$，最终得到

$$
L_{\rm guide}=\frac{\tau}{t}L_{\rm soft},
\qquad \nabla_xL_{\rm guide}
=t\tau\sum_h q_h\nabla_xE_h.
$$

因此低 t 时权重分散，较高 t 时更集中；同时 t(1-t) 在两个端点趋于零。实现必须用稳定 logsumexp，不在 t=0 或 1 处计算奇异表达式。[sampling.py](https://github.com/LingwuPro/InteractVLA/blob/main/code/interactvla/sampling.py) 的 temporal_objective 返回每个样本的损失及 [B,H] 权重，自动微分对 x 求梯度。

![合成误差下不同去噪时间的软匹配权重](/Embodied-AI-Roadmap/diagrams/interactvla-detail.svg)

图2：本站原创合成示例，不是论文实测。越晚的生成时间，权重越聚集在第3步；图中数据可用配套 CPU 脚本计算。

**反例：** 第1步的位置正确但方向错误；第2步方向正确但位置错误。若位置和方向各自取最小值，会错误报告目标已满足。共享 h 的完整位姿损失会拒绝这个虚假匹配。双臂协同也应在同一个 h 上聚合，不应给左右手各自独立取最优时刻后再声称协调成功。

## 6. 限幅和兼容性拒绝

源码 correction_and_compatibility 在有效的 H、D 维度上计算 RMS。令 G 和 V 分别为梯度和速度的 masked RMS：

$$
\alpha=\min(\lambda,\rho V/(G+\varepsilon)),\qquad
b=\alpha\,M\odot g.
$$

这里 lambda 是引导增益，rho 是相对速度上限。控制强度受 native velocity 约束；padding 通道不应稀释 RMS 或获得引导纠正。当 V 很小时，仍须避免数值爆炸，不能仅通过加大 lambda 解决。

拒绝依据是最佳时刻的残差 r 和限幅前纠正需求 d：

$$
d=\lambda G/(V+\varepsilon_v),\quad
C=\exp[-\max(r/r_0,d/d_0)].
$$

默认阈值和检查点由配置决定；论文以 exp(-1) 说明拒绝逻辑。特别区分 d 与实际限幅后的 b：先限幅再估计需求会把明显不兼容的目标伪装成小修正。C 衡量目标与当前策略生成的兼容程度，不是任务成功概率，也不是碰撞检测。

公开 README 缺省 10 次 Euler 评估、在 t=0.5、0.6、0.7 启用输入梯度。源码在配置的检查点上拒绝，批量运行时任一样本拒绝会返回 rejected；若需要独立批处理应保留逐样本状态而非直接假定原实现支持。拒绝候选可供诊断，不能直接执行。

以下伪代码展示结构（教学伪代码，不是官方接口）：

~~~python
for i in range(N):
    t = i / N
    active = is_guidance_step(t)
    x = x.detach().requires_grad_(active)
    v = expert(x, t, context)
    correction = zeros_like(x)
    if active:
        m = x + (1-t)*v
        energy = geometry_error(m, state, target)
        loss, q = temporal_objective(energy, t)
        g = autograd_grad(loss.sum(), x)
        correction, rejected = bounded_correction(g, v, energy, mask)
        if should_check(t) and rejected:
            return rejected_proposal()
    x = (x + (v-correction)/N).detach()
return proposal(x)
~~~

## 7. 反向接口：可检查的动作块和状态版本

官方 [runtime.py](https://github.com/LingwuPro/InteractVLA/blob/main/code/interactvla/runtime.py) 区分 observe、propose/guide、inspect、edit、execute、stop、memory。调用携带 schema version、唯一请求 ID；状态相关操作还需 observation ID 与 state version。

| 状态 | 允许 | 拒绝 |
| --- | --- | --- |
| 新观测 | 生成候选 | 使用旧标定、超龄观测 |
| 候选已接受 | inspect / 显式执行合法前缀 | 将未执行段写成事实 |
| 候选已拒绝 | 诊断、重新规划 | 直接 execute |
| 已执行 / 状态已改变 | 重新观测并生成 | 复用旧候选 |
| stop | 停止、刷新观测 | 自动续跑旧队列 |

源码 execute 前再次检查控制器 state version，执行后递增并记录每步 outcome，随后使旧 proposal 不可执行。重复 request ID 与相同内容可返回缓存；相同 ID 不同内容应拒绝。edit 针对剩余未执行轨迹，过时 continuation 需要显式 rebase。代码层面的这些检查不替代硬件急停、独立安全控制、碰撞与力限位。

相机反馈使用标定矩阵，将候选统一参考系的点转到相机再投影：

$$
p_C=R_{CW}p_W+t_{CW},\quad
u=f_xX/Z+c_x,\quad v=f_yY/Z+c_y.
$$

只有 Z 为正、内外参有效且时间对齐时，叠加图才有诊断意义。RGB-D 反投影需要深度单位及有效区域检查。图中候选线是 command pose，并非实测轨迹；应与实测关节/末端及物体状态分别显示。

## 8. 复现计划（从 CPU 到仿真）

第一阶段只运行本站脚本 examples/interactvla_guidance_contract_cpu.py，标准库即可。验证时间分布、解析梯度与有限差分、反向时间符号、tracking 近似、mask 限幅、版本拒绝。第二阶段核对官方环境：

~~~bash
git clone https://github.com/LingwuPro/InteractVLA.git
cd InteractVLA
git rev-parse HEAD
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest tests -q
~~~

上述命令来自官方 README 的安装测试流程；本站未运行其完整依赖。记录 git SHA、依赖版本及测试报告。第三阶段接入仿真 checkpoint，官方支持 StarVLA/PyTorch 和 OpenPI/JAX，后者还要独立安装与权重匹配的 OpenPI 环境；公开适配器不自带完整 checkpoint 与仿真资产。按 README 的 manifest 示例填写原生动作维度、归一化、状态定义、控制时间、外参和夹爪极性。

先不连接 executor，运行 proposal + inspect，固定 seed 比较 native 与 guided 的原生动作、几何误差、显存和时间。再连接仿真控制器，验证 stale proposal 拒绝、无引导等价性及物理 prefix；确认旧观测不驱动新状态。最后才在受控仿真中开启目标重规划及局部编辑。真实硬件部署需另行安全论证和独立约束。

## 9. 评测、工业价值与纠正数据回流

论文 LIBERO-Plus 的单轮在线工具使用表报告 agent+MR-0 从 93.1% 到 98.3%；RSI 多次尝试设置为 99.4%。两者不能互换。RoboDojo 单轮 Precision 子类反而由 9.7% 降到 8.3%，提示几何引导并非普遍获益。跨方法、不同 planner、不同任务子集及重试预算不可直接作无条件排名。Dexmal 的表是归一化任务分数，不是二元成功率。

建议对抓取高度偏差、过早闭合、失抓后三种问题进行成对随机种子实验。统一训练 checkpoint、仿真初始状态、总控制步、总推理调用与重试预算，对比：

| 组别 | 干预 | 需要观测 |
| --- | --- | --- |
| A | 原始 VLA | 单次成功率、重复失败率 |
| B | 高层仅修改语言 | 与 A 同预算 |
| C | 几何引导固定终点 | 目标残差、轨迹平滑 |
| D | soft-time + 拒绝 | 拒绝准确率、额外延迟 |
| E | D + 反馈重规划 | 恢复成功率、总时间、接管率 |

额外统计无效拒绝率、目标不可达比例、最小碰撞距离、动作上限违例和实际到达时间。记录成功与失败的 episode；不得将未执行候选或 rejected 轨迹当作正样本。建议日志结构包含 source_episode_id、observation_id、state_version、target、proposal_id、parent_id、native_action、decoded_pose、executed_prefix、measured_state、failure_reason、human_correction、success、calibration_version、model_sha。按原始 episode 做 split，以免同一次纠正的前后片段泄漏。

若目标是模型通过训练减少对智能体的依赖，应另行做行为克隆或强化后训练，比较真实执行的纠正数据与普通成功数据。这里的 RSI 是推理期尝试和经验利用，并不自动证明网络参数已改变。也不能仅因几何目标更精确就宣称工业接触、压装和柔性物体操作已解决；此类任务还依赖力、约束、柔顺控制和实际接触反馈。

## 10. 学习自测

1. 为什么在 x 上启用梯度不意味着训练 action expert？
2. 将正向生成式接到 openpi 的逆向时间时，哪两个符号必须转换？
3. 如果位置最小误差在第2步、旋转最小误差在第8步，能否判定已满足完整位姿？
4. 若一个候选已经执行部分前缀，为什么不能将剩余增量直接发给已变化的状态？
5. 为什么兼容性高不能替代物理安全检查？怎样记录数据避免自训练把失败当成成功？

参考核对入口：[sampling.py](https://github.com/LingwuPro/InteractVLA/blob/main/code/interactvla/sampling.py)、[geometry.py](https://github.com/LingwuPro/InteractVLA/blob/main/code/interactvla/geometry.py)、[runtime.py](https://github.com/LingwuPro/InteractVLA/blob/main/code/interactvla/runtime.py)。公式、合成例子及实验设计为本站教学推导，未进行官方 GPU 推理或真实成功率复测。
