---
title: "RTC 深度教程：π₀.₅ 异步动作块、软掩码引导与训练期前缀条件化"
description: "逐公式理解 Real-Time Chunking 与 Training-Time RTC，核查 LeRobot 源码与训练/部署参数，附双图、时序推导及 CPU 可执行契约"
sidebar:
  order: 4
---

> **核查状态（2026-11-10）**：主依据是 Physical Intelligence 两篇一手论文：[Real-Time Execution of Action Chunking Flow Policies，arXiv:2506.07339v2](https://arxiv.org/html/2506.07339v2)（初版 2025-06-09，v2 2025-12-05）与 [Training-Time Action Conditioning for Efficient Real-Time Chunking，arXiv:2512.05964v2](https://arxiv.org/html/2512.05964)（2025-12-09 v2）；官方[研究解读和实验视频](https://www.pi.website/research/real_time_chunking)。代码核查基线为 Hugging Face [LeRobot commit 81b8c5bf](https://github.com/huggingface/lerobot/tree/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee)。原作者已公开 [Kinetix RTC 代码](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)，与 LeRobot 的通用机器人实现不是同一个仓库。本文脚本是**本站独立 CPU 契约测试**，不是官方模型推理或真机精度复现。

# RTC：π₀.₅ 明明每秒生成一段动作，为什么机器人还会卡顿？

## 0. 本章定位：不是把 FlashVLA 重新讲一遍

在 [π₀.₅ 数据、Flow Loss 和 LoRA](../vla/pi05-libero-data-loss-lora-source-guide/) 已建立训练端动作语义之后，机器人执行端还有一个难题：**控制器每 20 ms 要拿到一个动作，VLA 每次推理却耗时约 100–300 ms**。如果一段长为 50 的 chunk 做完才启动新推理，机械臂就会停顿；如果一边执行旧 chunk 一边异步推理，新旧 chunk 又可能采取不同的路径，产生不连续的大幅加速度。

RTC（Real-Time Chunking）解决**跨 chunk 的实时连续性**。它与 [FlashVLA](./flashvla-streaming-flow-action-decoding/) 的区别：FlashVLA 在**模型内部**把去噪过程流水化，RTC 在**模型输出与真实时钟之间**规定哪些动作已承诺、哪些可修正，并指导新 chunk 与旧计划兼容。两者不能简单等价；[PACT-VLA](./pact-vla-precommit-flowres/) 研究提交前观测与残差修正，问题分解又不同。

![RTC：低层控制器、异步推理、动作前缀约束和安全闸门](/Embodied-AI-Roadmap/diagrams/rtc-asynchronous-system.svg)

*图 1｜本站根据论文重绘的时序与部署分层。蓝色为执行中已承诺的动作，橙色是被历史轨迹约束的重叠窗口，绿色为允许新观测修改的未来动作。图中安全闸门为本文工业部署建议，**非** RTC 论文原有模块。*

## 1. 时间语义与四个容易混淆的 Horizon

设模型在控制时刻 $t$ 收到图像、语言、本体状态 $o_t$，输出未来 $H$ 个动作：

$$
\mathbf A_t=[a_{t|t},a_{t+1|t},\ldots,a_{t+H-1|t}]
\in\mathbb R^{H\times D_a}.
$$

其中 $D_a$ 可以是单臂关节位置、双臂关节位置+夹爪，或经过规范化的末端目标；**RTC 不改变动作物理含义**。设控制周期 $\Delta t_c$，模型从观测到完整输出耗时 $\delta$，以控制步为单位的延迟为：

$$
d=\left\lfloor\frac{\delta}{\Delta t_c}\right\rfloor
\quad\text{（论文的离散化；部署宜保守向上预留裕度）}.
$$

还要区分 $s$：每次新请求之间已经执行的低层步数（论文称 execution horizon）。若当前 chunk 总长 $H$，推理启动时已经消费 $s$ 步，则旧 chunk 还剩 $H-s$ 步，接下来推理期间需要旧计划支撑 $d$ 步。满足无停顿的必要预算约束：

$$
0\le d\le s\le H-d,\qquad d\le H-s.
$$

第一部分 $s\ge d$ 源于论文所采用的请求节奏/实际执行设置，不应被当成所有任意异步控制器都不可突破的物理定律。核心覆盖关系是 $d\le H-s$：旧队列必须撑到新推理返回。

**手算例子**：控制频率 50 Hz，$\Delta t_c=20$ ms，$H=50$，新推理 140 ms $\Rightarrow d=7$。假设发起请求时旧 chunk 已消费 $s=15$ 步，剩余 $H-s=35$ 步，覆盖 7 步延迟绰绰有余。7 步已经发生的动作**不能**让新 VLA 再执行一次。新 chunk 索引 0…6 用作条件信息；真正上机的是新 chunk 的索引 7 之后。

| 符号 | 真实单位 | 例子 | 最常见错误 |
| --- | --- | --- | --- |
| $H$ | 动作采样点 | 50 steps | 误写成模型 Flow 积分次数 |
| $s$ | 控制步 | 15 steps | 误当成模型处理器里的所有参数都叫 execution_horizon |
| $d$ | 控制步 | 7 steps | 推理结束后不丢弃过期的前 7 步 |
| $N$ | 去噪迭代 | 10 steps | 和 $H=50$ 搞混 |
| $\Delta t_c$ | 秒/控制步 | 0.020 s | 忽略相机曝光与遥程通信 |
| $\delta$ | 秒/请求 | 0.140 s | 只算模型 GPU 前向，不算排队/网络/解码 |

对从 UMI 转换过来的 Replay 数据更要确认末端动作究竟是 $T_{\mathrm{base}\leftarrow\mathrm{tcp}}$ 的目标位姿、关节目标，还是速度命令。不能在 RTC 里对 quaternion 四元数直接做普通欧氏平均；对关节角、角速度和末端夹爪的限幅也必须分开。

## 2. 旧 chunk 如何分成三种不同的“承诺”？

作者把新 chunk 的索引 $i=0,\ldots,H-1$ 划成三段。

- $0\le i<d$：推理期间已经执行的历史动作。它们不应被新计划改变，**硬约束窗口**。
- $d\le i<H-s$：旧计划仍有覆盖，但执行尚未发生，可根据新观测调整，**软一致性窗口**。
- $H-s\le i<H$：旧 chunk 根本没有对应动作，必须由模型重新生成，**开放未来窗口**。

论文 [3.2 节式（5）](https://arxiv.org/html/2506.07339v2) 的软权重（注意这是**引导权重**，不是 Transformer Self-Attention 的 Attention Mask）为：

$$
W_i=
\begin{cases}
1,&i<d,\\
c_i\,\dfrac{\exp(c_i)-1}{e-1},&d\le i<H-s,\\
0,&i\ge H-s,
\end{cases}
\qquad
c_i=\dfrac{H-s-i}{H-s-d+1}.
$$

上面第二段对临近未来的权重高，对远期逐渐趋零。以教学设置 $H=12,s=4,d=3$ 为例，三区间分别是 $[0,3)$、$[3,8)$、$[8,12)$。当 $i=3$，$c=5/6$，得到 $W_3\approx0.635$；$i=7$，$c=1/6$，得到 $W_7\approx0.0176$。**看起来“平滑”的平均动作不一定在有效策略流形上**：例如上一 chunk 从左侧绕障碍，下一 chunk 从右侧绕障碍，直接对关节目标平均，手臂有可能撞向中间的障碍物。RTC 用生成模型的条件分布来保持连续性，而不是上机前硬平均两条路径。

![RTC：软权重、时间对齐、训练期无噪声前缀及损失掩码](/Embodied-AI-Roadmap/diagrams/rtc-prefix-mask-detail.svg)

*图 2｜上排展示三段时间权重，下排展示训练时“前缀位置输入干净、后缀位置才承担损失”。与原作者图片不同，细节图完全为本站重绘。*

## 3. 推理期 RTC：Flow Matching 上怎样实现“有条件重绘”？

### 3.1 先统一不同实现的流时间方向

原始 RTC 论文把**噪声**放在流时间 $\tau=0$，**数据**在 $\tau=1$，预测速度由噪声流向数据：

$$
x_\tau=(1-\tau)\epsilon+\tau A,\quad
u_\tau=\frac{dx_\tau}{d\tau}=A-\epsilon.
$$

顺向 Euler 积分是：

$$
x_{\tau+1/N}=x_\tau+\frac1N v_\theta(x_\tau,o,\tau).
$$

而 [openpi π₀.₅ 源码核验章节](../vla/pi05-libero-data-loss-lora-source-guide/) 使用 **数据在 0、噪声在 1** 的时间约定，逆向积分写作 $x_{t-\Delta}=x_t-\Delta v_\theta$。这是时间重参数化，不能复制一个库的指导项符号给另一个库。

在论文的顺时方向，当前去噪点对最终数据的一个显式一步估计是：

$$
\widehat A_1(x_\tau)=x_\tau+(1-\tau)v_\theta(x_\tau,o,\tau).
$$

对已知旧动作目标 $Y$ 定义带权匹配误差；令 $J=\partial\widehat A_1/\partial x_\tau$，则引导方向为：

$$
g=J^\top\operatorname{diag}(W)(Y-\widehat A_1).
$$

它是 **vector-Jacobian product（VJP）**：无需显式展开一个 $(HD_a)\times(HD_a)$ 的大矩阵，用一次反向自动微分完成。但这也使每次 Flow 迭代多出计算开销。

作者式（2）给出的引导速度（省略参数的条件记号）是：

$$
v_{\mathrm{RTC}}=v_\theta+
\min\!\left(\beta,\frac{1-\tau}{\tau r_\tau^2}\right)g,
\qquad
r_\tau^2=\frac{(1-\tau)^2}{\tau^2+(1-\tau)^2}.
$$

$\beta$ 是裁剪上限：没有它，少步数去噪时的指导增益可能导致不稳定。特别注意端点 $\tau=0,1$ 的数值处理；论文和实现均采取稳定化策略，不能直接把公式在端点作为普通浮点除法求值。

### 3.2 与 LeRobot 代码逐项对照

下表对应 [固定版本 LeRobot modeling_rtc.py](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/src/lerobot/policies/rtc/modeling_rtc.py)。源码经过了**反向 Flow 时间参数**的转换：

| 论文符号 / 概念 | LeRobot 对应行为 | 重要差别 |
| --- | --- | --- |
| $x_\tau$ | <code>RTCProcessor.denoise_step(x_t,...)</code> | 使用底层 PI0/PI05 的逆时间 |
| $\widehat A_1$ | <code>x1_t = x_t - time * v_t</code> | 与论文的正时间 $x+(1-\tau)v$ 不是同一行算式 |
| $W$ | <code>get_prefix_weights(...)</code> | 支持 LINEAR、EXP、ONES、ZEROS 四种 schedule |
| $g$ | <code>torch.autograd.grad(x1_t,x_t,grad_outputs=err)</code> | 在输出轨迹空间做 VJP |
| $\beta$ | <code>max_guidance_weight</code> | 对引导增益限幅 |
| 无条件初始 chunk | <code>prev_chunk_left_over is None</code> 直接返回原始速度 | 第一段没有历史可条件化 |

**代码审查容易踩的坑**：<code>RTCConfig</code> 固定版本里 <code>prefix_attention_schedule</code> 的 dataclass 默认值是 <code>LINEAR</code>，不是“所有实现默认 EXP”；官方 RTC 教程建议 EXP 入门，但必须显式指定。反向 Euler 与 VJP 的正负号也不能只看一行局部代码就下结论。

## 4. 真机线程和 ActionQueue：哪些动作允许上机？

部署中**至少有两条时钟**：推理线程什么时候取得图像、什么时候返回预测；执行线程每个控制 tick 何时发命令。下列文字是接口级**教学伪代码**，非可直接控制设备的完整程序：

~~~python
# 控制线程：时间必须单调，每个 tick 都有一个可追溯动作
while control_enabled:
    packet = queue.get_with_provenance()  # 动作 + 来源任务/请求
    if packet is None:
        safety_stop("queue underrun")     # 不允许盲目重复旧动作
    action, task_id, provenance = packet
    assert provenance.observation_ts <= monotonic_now()
    checked = safety_limits(action, joint_limits, velocity_limits, workspace)
    robot.send(checked)
    sleep_until_next_tick()

# 推理线程：读取尚未执行的旧尾部，结合新观测引导生成
obs = snapshot_observation_and_timestamp()
old_tail = queue.get_left_over()
pred = policy.predict_action_chunk(
    obs, inference_delay=estimated_delay_steps,
    prev_chunk_left_over=old_tail
)
real_delay = measured_latency_steps()
# 基于真实已过步数跳过过时前缀；若观测过期/队列耗尽则拒绝
queue.merge(pred, pred, real_delay, provenance=obs.meta)
~~~

在 [LeRobot action_queue.py](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/src/lerobot/policies/rtc/action_queue.py) 中，RTC 模式的 <code>_replace_actions_queue</code> 会丢弃新 chunk 的前 <code>real_delay</code> 步，线程锁保护替换；<code>QueueSnapshot</code> 还用 generation/cursor 管理替换一致性，并带来源 metadata。即便如此，**论文证明不是安全认证**：感知延迟、网络延迟、机器人驱动器队列和真实执行时间都要单独测量；特别不能凭离线数组切片就宣称机器人 50 Hz 端到端稳定。

### 4.1 队列和状态的最小合同

| 关键字段 | 形状 / 单位 | 允许值 / 验证 |
| --- | --- | --- |
| <code>observation_ts</code> | 单调时钟 s | 不晚于预测发起时刻 |
| <code>request_id</code> | UUID/整数 | 唯一、不得串跨任务 |
| <code>action_chunk</code> | $[H,D_a]$ | 有限数、类型/单位和 Robot API 一致 |
| <code>committed_index</code> | int | 单调递增，不能重复执行同一步 |
| <code>latency_steps</code> | int | 实测而非固定假设 |
| <code>task_id</code> | string | 队列更换必须锁步匹配 |
| <code>norm_stats_id</code> | 版本号 | 训练与推理同一归一化版本 |
| <code>action_frame</code> | base/world/ee | 位姿必须与机器人控制器一致 |

工业双臂操作建议为**双手输出同一时基**，分别限制左、右臂速度、加速度、距离夹具和自碰撞。在夹具接触、衬套插接等高刚度场景，RTC 的连续性不能替代力控/阻抗控制与急停。

## 5. Training-Time RTC：将“推理期修图”变成模型自带条件生成

2025-12 作者发布的[后续论文](https://arxiv.org/html/2512.05964)指出：推理期 RTC 每步 VJP 有额外时间成本，且大延迟下线性化近似更不稳定。因此在训练时随机模拟延迟 $d_b$，让模型学习**知道先前若干步已经被承诺时，如何只生成后续动作**。

设批量动作为 $A\in\mathbb R^{B\times H\times D_a}$，每个样本独立采样 $d_b$。对第 $b$ 个样本的第 $i$ 帧，定义：

$$
M_{bi}=\mathbb 1[i<d_b],\quad
x_{bi}^{\mathrm{train}}=
\begin{cases}
A_{bi},& M_{bi}=1,\\
(1-\tau_b)A_{bi}+\tau_b\epsilon_{bi},&M_{bi}=0.
\end{cases}
$$

这里使用 **LeRobot / openpi 的 noise-at-one 约定**，$x_\tau=(1-\tau)A+\tau\epsilon$，速度真值是 $\epsilon-A$。如果对训练前缀的流时间单独设为**数据端 $\tau=0$**，模型可识别该 token 是干净而且承诺的；其余后缀使用随机 $\tau_b$。在[后续论文原文](https://arxiv.org/html/2512.05964)中采用另一种时间定义，并把**干净前缀的 flow 时间设为 1**。两个说法对应反向参数化，逻辑一致；禁止把它们混着用。

只在后缀算速度 MSE：

$$
\mathcal L_{\mathrm{postfix}}
=\frac{\displaystyle\sum_{b,i,j}(1-M_{bi})
\left[v_\theta(x^{\mathrm{train}},o,\tau)_{bij}
-(\epsilon_{bij}-A_{bij})\right]^2}
{\displaystyle D_a\sum_{b,i}(1-M_{bi})}.
$$

**为什么要重新计算分母？** 不同样本前缀长度不同，直接先 <code>mean</code> 后乘掩码，会让损失数值随 $d$ 改变；使用 $BHD_a$ 当分母则错误稀释长前缀样本。训练期 RTC 在输入前缀里放的是**当前训练样本的 ground-truth action**，但推理时放的是**当前执行队列的旧预测/承诺动作**：这存在 teacher-forcing gap，应在失配/扰动实验中评估，不能把 GT 前缀直接偷渡到部署中。

### 5.1 源码实现核查（固定 LeRobot commit）

- [configuration_pi05.py](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/src/lerobot/policies/pi05/configuration_pi05.py) 定义 <code>rtc_training_max_delay: int = 0</code>（0 代表关闭）；同时 <code>chunk_size=50</code> 是此配置默认，不是所有模型固定值。
- [modeling_pi05.py 的 _sample_training_rtc_prefix_mask](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/src/lerobot/policies/pi05/modeling_pi05.py#L131-L161) 使用每样本 <code>torch.randint(0,max_delay+1)</code>，并用 <code>postfix_mask</code> 重新归一化 loss。
- <code>forward</code> 中 <code>make_flow_matching_inputs(..., convention=FlowConvention.NOISE_AT_ONE, prefix_mask=...)</code> 建立干净前缀和带噪后缀，随后逐 token 的时间条件进入 Action Expert。
- <code>sample_actions</code> 的 <code>rtc_mode="trained"</code> 调用 <code>_prepare_trained_rtc_prefix</code>，后续 Euler 走 hard prefix，而 **guided 模式**才调用 <code>RTCProcessor</code> 的 VJP 引导。
- [configuration_rtc.py](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/src/lerobot/policies/rtc/configuration_rtc.py) 对 trained checkpoint 强制 <code>rtc_training_max_delay &lt;= execution_horizon &lt;= prediction_steps - rtc_training_max_delay</code>；普通 guided 模式不受这项训练专属约束。

因此不能把一个**普通 π₀.₅ checkpoint**的推理模式直接指定为 <code>trained</code> 就声称获得无 VJP 的 RTC。模型没有看过训练前缀时，必须先采用 guided 或重新微调得到兼容 checkpoint。

## 6. 官方 LeRobot 复现路径：分三个层级，不得混称“复现成功”

### 6.1 A｜CPU 独立契约（本仓库可跑）

本仓库新增 [examples/rtc_prefix_contract_cpu.py](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/rtc_prefix_contract_cpu.py)，只依赖 Python 标准库：

~~~bash
python3 examples/rtc_prefix_contract_cpu.py
~~~

它检查以下可证伪性质：$d\le H-s$ 预算约束、论文 EXP 软掩码数值、干净前缀与后缀 loss 分母、真实延迟丢弃、无效重叠拒绝、数据端/噪声端时间符号反向时的 Euler 方向。脚本**没有 PyTorch GPU、没有 VLA 权重**，不会报告虚构的成功率。

### 6.2 B｜GPU 离线数据测试

按 [LeRobot RTC 官方教程](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/docs/source/rtc.mdx)，在**锁定版本的环境**中运行：

~~~bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && git checkout 81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee
pip install -e ".[pi]"
python examples/rtc/eval_dataset.py \
  --policy.path=lerobot/pi0_libero_finetuned \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --rtc.execution_horizon=10 \
  --rtc.max_guidance_weight=10.0 \
  --device=cuda
~~~

这是官方文档给的**离线可视化入口**，需先核实该 commit 下路径、模型权重权限、版本和设备；本次尚未在本仓库执行，不能将代码块出现当作“跑通”。

### 6.3 C｜π₀.₅ 训练期 RTC 专项微调

[LeRobot pi05 文档](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/docs/source/pi05.mdx) 提供训练期参数：

~~~bash
lerobot-train \
  --dataset.repo_id=YOUR_DATASET \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.rtc_training_max_delay=10 \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=30000 \
  --output_dir=outputs/pi05_rtc \
  --job_name=pi05_rtc
~~~

**此处是文档示例，不是本站在四张 RTX 6000 Ada 上测得的可行 batch**。训练前做一次 <code>--help</code> 与单 batch smoke test，并确认 <code>max_delay&lt;chunk_size</code>。推理使用 <code>--inference.type=rtc --inference.rtc.mode=trained</code>，而且需让 <code>execution_horizon</code> 落在训练版本的接受范围内。对于 $H=50$、最大训练延迟 $D=10$，保守条件给出 $10\le s_{\mathrm{config}}\le40$；**这里 LeRobot 配置中的 execution_horizon 是“历史前缀容量”，不能不加解释就等同于上文论文的实际连续播放步数 $s$**。

## 7. 关键消融：比较的不只是每秒多少 token

建议对同一任务、同一模型权重与同一网络配置逐级评估：

| 策略 | 是否改训练 | 是否额外 VJP | 关键监控 |
| --- | --- | --- | --- |
| 同步 chunk 执行 | 否 | 否 | 等待占比、成功率、耗时 |
| 朴素异步 | 否 | 否 | chunk 交界 jerk、接触失败 |
| Temporal Ensembling | 否 | 否 | 平滑但偏离有效轨迹的比例 |
| Guided RTC（inference-time） | 否 | 是 | 推理延迟/尾部连续性/成功率 |
| Trained RTC | 是 | 否 | 训练延迟分布覆盖、拒绝率/后缀质量 |

需要同时记录 **任务成功率、接管率、执行时间、因等待丢步的比例、接触前最大速度/加速度/jerk、每次推理耗时 P50/P95/P99、过期 chunk 丢弃率**。只展示平均推理速度而不展示失败率，可能误导。

来源边界：作者最初 [RTC 研究页](https://www.pi.website/research/real_time_chunking) 的 50Hz 双臂精密任务和吞吐率，是 Physical Intelligence 的报告；后续 [训练期 RTC 论文第 V-B 节](https://arxiv.org/html/2512.05964) 报告 π₀.₆ 50Hz 机器人上，训练期 RTC 平均端到端延迟 108ms、推理期 RTC 135ms，且前者在 box building 和 espresso making 上性能/速度相近。这是**不同 checkpoint 与引导机制的作者评估**，不是“π₀.₅ 改一个参数就必然快 27ms”的证据。论文也指出训练期 RTC 大延迟表现更好，但会损失软权重的灵活性。

### 7.1 在工业双臂上选择第一批实验

针对线束插接、衬套压装和双臂协同搬运：第一批选择**可回放、低速、明确可判定成功的桌面任务**，先跑同期基线与 naive async，观察是否真的存在 chunk 交界抖动，再将 guided RTC 接到只读影子模式，最后通过位置/速度/加速度限幅与碰撞检测后逐步放开闭环。

**禁止把重放的“成功”写成真机成功**：Replay 只证明数据与动作接口的若干环节可对齐，接触力、时延、状态反馈和错误恢复需要独立验证。

## 8. 五个常见错误及学习练习

**误区一**：把 RTC 理解为简单拼接旧 chunk 剩余动作。纠错：它改变的是新计划的生成条件，强调连续性与可响应性；简单拼接只能推迟切换。

**误区二**：认为前缀约束越长越安全。纠错：在接触突变或移动物体场景，过长约束会让策略反应过慢；软权重与窗口选择是控制超参数。

**误区三**：说“训练期 RTC 不改模型就一定无需训练”。纠错：它不一定增加参数结构，却必须**改变训练时的数据/时间条件及 loss mask**，得到新 checkpoint。

**误区四**：以训练末端的真实动作来指导线上生成。纠错：线上只有截至当前时刻的观测和旧策略承诺轨迹；未来 GT 只能在离线训练存在。

**误区五**：把两种 Flow 时间方向的梯度引导公式直接粘贴。纠错：第一步应手算 $x_\tau$ 的导数和 Euler 方向，并确认 <code>modeling_rtc.py</code> 的 <code>tau=1-time</code>。

**练习 A（时序）**：在 $H=20,s=12,\delta=180$ms、50Hz 时，新 chunk 能否在旧队列耗尽前返回？**参考**：$d=\lfloor180/20\rfloor=9$，$H-s=8$，预算不够；在真实系统更应保守考虑时延抖动。

**练习 B（掩码）**：两个样本的前缀分别长 0 与 3，$H=5,D_a=2$，逐元素平方误差均为 4。正确平均 loss 是多少？**参考**：后缀有效元素 $(5+2)\times2=14$，分子 $14\times4=56$，答案仍为 4；不能先做零填充平均得到 2.8。

**练习 C（工程）**：若上一段计划从物体左侧绕行，新 chunk 从右侧绕行，为什么普通加权平均不是等效于轨迹优化？**参考**：凸组合在关节空间未必保留碰撞安全、IK 可行性和接触模式，必须在约束模型或策略条件分布中处理。

## 9. 来源与复现边界

一手论文：[RTC v2](https://arxiv.org/html/2506.07339v2)、[Training-Time RTC v2](https://arxiv.org/html/2512.05964)、[Physical Intelligence 官方示范页](https://www.pi.website/research/real_time_chunking)。官方实现对照：[LeRobot RTC](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/src/lerobot/policies/rtc/modeling_rtc.py)、[LeRobot PI05](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/src/lerobot/policies/pi05/modeling_pi05.py)、[LeRobot RTC 文档](https://github.com/huggingface/lerobot/blob/81b8c5bfdd5591ec59e512cfc7ac6db8151c70ee/docs/source/rtc.mdx)。所有来源以本文发布日期和固定代码 SHA 为边界，可能与未来 <code>main</code> 不同。

**本篇交付**：两张原创 SVG、推理/训练两种 RTC 的公式与代码路径、CPU 时间掩码验证和工业化验收清单。**未做**：GPU VLA 微调、Kinetix/LIBERO 闭环成功率、双臂真机验证。CPU 数学测试的 PASS 不代表后面三项已完成。
