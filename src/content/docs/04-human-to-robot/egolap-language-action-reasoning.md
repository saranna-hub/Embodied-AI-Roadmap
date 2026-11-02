---
title: "EgoLAP 深度教程：人类 EGO→语言动作→推理监督→VLA 连续控制"
description: "2026-10 EgoLAP 论文逐公式与附录核对：双专家张量、三段 attention mask、knowledge insulation、语言动作坐标归一、Reasoning Dropout、GRPO 和 EGO/UMI 最小复现实验"
sidebar:
  order: 9
---

> **核对时间：2026-11-02。来源等级**：【原文】arXiv [EgoLAP 2610.08726](https://arxiv.org/abs/2610.08726)（2026-10-06，含 Appendix A–D）；【作者项目】[ego-lap.github.io](https://ego-lap.github.io/)；【官方基线】[Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi)（注意并不是 EgoLAP 官方实现）；【本站替代 Demo】[egolap_contract_cpu_demo.py](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egolap_contract_cpu_demo.py)。**作者项目标注 Code / Checkpoints Coming soon。本篇未跑过 EgoLAP 原始预训练、GRPO 或真机闭环。** 下述固定超参数来自论文，工程改造建议及教学测试会明确标识。

# 为什么这篇值得和 Ego-Pi、WING 放在一起学？

最难的问题不是「人类视频有没有动作」，而是 **人的手、UMI 夹爪、机器人的手爪执行方式不同，哪些知识能够跨本体迁移？** EgoLAP（2026-10-06）给出第三条路线：不强行把人手轨迹逐点映射到机器人关节，也不只训练不可解释的视觉 latent，而是提取**用共同语言描述的一秒动作意图（What）**，再用**可解释当前场景几何、接触与可供性的局部物理理由（Why）**监督 VLM。

这两种文字仅在**训练**期间承担辅助损失。机器人推理仍靠连续 Action Expert 输出动作块；没有耗时的文本逐 token 推理。特别注意：作者人类训练数据**并非只有 RGB**，还需要在头部相机坐标系下追踪单/双手位姿，进而建立人类连续动作。缺少动作标签的纯视频不能直接套本文 FM 损失。

![EgoLAP 训练与推理双分支以及人机共享语言动作](/Embodied-AI-Roadmap/diagrams/egolap-system.svg)

*图 1｜本站按论文 3.1–3.4、Appendix A.1 重绘。训练时语言动作和推理文本仅监督 Backbone，FM 专家从图像/指令/状态前缀读取条件；推理不生成文字。图中 EGO 标定、质量门槛为工程扩展，不是论文额外模块。*

## 0. 四条路线的位置关系，不要混为一个方法

| 路线 | 人类侧监督 | 人与机器人如何对齐 | 真正监督动作专家的是什么 |
| --- | --- | --- | --- |
| [Ego-Pi](./ego-pi-deep/) | 人体双手动作、机器人动作 | 跨域视觉提示、双手交错 Action Token、联合微调 | 真实动作/人手动作对应的 FM 目标 |
| [Human-to-Robot Transfer](./human-robot-emergence-deep/) | 人类 3D 手位置与任务 | 依赖已有多样化机器人预训练实现跨域表征 | 各自域的动作监督 |
| [WING](./wing-ego-spectral-transfer/) | 纯 RGB 视频可参与 latent 学习 | 先分离相机运动，再蒸馏交互 latent 和谱特征 | 机器人侧仍需要 action label |
| **EgoLAP（本文）** | **人类 3D 手姿/开合** + 视频 | 同一类语言动作 + 物理运动理由 | **人类动作和机器人动作都进入 FM** |

**三者可互补，但不存在「EgoLAP 论文已在你的 EGO+UMI 硬件跑通」的证据。** EGO 原始 MCAP 只有 RGB/IMU 时，应先完成手部姿态估计与质量筛选；若存在可靠 UMI/TCP 轨迹，可作为更直接的几何监督。

## 1. 输入数据：三种监督不等于三个必需输入

论文定义机器人样本与人类样本：

$$
D_R:\;(o_t,\ell,s_t,a_{t:t+K}),\qquad
D_H:\;(o_t,\ell,q_{t:t+K}).
$$

其中 $o_t$ 是一个或多个 RGB 观察，$\ell$ 是指令，$s_t$ 是机器人状态，$a_{t:t+K}$ 是未来连续动作块；人类 $q$ 是头部相机系中单/双手跟踪位姿。人类当前姿态充当状态，手姿变化推导连续动作。动作标注、语言动作标注、可选的运动理由为**三个训练目标**，不是三种推理输入。

$$
m_t=G_e(q_{t:t+K}) \quad\text{或}\quad
m_t=G_e(a_{t:t+K},s_t),
$$

其中 $G_e$ 是根据单臂/双臂/人类本体元信息选择的**确定性**规范化器；没有训练一个新的动作离散化网络。语言动作 $m_t$ 描述未来约一秒的手/末端运动，$r_t$ 是可选局部 reasoning。$m_t$、$r_t$ 是监督标签，**不能在推理时当已知输入喂给动作专家**。

| 字段 | 论文实际张量约束 | 工业数据管线检查 |
| --- | --- | --- |
| RGB | $[B,V,224,224,3],V\le3$ | head/外部相机、左右腕部相机键与时钟 |
| 当前状态 | $[B,20]$，每臂 10 维：位置3、6D旋转6、开合1 | 旋转表示、坐标系和 gripper 离散语义 |
| 动作真值 | $[B,30,14]$，每臂 7 维 Cartesian delta | 30 步 / 一秒来源频率、左右手排列 |
| 文本输入 | $[B,320]$ 最大填充长度 | 指令 tokenizer、状态 bin、文字 target mask |
| 视觉 token | 每图 256 | 224×224 SigLIP patches；缺失视角 padding 与 mask |
| 速度场 | $[B,30,14]$ | 仅连续动作维度参与 FM |
| 数据来源 | robot/human/单臂/双臂 | 保存 provenance，禁止错误共享归一化统计 |

注意 **30 个动作时间步**和**Flow 反向积分的 10 次**是两个不同的轴。$D=14$ 不是五指手关节，而是两臂各 7 维 Cartesian 控制；Ego-Pi 的双手 58D 交错编码是另一种设计，不能照搬到这里。

## 2. 架构：为什么 2048 与 1024 的 Transformer 能在一层 Attention 里交互？

作者使用 Mixture-of-Transformers（MoT），**不是 top-k Router MoE**。VLM 采用 PaliGemma-2B 语言专家；动作专家从零初始化，约 300M 参数。论文 Appendix A.1 真实配置：

| 指标 | VLM 专家 | Action Expert |
| --- | ---: | ---: |
| Transformer 层数 | 18 | 18 |
| hidden width | 2048 | 1024 |
| FFN width | 16,384 | 4,096 |
| Query heads | 8 | 8 |
| KV heads | 1 | 1 |
| 视觉每路 token | SigLIP 256 | 从前缀间接读取 |
| 条件状态 | 0–255 bin 文本 | 读取前缀状态 |
| 专家输入 | $B\times L_p\times2048$ | $B\times30\times1024$ |
| 专家输出 | 语言 token logits | $B\times30\times14$ 流速度 |

完整三视角的前缀长度上界 $L_p=3\times256+320=1088$；只有有效 view 和有效 token 参与 Attention。离散 state 进入文本 prefix，**不是直接串在动作序列首位的浮点 State Token**。

虽然两侧 hidden width 不同，但每头 Q/K/V 投影在注意力点积空间里兼容。设观察侧 $X_p\in\mathbb R^{B\times P\times2048}$、动作侧 $X_a\in\mathbb R^{B\times30\times1024}$：

$$
Q_p=X_pW^p_Q,\quad K_p=X_pW^p_K,\quad V_p=X_pW^p_V,
$$

$$
Q_a=X_aW^a_Q,\quad K_a=X_aW^a_K,\quad V_a=X_aW^a_V.
$$

在 head 维统一后将 K/V 沿 token 轴组合，按访问 mask 计算 Attention，再分别经过各专家输出投影。**联合关注不意味着专家共享 QKV 参数**。这是对你已经熟悉的一段式自动驾驶中 PV/BEV 特征交互的直接类比：交互空间一致即可，不要求两个模块的 hidden width 一致。

## 3. Attention Mask：本文比常规 Prefix/Suffix 多了「训练期文字」分区

将序列按功能分为三组：

- $P$：观测、指令、状态前缀，双向注意力。
- $T$：**训练期** reasoning + language-action target，按自回归因果注意力。
- $A$：30 个动作专家 Token，块内可互相读取，并读取 $P$。

允许访问矩阵（行 Query，列 Key）：

| Query \ Key | P | T | A |
| --- | :---: | :---: | :---: |
| P | **✓** | × | × |
| T | ✓ | **因果 ✓** | × |
| A | ✓ | **×** | **✓** |

![EgoLAP 三分区注意力、Stop Gradient 与损失回传](/Embodied-AI-Roadmap/diagrams/egolap-mask-gradient.svg)

*图 2｜训练 Attention 不是单纯把全部 Token 拼接后做全注意力。Action Token 不能看文字 target，才能在完全不生成文字的推理路径中独立工作。蓝色梯度仅到 VLM，绿色梯度仅更新动作专家。*

为什么还要**前缀读不到文字**？因为第 $\ell$ 层的前缀若读到了含未来动作答案的文字，下一层动作专家虽然禁止直接读 T，却会读已经被污染的前缀 P，形成**跨层间接答案泄漏**。同理，Prefix 也不应该读含未来带噪动作的 A；否则多步去噪时观测表示依赖不断变化的噪声，无法稳定缓存。

**梯度隔离**（knowledge insulation）：

$$
\mathcal L_{\mathrm{FM}}\big(v_{\phi}(A^{\tau},\tau,\operatorname{sg}[h_\theta(x)])\big),
$$

其中 $\operatorname{sg}$ 在反向传播视为常量。它保证

$$
\nabla_\theta\mathcal L_{\mathrm{FM}}=0,\qquad
\nabla_\phi\mathcal L_{\mathrm{FM}}\ne0
$$

（这里的第一式指 **FM 梯度支路** 对前缀参数为零，并不表示 $\theta$ 整体没有更新）。VLM 由语言动作/理由的 Cross Entropy 更新，Action Expert 由 FM 更新。

反事实比较：**只截断动作头的参数梯度**不等于阻止 FM loss 回传 Backbone；关键是切断 $h_\theta$ 进入动作专家的计算图边界。**只冻结 VLM**则连文字监督也无法更新，丧失作者利用 human motion intent 调整共享表征的主要机制。

## 4. 一秒 Language Action 怎么从世界系位姿得到？

对时间窗口起点 $t$，定义 $F_t=(R_t^F,p_t^F)$ 作为本体相关的 egocentric frame。给定世界系手/末端 $(R_\xi,p_\xi)$，求其在该帧中的表示：

$$
\bar p_\xi=(R_t^F)^\top(p_\xi-p_t^F),\qquad
\bar R_\xi=(R_t^F)^\top R_\xi.
$$

并计算**相对于同一个固定窗口起点坐标系**的终点变化：

$$
\Delta p=\bar p_{t+1s}-\bar p_t,\qquad
\Delta R=\bar R_t^\top \bar R_{t+1s}.
$$

$\Delta R$ 被转换为 **extrinsic XYZ Euler**，在此定义下 roll/pitch/yaw 顺序不能任意互换。不同数据域的 $F_t$：带 wrist camera 单臂机器人用 EE frame；无 wrist 的单臂与双臂机器人用 base frame；人类 EGO 用 head-camera frame。它们并非同一物理坐标系，而是选择视觉方向更可比的**相机相关约定**。

量化规则：平移到**整数厘米**，旋转角到**5 度倍数**，小于半量化格的分量通常省略，最终夹爪/手开闭状态始终保留。序列顺序是 x（前/后）→ z（上/下）→ y（左/右）→ roll → pitch → yaw → gripper，例如：

> Left arm: move forward 5 cm, move left 2 cm, tilt left 10 degrees, close gripper. Right arm: move down 10 cm, open gripper.

注意上述句子仅为**教学输出格式范例**，不是机器人执行脚本。语言动作量化误差理论上每维平移可达约 0.5cm、角度约 2.5°；它不适合直接做亚毫米插接控制。真正精细控制仍靠 30×14 连续 FM 动作序列。

### 4.1 为什么要用相机方向规范化？

假设 EGO 相机相对世界旋转 90°，世界系的「向北移动」在画面里可能是「向右」；若不做坐标转换，模型会看到相近画面却被监督成完全不同的动作词。旋转到 $F_t$ 后，语义可在某些视角下更一致，但**并非严格跨本体不变量**：镜头高度、光轴方向、左右镜像和手爪结构仍会引入差别。

### 4.2 图像-only EGO 的边界

公式必须用 $q_{t:t+K}$ 才能产生语言动作与 FM 真值。**仅 RGB 的 EGO 视频既没有 $q_t$，也没有夹爪状态**。要复用论文管线，须先通过具置信度的手部重建/UMI 轨迹/视觉 VIO + 标定把动作恢复出来，或者只做单独的视觉目标自监督，不能伪装成作者式直接监督。

## 5. 三条训练路径必须从计算图分别理解

**路径一：人/机共享语言动作 CE。** VLM 看到图像/指令/状态，训练目标为 $m_t$。这条梯度使人和机器人对于「同样往前移动一小段」共享预训练文字 token 空间；不是证明完整执行控制能完全泛化。

**路径二：Motion-level Reasoning CE。** 部分训练样本另有 $r_t$，通常解释当前物体接触、几何关系、为何此方向可达、接下来物体会如何改变。它不是长程子任务清单，不应单纯重复动作词。作者使用离线 Gemini 标注，**将当下目标 Language Action 作为标注输入**；这是 label-generation 的特权信息，允许作为离线训练标签，但绝不能作为推理时输入。

**路径三：连续 FM。** 人类手部动作与机器人动作都监督 Action Expert；原文 Appendix C.1 验证如果**屏蔽人类 FM loss**，模拟总体成功率下降 **4.4 个百分点**，语言条件类任务从 **26.0% 降至 10.0%**。它直接否定了「人类数据只训练 VLM，FM 只训练机器人」这种过度简化说法。

## 6. 训练损失完整推导（论文 Eq.2、5–8）

选择已按域归一化的真值动作块 $A\in\mathbb R^{30\times14}$、高斯噪声 $\epsilon$。原文使用

$$
\tau=0.001+0.999z,\quad z\sim\operatorname{Beta}(1.5,1),
$$

$$
A^\tau=(1-\tau)A+\tau\epsilon,\qquad
u^\star=\epsilon-A.
$$

故模型在噪声端 $\tau=1$、动作端 $\tau=0$ 的方向约定下训练：

$$
\mathcal L_{\mathrm{FM}}=
\mathbb E_{\tau,\epsilon}
\frac{1}{30\cdot14}
\left\|v_{\phi}(A^\tau,\tau,\operatorname{sg}[h_\theta(x)])-(\epsilon-A)\right\|_F^2.
$$

这与把噪声设为 $\tau=0$、动作设为 $\tau=1$ 的等价记法**目标速度正负相反**，读 openpi 必须先确认 $\tau$ 约定。

文本目标分为理由 $r=(r_1,\dots,r_M)$ 和 Language Action $m=(m_1,\dots,m_N)$；训练按 token 交叉熵，只在有效位置监督。论文使用两个独立 Bernoulli：

$$
B\sim \operatorname{Bernoulli}(0.5),\qquad
C\sim \operatorname{Bernoulli}(0.5).
$$

$B$ 决定是否把理由以 teacher-forced 前缀放入 $m$ 的因果上下文；$C$ 决定当 $B=1$ 时是否对理由本身计算 CE。对**有理由标签**的样本，统计分布：

| 训练模式 | 概率 | 训练目标 | 意义 |
| --- | ---: | --- | --- |
| $B=0$：直接动作语言 | 50% | 只算 $m$ CE | 不能依赖永远出现的理由 |
| $B=1,C=0$：理由作上下文 | 25% | 算 $m$ CE，不算 $r$ CE | 可利用理由上下文，避免理由过强监督 |
| $B=1,C=1$：理由与动作 | 25% | 同时算 $r$ 与 $m$ CE | 让模型能显式解释物理原因 |

对**没有理由标签**的样本固定 $B=C=0$；$m$ 永远监督。总目标：

$$
\mathcal L=
\mathbb E_{p_{\rm mix}}
\left[
0.8\,\mathcal L_{\mathrm{text}}
+1.0\,\mathcal L_{\mathrm{FM}}
\right].
$$

**不要把「32% 的训练 batch 有 reasoning」误读成「32% 的样本一定同时训练 reasoning CE」**。考虑上面的随机机制，只有带 reasoning 的样本的一部分真正计算理由 CE；但语言动作目标仍覆盖全数据。

## 7. 数据混合、标签来源与训练计算账本

论文总体训练数据约 **10,000 小时**，一个 batch 里权重如下，不是按小时自然比例抽样：

| 数据组 | batch 权重 | 域 | reasoning |
| --- | ---: | --- | --- |
| EgoVerse MECKA / Scale / Aria | 25%（20 / 4.15 / 0.85） | Human EGO | 只 MECKA 有 |
| ABC YAM | 30% | Bimanual robot | 无 |
| AgiBot / Galaxea / MolmoAct2 | 25%（15 / 5 / 5） | Bimanual robot | 无 |
| OXE（DROID 等） | 20%（DROID 12） | Single-arm robot | 只 DROID 有 |

因此 reasoning 标注覆盖 **MECKA 20% + DROID 12% = 32% batch**。其余样本只有 Language Action 与 FM。*不同来源 action/state 归一化用每类的 1%–99% 分位区间*；不要把人手坐标和机器人 EE 直接用混合全局均值归一化。

论文 Table 4 真值配置：**batch 2048、64×TPU v6e、40k steps（EgoLAP 实际对照 checkpoint 约 30k）、Adam β=(0.9,0.95)、峰值学习率 5e-5、2500-step warmup、weight decay 1e-4、EMA 0.999、全局梯度裁剪1、bf16、10次 Euler 推理积分**。这种预训练资源不是普通单卡 LoRA 即可完全复现的实验；单卡建议仅做小任务受控的研究性替代验证。

### 7.1 一次 batch 的伪代码（概念结构，不冒充官方脚本）

~~~python
for batch in mixer(human=0.25, abc=0.30, bimanual=0.25, oxe=0.20):
    images, instruction, normalized_state = preprocess(batch)
    motion_text = deterministic_serialize(batch.pose, batch.action, batch.embodiment)
    rationale, reason_loss_mask = stochastic_reasoning_supervision(batch)
    prefix = vlm_encode(images, instruction, normalized_state)
    text_loss = masked_language_ce(prefix, rationale, motion_text)

    truth_action = per_domain_normalize(batch.action)  # [B,30,14]
    eps, tau = sample_noise_and_beta_time()
    noisy = (1 - tau) * truth_action + tau * eps
    target_velocity = eps - truth_action
    pred = action_expert(noisy, tau, stop_gradient(prefix))
    flow_loss = mean((pred - target_velocity) ** 2)

    loss = 0.8 * text_loss + flow_loss
    backward_and_step(loss)
~~~

真实工程需额外实现 **相机缺失 mask、state bin、有效动作维 mask、时间重采样、每类归一化、checkpoint/EMA 和序列边界**。上例没有隐藏第三方 API 调用，也没有假定作者训练代码已公开。

## 8. 推理：为什么文字可以完全不生成？

因为动作分支的 Query **从训练开始就禁止读取 T**。推理只需观测 $x_t=(o_t,\ell,s_t)$、一个高斯噪声动作块，执行：

$$
A^1\sim\mathcal N(0,I),\qquad
A^{\tau-\Delta}=
A^\tau-\Delta\,v_\phi(A^\tau,\tau,h_\theta(x_t)),\quad \Delta=0.1.
$$

重复 10 次从 $1\rightarrow0$，反归一化得到 30×14 chunk；**没有**先让 VLM 生成一段 reasoning 再喂专家。连续控制端仍需要动作解包、控制频率限制、在线重感知和安全检查。最常见延迟错误是把 30 物理动作步当成必须一次全部开环执行；本体控制应依据真实任务执行部分 chunk 并重规划。

## 9. GRPO 后训练：不是用成功率给 Action Expert 直接做 RL

语言动作语法可解析，所以作者在 **ABC 数据上仅后训练 VLM 文本分支**。每个样本生成多个 Language Action 候选，并与记录的动作比较，构造负误差奖励：

$$
C=\sum_{b\in\{L,R\}}
\left[
\frac{\|p_b-p_b^\star\|_2^2}{(2\,{\rm cm})^2}
+\frac{\|\log((R_b^\star)^{-1}R_b)^\vee\|_2^2}{(\pi/18)^2}
+\mathbf 1[g_b\ne g_b^\star]
\right],\qquad r=-C.
$$

归一化后组内优势（32 条观测，每条 64 个样本）：

$$
\widehat A_{ij}=
\operatorname{clip}
\left(
\frac{r_{ij}-\bar r_i}{\max(\sigma_i,10^{-4})},
-4,4\right).
$$

作者用 PPO clipped surrogate + 相对冻结 SFT 的 $k_3$ KL penalty（clip 0.2、KL 权重 0.02）更新**语言 Backbone 与 tied embeddings**，冻结视觉 encoder 与 Action Expert。它优化的是**与示范动作的文本匹配代理分数**，不是真机任务成功奖励，也不意味着连续动作专家直接参与 PPO 更新。

有一个容易忽略的消融：**推理阶段不生成 reasoning**；GRPO 采样时只生成 Language Action 的方式比连 reasoning 一起采样更好，作者报告相对提升约 19.4% vs 12.7%。原因之一是用于 GRPO 的 ABC 场景并非原 reasoning 标注分布，直接要求模型先生成陌生理由会产生分布偏移。

## 10. 论文实验读数：不要把进度当成功率

| 实验 | 结果 | 指标范围 |
| --- | --- | --- |
| EgoLAP 全模型 | 80.1% | 五项真机任务**阶段进度**平均，不是完全任务成功率 |
| EgoLAP 全模型 | 38.3% | 14 项**零样本仿真成功率**（1400 episodes） |
| 人机混训、无理由 | 16.4% | 同一仿真评测 |
| 机器人独训、无理由 | 7.2% | 同一仿真评测 |
| 人机混训+理由 | 38.3% | 相对无理由 +21.9 pp |
| 机器人独训+理由 | 18.4% | 相对无理由 +11.2 pp |
| EgoFAST（无理由） | 1.6% | 零样本仿真；相同连续专家，文字 token 换 FAST |
| EgoOAT（无理由） | 2.7% | 零样本仿真；相同连续专家，文字 token 换 OAT |

**两层贡献要分开看**：人类数据把 7.2→16.4，说明语言动作能更好吸收 human 数据；理由再把 16.4→38.3，说明 Physics-why 标签进一步组织表征。作者视觉表征检索实验：Robot→Human 三技能 R@1 由 51%→76%（+25pp）。这说明同技能表征更接近，但**不代表每个动作都具备可靠跨本体控制能力**。

作者在五项真机任务上测的是：任务分解阶段完成数 / 总阶段数，再对 episode 平均。不能将 80.1% 描述为「80.1% 整任务成功」。

## 11. 教学复现：从 EGO/UMI 视频到语言动作数据的一整条最小链路

**Step 0｜先检查你能否使用这条技术路线**。至少需要 RGB+手/夹爪位姿+开合（含 timestamp），以及可执行机器人数据，或者一个已在多样机器人上预训练好的基座。只有 RGB 请先学习 [RGB-only EGO 的四条监督路线](./ego-rgb-only-to-pi05/)；不要凭文字 label 伪造 30×14 机器人 Action。

**Step 1｜多源时间轴**。对头部相机、腕部鱼眼、UMI VIO/Tracker、编码器分别建立统一时戳。按目标控制频率重采样到一秒 30 步，确保文字与连续目标取自**同一时刻、同一段轨迹**。位姿断裂/遮挡 > 阈值时应设置 valid mask 而不是线性插值穿越遮挡。

**Step 2｜统一 Frame Tree**：先求得 $T_{world\leftarrow camera}(t)$、$T_{world\leftarrow hand}(t)$；每段按论文指定 $F_t$ 转成动作坐标。固定桌面 AprilTag 用于校验尺度和世界系稳定，不应把头坐标误认成机械臂 base。检查手移动固定物体时的重投影误差及相机运动补偿。

**Step 3｜生成确定性 Language Action**。把每一秒 $\Delta p,\Delta R,c_{final}$ 量化成统一英文规范句；必须测试旋转 90° 后预期前/后/左/右是否与所选 egocentric frame 一致。输出每条训练样本的 label 可重复生成、解析出近似原值，并量化误差。

**Step 4｜离线理由标注**。以**当前帧、任务、近 5 秒动作摘要、未来一秒动作摘要**生成理由；生成模型只能读取离线标注中允许的未来动作信息，不能进入部署端。以固定比例人工抽检：是否把“即将接触”误说成“已经接触”、物体名称是否错配、是否编造力信息。标注脏数据不得训练。

**Step 5｜生成数据契约**（建议单条记录）：

~~~json
{
  "episode_id": "lineA_00014",
  "source": "human_umi",
  "frame_id": "head_camera",
  "timestamp_ns": 123456789000,
  "calibration_id": "ego_fisheye_v2",
  "images": ["base.jpg", "wrist_left.jpg", "wrist_right.jpg"],
  "instruction": "把两个接头对齐放入治具",
  "state_20": "20 floats; measured / padded by valid mask",
  "action_30x14": "30*14 floats; normalized per embodiment",
  "language_action": "Left arm: move forward 4 cm, close gripper. Right arm: open gripper",
  "reasoning": "离线文本或 null",
  "action_valid_mask": "30*14 bool",
  "image_valid_mask": [true, true, false],
  "split_group": "physical_station_fixture_02"
}
~~~

这里**示例为教学 schema，不是 EgoLAP 官方数据文件格式**。尤其需要保存 action_mode/频率、旋转单位、左右手身份、遮挡置信度和机器人可达性 QC。不可把「只观察到左手」误标成右手停止运动，缺失动作需通过 mask 处理。

**Step 6｜动作语义审计**。同一动作块能被 Language Action 解析；文本量化误差 < 0.5cm/2.5°（不含追踪与坐标误差），但**不能**以此证明 3D 位姿预测精度。在机器人的 base frame 下还需做逆运动学可达性测试和碰撞检查。

**Step 7｜先跑 CPU 契约，再训练**。本站 [egolap_contract_cpu_demo.py](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egolap_contract_cpu_demo.py) 使用**标准库**完成旋转坐标测试、语言动作量化、三分区 Attention Mask 防泄漏、FM 噪声方向与10步欧拉 oracle、Reasoning Dropout 概率及 GRPO 组内优势。运行：

~~~bash
python3 examples/egolap_contract_cpu_demo.py
~~~

脚本 **不实现** SigLIP/PaliGemma、FM 模型训练、LLM 理由生成或机器人动力学；PASS 只代表数据/数学契约成立。

**Step 8｜训练阶段**。小规模路线建议拿已有开源 π₀.₅（[openpi](https://github.com/Physical-Intelligence/openpi)）重用动作专家与图像预处理，另外实现语言动作 CE 和固定原因句的 teacher forcing。先在 10–20 段样本上过拟合，验证 Text CE 与 FM loss 的下降、前缀 grad 只来自 CE；再加入人机数据做 matched ablation。**此为自行改造，不是 EgoLAP 作者开源版本。**

**Step 9｜仿真和实机**。最低四组：Robot-only vs Human+Robot，再各加/不加 reasoning；动作表征继续比较 Language vs FAST。固定 robot demo 数、训练步数、camera 数和评价初始种子，分别报告整体成功率、阶段进度、接管率、动作耗时及 OOD camera/新治具表现。缺少真实 robot 数据且无多样机器人预训练底座时，不能把实验名称写成「从纯人类 EGO 直接训练通用机器人」。

**Step 10｜错误恢复闭环**。按 episode 记录所有失败模式（手部追踪失锁、标定偏移、量化损失、对象接触错位、不可达、语言任务错解、FM 动作错误）。每一类都要归因到采集/坐标/标注/模型/执行，才能让新增 EGO 小时数产生可衡量的收益。

## 12. 研究风险、可证伪实验与工程边界

**风险一：理由标签的特权信息。** Gemini 生成理由时**已经看到了真值 Language Action**，因此理由改善可能一部分来自更丰富的动作转述。建议做三组：只看当前视觉生成理由、看真值动作生成理由、随机打乱理由，测是否还提升。这是本站提出的额外实验，不是作者宣称已完成的全部消融。

**风险二：语言量化丢掉接触精度。** 1cm/5° 格子无法表示小间隙插接或精细力控。注意语言动作是 Backbone 对齐目标，连续控制另有全精度动作专家。

**风险三：坐标假设并不严格相同。** Human 用 head frame、双臂机器人用 base frame，受摄像机安装高度和方向影响。生产场景要测试视角扰动下的 Language Action 一致性，而不只是 tokenizer 语义。

**风险四：训练数据高度混合。** 10k 小时预训练与多类硬件先验对结果贡献很大，不能把 80.1% 归因于 32% reasoning 标注单因素。

**风险五：所谓零样本边界。** 作者未在 14 项仿真任务上训练，不代表不使用机器人预训练数据；真机 YAM 机架和相机位置是未见配置，但训练含其他 YAM 数据。实际新企业工位的泛化需要独立测试。

**风险六：GRPO 优化的是演示匹配代理**，不是任务回报。奖励只能推动更像记录动作的**文字**；论文没有证明这等于复杂插接任务的在线成功强化学习。

**风险七：源码开放状态。** 2026-11-02，作者项目页的代码与 checkpoint 均标为 Coming soon。本站 Python 合成契约与架构教学不是完整作者代码复现，也不能凭推导假称已经在 GPU 跑通。

## 13. 一个高级工程师应会回答的面试问题

**问：既然动作专家会读取 VLM 前缀，为什么还要语言动作标签？**  
答：因为隔离梯度后 FM 只更新专家，不会告诉 Backbone 哪些跨本体运动语义重要。Language Action/Reasoning CE 通过 Backbone 的语言空间学习共享的运动语义，再由 Action Expert 消费这个前缀。不是「文字通过后续 Token 强行流入动作」，而是它们在训练中改变了可复用的 Prefix 表征。

**问：Prefix 为什么不能看 reasoning？**  
答：多层 Attention 下 Prefix 若读到训练期真值未来动作，Action Expert 下一层再读 Prefix，就发生隐式标签泄漏；训练推理输入分布不一致。Mask 必须对每层都成立。

**问：有了 Language Action，为什么推理不先生成文字？**  
答：Action Query 从训练时就与文字 Target 隔离，连续 FM 仅依赖观测前缀，故无需文字解码。GRPO 的文本采样也不属于在线控制时的必须步骤。

**问：人类数据是否只帮助语义？**  
答：不是。原论文 Appendix C.1 直接证明屏蔽人类 FM loss 会降低 4.4pp 模拟总体成功率，说明作者让人类动作也训练了动作专家；即便跨本体动作不能直接执行，其运动分布仍可作有益的增强信号。

**问：现实里是先上 EgoLAP 还是 EgoEngine？**  
答：如果你已有可靠的 EGO/UMI 6DoF 轨迹和机器人数据，先尝试 Language Action 共训练与语义对齐，数据成本较小。如果只有 RGB、无法可靠追踪双手/物体，可先做 WING 的视觉交互 latent 或建设感知重建链；如果要求**生成可执行机器人动作伪示范**，再转 EgoEngine 的重建/重定向路线。

---

## 论文、权威实现与状态

- 原论文及完整 Appendix A–D：[arXiv 2610.08726（2026-10-06）](https://arxiv.org/abs/2610.08726)。
- 作者项目页与真实训练数据混合/评测图：[EgoLAP 官网](https://ego-lap.github.io/)；截至核查日代码、权重均未开放。
- 其理论基础 LAP：[LAP 项目](https://lap-vla.github.io/)；对应 openpi 需自行增加本文文字监督。
- 可访问的连续动作开源基础：[Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi)，与作者代码**不是同一个仓库**。
- 本站前置课程：[π₀/π₀.₅ Attention 逐层审计](../../01-foundations/vla/pi0-joint-attention-code-audit/) · [EGO RGB-only 的监督边界](./ego-rgb-only-to-pi05/) · [EgoEngine 重建闭环](./egoengine-code-reproduction/) · [WING 频谱与交互 latent](./wing-ego-spectral-transfer/)。

*出版与实验数字均应按论文原文重新核验；本站合成实验没有任何真机结果。*
