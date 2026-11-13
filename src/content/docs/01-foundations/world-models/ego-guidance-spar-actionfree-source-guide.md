---
title: "Ego Guidance / SPAR：无动作 EGO 视频如何训练 VLA 视觉表征（论文与源码精读）"
description: "2026-11-10 论文及官方实现，涵盖 Qwen3.5 双 Expert、Flow Matching、视觉特征回归、时序对齐、Ego4D 双质量门、跨本体数据混训和 LIBERO-Plus 复现"
sidebar:
  order: 3
---

> **资料更新 2026-11-13**。一手来源：[论文 HTML（2026-11-10，v1）](https://arxiv.org/html/2611.13685)；[作者开源仓库](https://github.com/llong-cs/EgoGuidance)；[训练配置](https://github.com/llong-cs/EgoGuidance/tree/main/configs)；[数据合同](https://github.com/llong-cs/EgoGuidance/blob/main/docs/data.md)。下文明确区分【论文结果】【源码核查】【教学推导】【建议实验】。本站没有完成 GPU 训练、LIBERO rollout 或真机复现，只有 CPU 数学合同。

# 为什么一段没有动作标签的人类 EGO 视频仍值得保留？

工业 EGO 视频的定位可能不稳定，6D 手部轨迹噪声大；直接把伪动作当目标机器人动作会污染行为克隆。**SPAR（Shared Prediction on Action and Representation）**走另一条路：机器人数据学动作，人类 EGO RGB 视频**不需要重建动作**，只教共享视觉语言主干预测未来视觉特征。它与 [EgoWAM](./egowam-world-targets-hpt-cotraining/) 不同：SPAR 的视觉目标是冻结编码器的 future-frame representation，不预测必须可执行的人手动作，也不依赖相机补偿的 3D Flow。

![SPAR 双 Expert 训练与部署架构](/Embodied-AI-Roadmap/diagrams/egoguidance-spar-system.svg)

*图 1｜原创教学重绘；图中的 future RGB/feature 属于训练标签，不会进入推理输入。*

## 1. 把模型拆到输入、输出、shape

【论文 Appendix A】视觉语言主干为 **Qwen3.5-2B**，LLM 24 层、宽度 2048；Action Expert 与 Generation Expert 均为独立 24 层 Transformer，32 heads ×64 dim，逐层 cross-attend 同一 VLM hidden state。两路 Expert **不互相 Attention**。

| 数据流 | 在线可见输入 | 监督真值 | Expert |
| --- | --- | --- | --- |
| Robot (DROID) | 当前 RGB（主视角/腕视角）、文字、相机 role | 动作块 \(A_t\)，未来主视角的特征 \(Z_{t+\Delta}\) | Action + Generation |
| EGO (Open-AoE/Ego4D) | 当前头相机 RGB、step description | 未来头相机特征 \(Z_{t+\Delta}\) | **Generation only** |
| LIBERO 下游 | 当前多视角、语言、有效本体状态 | Robot action / future observation | Robot 监督 |
| 部署 | 当前观测与文本 | 输出动作块 \(\hat A_t\) | **只保留 Action** |

默认视觉目标使用**冻结 SigLIP2 So400m/16** 的 future frame patch features，张量 \([B,256,1152]\)；Action Expert 是 32 个可学习提示 token + \(H\) 个噪声动作 token。DROID 配置为 \(H=8,d_a=7\)（单臂速度+夹爪）；论文真实双臂测试用 14 维关节+夹爪，并采用 50 步预测/30 步执行。**不能把 DROID 的 8×7 接口当成已验证的工业双臂通用动作接口**。

## 2. 逐项推导 Flow Matching 和视觉目标

设共享 VLM 参数为 \(\theta\)，动作 Expert 为 \(\psi\)，生成 Expert 为 \(\omega\)，冻结目标编码器为 \(\phi\)。

$$
\mathcal H=F_\theta(o_{\le t},\ell),\quad
A_t=[a_t,\ldots,a_{t+H-1}]\in\mathbb R^{H\times d_a}.
$$

【论文公式 1】令 \(s\in[0,1]\)、\(\epsilon_A\sim\mathcal N(0,I)\)，在高斯噪声与动作标签之间建立直线路径：

$$
A_t^s=(1-s)\epsilon_A+sA_t,\qquad
v^\star_A=\frac{\partial A_t^s}{\partial s}=A_t-\epsilon_A.
$$

动作速度回归损失：

$$
\mathcal L_A=\mathbb E_{s,\epsilon_A}
\frac{\|f_\psi(A_t^s,s;\mathcal H)-(A_t-\epsilon_A)\|_2^2}{H d_a}.
$$

**【教学手算】** 若 1 维动作 \(A=0.8\)、噪声 \(\epsilon=-0.2\)、\(s=0.3\)，则 \(A^s=0.1\)、目标速度 \(v^\star=1.0\)。网络不该直接输出“真动作 0.8”，而应输出流场速度 1.0。零噪声起点并非该方程的必要条件。

【论文公式 2】给当前视觉条件的未来主相机图像取冻结特征：

$$
Z_t=\phi(o_{t+\Delta})\in\mathbb R^{N\times d},\quad
Z^s_t=(1-s)\epsilon_Z+sZ_t.
$$

视觉分支可用独立 Flow Matching：

$$
\mathcal L_G^{flow}=
\mathbb E\frac{\|g_\omega(Z_t^s,s;\mathcal H)-(Z_t-\epsilon_Z)\|_2^2}{Nd}.
$$

也可直接回归未来视觉表征（对应不同 Generation Expert 头）：

$$
\bar Z_t=\frac{Z_t-\mu_\phi}{\sigma_\phi},\qquad
\mathcal L_G^{reg}=\mathbb E\frac{\|g_\omega(\mathcal H)-\bar Z_t\|_2^2}{Nd}.
$$

这里 \(\mu_\phi,\sigma_\phi\) 为未来视觉特征的标量统计，不是机器人的动作 q01/q99。视觉编码器 \(\phi\) 冻结，VLM 与专家训练；单图预测时 \(N=256,d=1152\) 对应 SigLIP2 target 的默认形状。

## 3. 为什么视频能改变动作网络？反向传播链路

【论文公式 3】机器人批次与人类视频批次优化的目标分别为：

$$
\mathcal L_R=\mathcal L_A+\lambda\mathcal L_G,\quad
\mathcal L_E=\lambda\mathcal L_G,\quad\lambda=1.
$$

共享参数的梯度：

$$
\nabla_\theta\mathcal L=
\nabla_\theta\mathcal L_A^R+
\lambda\nabla_\theta\mathcal L_G^R+
\lambda\nabla_\theta\mathcal L_G^E,
\qquad \nabla_\psi\mathcal L_G^E=0.
$$

最后一项解释了**没有动作标签的 EGO 数据如何影响未来的动作**：通过改变 VLM hidden states。Generation Expert 并不会给 Action Expert 发送生成帧，也没有在线规划。若错误地从 Generation loss 中 detach 共享 VLM，EGO 监督就不能改善动作路径的共享表征；反过来两路梯度也可能冲突，不保证提升。

训练期间两个 Mini-batch 分别 forward，按自己的有效样本 mean loss 后累积梯度，再做一次 optimizer step。**1:1 sampling_ratio 指的是每步样本曝光比，不等于将生成 loss 乘 0.5**，这是复现代码中最容易混淆的细节。

推理从高斯动作噪声用 \(M=4\) 步 Euler 离散积分：

$$
A^{s_{k+1}}=A^{s_k}+\frac{1}{M}f_\psi(A^{s_k},s_k;\mathcal H).
$$

Generation Expert 在部署时移除，所以推理量级相当于常规 VLA；但预训练增加了一套视觉专家，训练成本明显更高。它没有自动解决抓取失败、控制接触误差与长任务恢复。

## 4. 源码逐文件追踪（2026-11-13 核对）

| 官方文件 | 为什么要读 | 契约/坑 |
| --- | --- | --- |
| [README](https://github.com/llong-cs/EgoGuidance/blob/main/README.md) | 权重、环境、两阶段训练、LIBERO/Plus 评测指令 | 代码已开源，但数据、模型须自行准备 |
| [cotrain_openaoe.yaml](https://github.com/llong-cs/EgoGuidance/blob/main/configs/pretrain/cotrain_openaoe.yaml) | DROID + Open-AoE 1:1 | `video_data.losses: [gen_loss]`；没有 action_loss |
| [train_starvla_gen_cotrain.py](https://github.com/llong-cs/EgoGuidance/blob/main/starVLA/training/train_starvla_gen_cotrain.py) | 多流绑定 loss、前向、梯度累积 | robot action+gen；ego gen only |
| [gen_target_sampling.py](https://github.com/llong-cs/EgoGuidance/blob/main/starVLA/dataloader/gen_target_sampling.py) | future time offset / multi-frame grid | 训练物理时间必须一致 |
| [gen_model/experts.py](https://github.com/llong-cs/EgoGuidance/blob/main/starVLA/model/modules/gen_model/experts.py) | RegressionExpert 与 DiffusionExpert | 跨层 cross-attention，不能当成一个普通 MLP |
| [gen_model/losses.py](https://github.com/llong-cs/EgoGuidance/blob/main/starVLA/model/modules/gen_model/losses.py) | MSE/Cosine 与 Flow velocity | 表征与 VAE 目标的噪声 schedule 不同 |
| [tests/test_temporal_alignment.py](https://github.com/llong-cs/EgoGuidance/blob/main/tests/test_temporal_alignment.py) | 官方 offset 单测 | 无 GPU 先跑对齐合同 |
| [docs/data.md](https://github.com/llong-cs/EgoGuidance/blob/main/docs/data.md) | LeRobot v3 和视频 `index.jsonl` | 原始 EGO 裁剪与文本生成是预处理 |

**【源码】** README 说明 Python3.10、torch2.6/CUDA12.4、flash-linear-attention 0.3.2 和 causal-conv1d 1.5.0.post8；预训练默认全局 batch 1,024（32 GPU ×32，robot / EGO 512:512），20k steps。论文报告 Qwen3.5-2B 的 LLM 学习率 \(2.83\times10^{-5}\)，专家 \(2.83\times10^{-4}\)，500 warmup、ZeRO-2、bf16。对于 4 卡显存资源不能直接保证可跑，先配置极小 smoke batch，并测 activation/optimizer 实际开销。

## 5. 时间对齐：0.533 秒不能偷换成“8 帧”

论文规定主视角未来帧与动作 chunk 末端时间对齐。若机器人动作为 \(H=8,f_R=15Hz\)：

$$
\tau=\frac{H}{f_R}=\frac8{15}\simeq0.533\mathrm{s},
\qquad \Delta_s=\left\lfloor\tau f_s+\frac12\right\rfloor .
$$

故 DROID 15FPS 对应 **8 帧**，EGO 30FPS 对应 **16 帧**；LIBERO 20FPS 下 \(H=8\) 对应 **0.4 秒 / 8 帧**。如果未来序列采样四帧，按 \(k\tau/T\) 取点：DROID \([2,4,6,8]\)，EGO \([4,8,12,16]\)。

**错误示例**：EGO 30FPS 仍偏移 8 帧，等于把 0.533 秒预测目标变成 0.267 秒；同一 Generation Expert 同时学两种未来长度，损失下降也不能解释为一致世界动态建模。不能在片段末端简单 clip future index 到最后一帧，这会破坏严格未来监督。

![SPAR 不同频率 future target 对齐与 EGO 双质量门](/Embodied-AI-Roadmap/diagrams/egoguidance-spar-timing-qc.svg)

*图 2｜原创示意。横向刻度与物理时间对应，并展示数据质量门及 source→loss 路由。*

## 6. Ego4D 双质量门：先保留有交互价值的帧

【论文 Section 4 / Appendix B】原始 Ego4D **3,976h**，按**非重叠 5s**切片；并非所有片段都应无条件训练。

**Gate 1（角点/光流质量）**：\(N_F\) 个可评价点，\(n_F\) 个通过，最长失败 run 为 \(L_F\)，要求：

$$
N_F>0,\quad \frac{n_F}{N_F}\ge0.85,\quad n_F\ge3,\quad L_F\le2.
$$

保留约 **1,893h（48%）**。注意相机运动小不代表没有真实物体运动；该门筛掉光流不可靠的视频，不是高精度相机 SLAM 的替代。

**Gate 2（可见手）**：有效手框 conf≥0.30、面积占图像 2–50%、与 ROI \([.075,.075,.925,1]\) 相交；若 \(N\) 个样本帧中至少一只手可见的帧数为 \(n_H\)，最长不可见 run 为 \(L_H\)：

$$
N>0,\quad \frac{n_H}{N}\ge0.1,\quad n_H\ge1,\quad L_H\le5.
$$

两门累计保留约 **975h（25%）**。不要求左右手同时可见，也不意味着 hand pose 准确。论文显示原始未经筛选视频直接混训甚至弱于 Robot-only；双门有效提升 OOD 实验表现。作者通过预处理从 5s 片段采样 15 帧，以本地 Qwen3.6-35B-A3B 生成单句英文 step description。**官方仓库需预先准备 cut clips / captions，不是运行训练时自动做视频理解**。

## 7. 视觉目标选择，不等于“SigLIP 永远胜 DINO”

【论文 LIBERO-Plus】对冻结 future frame 视觉特征比较（对应 Diffusion 目标的平均 OOD 结果）：

| 未来预测特征 | \([N,d]\) | 平均 OOD 成功率 |
| --- | ---: | ---: |
| SigLIP2 So400m/16 | [256,1152] | **76.2%** |
| Qwen-ViT | [49,2048] | 75.6% |
| DINOv2 ViT-g/14 | [257,1536] | 74.9% |
| Wan2.2 VAE | [256,48] | 60.0% |

DINOv2 在传感噪声与物体布局上更好，Qwen-ViT 对光照/背景更好；不能凭总平均宣布某视觉编码器必然适合所有工业任务。Future SigLIP2 **1 帧 74.32% / 4 帧 74.62%**，只有 0.30 个百分点差异；而额外帧增加视觉目标计算开销。作者还报告直接 regression 是可用对照，不一定非做视觉 diffusion。

## 8. 数字必须如实呈现：OOD 变好 ≠ 真机任务全胜

【论文】Open-AoE 视频从 0 / 150 / 450 到 1500 小时时 LIBERO-Plus OOD 整体向上，LIBERO ID 变化较小；但固定 200k clip-caption 样本时，单独提高场景/对象/交互多样性不一定单调提升。

【论文 Table 1】实际五个双臂任务，每任务 **10 次**：

| 任务 | π₀.₅ | SPAR |
| --- | ---: | ---: |
| 黄瓜转移 | 9/10 | 9/10 |
| 花束交接 | 9/10 | 8/10 |
| 面包片装载 | 2/10 | 2/10 |
| 杯子挂置 | 1/10 | 0/10 |
| 瓶盖旋开 | 1/10 | 1/10 |
| **总计** | **22/50 = 44%** | **20/50 = 40%** |

黄瓜转移的视觉扰动上 SPAR 强于 π₀.₅，但花束交接光照/干扰条件下表现更差。样本小、任务分布和预训练数据不同，不应统计宣称“等价”“全面更鲁棒”或“超越 π₀.₅”。作者自己承认精细抓取、双臂旋转、长时任务仍难。

## 9. 动手复现三层：CPU → 小样本 smoke → OOD

**第一层：CPU 数据合同，无模型权重。** 本站脚本 [egoguidance_spar_contract_cpu.py](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egoguidance_spar_contract_cpu.py) 检查 flow velocity、跨FPS时域、质量门边界、EGO 无动作损失和未来数据推理泄漏，运行：

~~~bash
python3 examples/egoguidance_spar_contract_cpu.py
# PASS ALL: EgoGuidance SPAR CPU synthetic contracts
~~~

**第二层：作者官方 CUDA smoke（本站未实际跑）**。先按 README 下载 Qwen3.5、SigLIP2、DROID LeRobot v3 与 Open-AoE；环境需要 Torch CUDA fast kernels。

~~~bash
git clone https://github.com/llong-cs/EgoGuidance.git
cd EgoGuidance
# 先执行官方 README 安装和数据准备
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/pretrain/cotrain_openaoe.yaml \
  --run_id smoke --trainer.max_train_steps 2 --trainer.global_batch_size 4 \
  --datasets.total_per_device_batch_size 4 --datasets.vla_data.episode_fraction 0.002 \
  --trainer.save_interval 100 --trainer.eval_interval 100 --trainer.logging_frequency 1
~~~

如果视频裁剪、时间 index、权重、CUDA 版本不全，则不能把上述命令写成已经执行成功。官方提供 [LIBERO](https://github.com/llong-cs/EgoGuidance/tree/main/examples/LIBERO) 和 [LIBERO-Plus](https://github.com/llong-cs/EgoGuidance/tree/main/examples/LIBERO_Plus) 的 policy server + simulator client，分别记录 suite 与扰动成功率。

**第三层：博士生消融矩阵（建议实验，未跑）**。固定机器人轨迹数量、action chunk、VLM 初始化、训练步数、3 个随机种子与评测集，做 robot-only / robot+150h filtered EGO / +450h / +1500h / +450h raw EGO / +450h filtered EGO with DINOv2。每组报告 ID、七类 OOD、Wilson 置信区间、显存、GPU 小时、训练前向时延。**禁止把扩大数据同时引入的场景变化和视觉目标变化都归因于 EGO 数据量**。

## 10. 放回工业 EGO/UMI 数据管线的建议

建立三层样本表：`RGB-only EGO`、`Action-valid EGO/UMI`、`Robot GT`。统一记录 `source_type`、`fps`、`capture_timestamp`、`future_offset`、`action_valid`、`gate1/gate2`、`caption_provenance` 与版本。对只有 RGB 但动作定位失败的片段，可以优先保留视觉未来监督价值；**不要悄悄把 EGO 手部伪标签当真实机器人 action**。只有通过动作坐标、时序、可执行性检测的样本才可进入后续动作共训研究。

**与 π₀.₅ 的关系**：SPAR 官方并不是 openpi 的即插即用 patch。若要在 π₀.₅ 上增加未来视觉专家，必须单独实现 VLM hidden-state 接入、loss routing、future-target 读取、action-only 导出及完整的 baseline/OOD 消融。对双臂精密插装，视觉目标学到世界先验，不代表可以替代夹爪滑移监测、恢复策略或力控。

**复盘问题**：为什么 \(\nabla_\psi L_G^E=0\) 但 \(\nabla_\theta L_G^E\neq0\)？为什么 15FPS 的 8 帧不可照搬到 30FPS？为什么 raw Ego4D 可能使结果比 robot-only 差？为什么 20/50 对 22/50 不说明等价？答案都在上述公式、时域和作者试验数据中。

**本文来源与状态**：[论文原始 HTML](https://arxiv.org/html/2611.13685)、[完整官方开源实现](https://github.com/llong-cs/EgoGuidance)、[数据/配置文档](https://github.com/llong-cs/EgoGuidance/blob/main/docs/data.md)、[独立 CPU 检查代码](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egoguidance_spar_contract_cpu.py)。作者论文结果均非本站复现结果。
