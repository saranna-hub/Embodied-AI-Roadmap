---
title: "SLA：从三维视觉教师到 π₀.₅ 持续预训练"
description: "OmniVGGT 空间特征、多层门控、状态丢弃与跨本体动作坐标的源码级复现"
sidebar:
  order: 32
---

> 2026-11-28 核查：[论文 v1，2026-11-05](https://arxiv.org/html/2611.08373v1)、[项目页](https://embodiedsla.github.io/)、[官方 SLA 目录](https://github.com/OpenHelix-Team/Spatial-Forcing/tree/master/SLA)。本文区分论文报告、已读取源码及独立教学建议；没有执行 GPU 训练。代码 master 会变化，下文给出关键文件 blob 前缀以便追溯。

## 1. 问题：VLA 为什么会抓高、抓偏

在固定工位收集的示教中，关节状态与下一个动作有很强的统计相关性。模型可能不必理解物体相对夹爪的距离，只要记住常见轨迹就能降低训练误差。当目标高度变化时，动作模型仍执行熟悉的轨迹。空间教师的作用是给视觉表征附加学习信号，而不是直接修改执行器命令。

SLA 从已有 π₀.₅ 权重出发，在持续预训练和任务后训练中共同优化动作与空间表征。不同于直接增加深度输入的 VLA，它在常规部署中移除辅助教师；RoboDojo-Real 的历史图像记忆是单独的例外，不能把该变体也描述成完全不变的输入。

![空间教师训练与部署](/Embodied-AI-Roadmap/diagrams/sla-spatial-system.svg)

图 1：本站原创。实线为策略路径，虚线为训练辅助路径。标定信息只给教师，不表示部署必须有深度相机。

## 2. 张量和动作生成：先跑通原始模型

设 batch 为 B、相机视图数 V、每视图 patch 数 P、语言主干宽度 D、层数 L。学生的视觉隐藏状态堆叠形状为 B×L×(VP)×D；固定层时去掉 L 维。动作张量是 B×H×A，H 是预测长度，A 是内部动作维度，不等于机器人实际接口维数。训练时图像和指令构成 prefix，噪声动作构成 suffix；不要把语言 token、状态 token 或 action expert 的隐藏状态误作视觉 patch。

官方 [pi0_align.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/src/openpi/models/pi0_align.py)（blob 85013715）直接继承 Pi0，使用 HiddenCaptureSpec，expert_index=0 指向 PaliGemma。实际图像 token 数由输入图像高宽除以 14 后相乘并跨视图求和。这个规则依赖当前视觉编码器，换 backbone 时必须重验，而不是写死 256。

该文件的 Flow Matching 使用噪声端 1、数据端 0 的时间约定：

$$
x_t=t\epsilon+(1-t)a,\qquad u_t=\epsilon-a.
$$

$$
t=0.999s+0.001,\quad s\sim\mathrm{Beta}(1.5,1).
$$

$$
L_{\mathrm{act}}=\mathrm{mean}((v_\theta(x_t,t,o)-u_t)^2).
$$

采样从噪声出发以负时间步积分。若动作增量方向颠倒，即使空间蒸馏准确也会执行错误。源码中的动作损失为直接 mean；若移植到含无动作 EGO 或缺失手臂维度的数据，需另加有效维和样本掩码，不能把这项工程扩展冒充官方实现。

## 3. 教师：为什么是 OmniVGGT

教师通过跨视图注意力建立空间关联，同时允许 RGB、内外参和深度的不同组合。缺少几何输入不代表空间坐标为零；缺省标记与真实零值必须有不同语义。对同一时刻多相机，需保证相机身份、时间同步、尺度与外参方向一致。

论文取教师末层特征，加入位置编码后按 patch 网格做双线性重采样：

$$
Y=\mathcal R(F_{\mathrm{3D}}(I,K,T,d)+E).
$$

其中 K 为内参，T 为外参（传给教师前必须按其接口确认方向），d 为深度，缺失几何可由教师自身机制处理；E 是位置编码。教师输出的特征不是直接米制深度，不能将每个通道解释成毫米。位置对应关系比张量形状相同更重要：图像 crop、resize、padding、左右相机顺序错乱都会导致学生学到错误区域。

例如 224×224 图像的 14 像素 patch 网格为 16×16。若教师给出 24×32 网格，需沿两个空间轴插值到 16×16，不能将 768 个 token 当作一维序列插值；相机之间也不能混插。若把图像横向缩放一半，像素内参的焦距与主点坐标必须同步变换。裁剪原点为 (c_x,c_y)，缩放因子为 s_x,s_y 时：

$$
K'=\begin{bmatrix}s_x&0&-s_xc_x\\0&s_y&-s_yc_y\\0&0&1\end{bmatrix}K.
$$

这是工程推导；具体像素中心约定应与实际图像库一致。

[align_utils.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/src/openpi/training/align_utils.py)（blob 35e0e440）在 Torch 教师抽取函数上使用 inference_mode，从 aggregator 的 patch_start_idx 以后提取空间 token；并构建 image/padding mask。这里 JAX 学生与 Torch 教师分属不同框架，目标通过主机及设备传输。不要为冻结教师保留计算图。

## 4. 多层门控：不是 softmax 注意力

论文在 PaliGemma 的 18 层上共享归一化、两层 MLP 和门控参数。对每个 patch 与层的未投影特征 h，先算投影和通道门：

$$
q_{\ell i}=\mathrm{MLP}(\mathrm{LN}(h_{\ell i})),\qquad
g_{\ell i}=\sigma(W_gh_{\ell i}+b_g).
$$

$$
z_i=\sum_{\ell=1}^{L}g_{\ell i}\odot q_{\ell i}.
$$

门值在 0 和 1 之间，层之间不要求和为 1。所有层均有正证据时可共同贡献；不同通道可选择不同层。用 softmax 替换 sigmoid 是不同实验，不是等价改写。

![门控、掩码及动作坐标](/Embodied-AI-Roadmap/diagrams/sla-spatial-detail.svg)

图 2：本站原创。每个 patch 对应一个教师目标；状态丢弃是样本级；动作增量以 chunk 起点为锚。

论文使用 cosine loss：

$$
\ell_i=1-\frac{z_i^\top y_i}{\max(\|z_i\|,\varepsilon)\max(\|y_i\|,\varepsilon)}.
$$

[projectors.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/src/openpi/models/projectors.py)（blob 305b8f44）实际采用每样本有效 token 均值，再跨 batch 均值：

$$
L_{\mathrm{align}}=\frac1B\sum_b\frac{\sum_i m_{bi}\ell_{bi}}{\max(1,\sum_i m_{bi})}.
$$

这与把所有样本有效 patch 混在一起做 global mean 不同。比如一个样本只有 1 个有效 token 且损失 1，另一个有 3 个有效 token 且损失均为 0，源码式聚合是 0.5，全局 token 均值是 0.25。全无有效 token 的有限特征样本贡献 0；若无效位置含 NaN，应在计算前清洗，不能依靠 NaN×0。

$$
L=L_{\mathrm{act}}+\alpha L_{\mathrm{align}}.
$$

教师冻结，梯度从辅助分支回传学生和 projector；cosine 不监督目标模长，也不等价于直接深度误差。可用固定 depth probe、场景外泛化及动作成功率三类证据区分表示质量与控制能力。

## 5. 源码与论文配置并不完全相同

| 源码位置 | 本次核查的实际行为 |
|---|---|
| [train_align.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/scripts/train_align.py)（c760269b） | Pi0Align、JAX value_and_grad、TorchAlignFeatureExtractor、AsyncAlignFeatureWorker；同卡串行，分卡可重叠 |
| pi0_align.py | 捕获视觉 prefix，按配置固定层或多层；返回 total/action/align 三个损失 |
| projectors.py | 多层时沿 layer 轴 sigmoid 加权求和；向量归一化后做 mask cosine |
| [train_libero_plus.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/src/openpi/external_config/train_libero_plus.py)（f62c2291） | pi05=True、H=10、state_dropout=0.25、align_loss_coeff=0.5、固定第12层、自适应关闭、相机参数关闭、共享设备8卡、batch512 |

特别注意：论文的 RoboTwin 消融最佳是 18 层自适应，但公开 LIBERO-Plus 示例配置默认固定层。两者不是矛盾，也不能只运行默认配置就宣称复现所有论文消融。若改成多层，需确认 capture 层索引、堆叠轴、gate 输入与输出宽度，记录有效训练参数及显存。

共享 GPU 模式有串行教师提取与学生训练的代价；分离设备模式可重叠，但需严格把教师输出绑定到相同 batch ID。异步队列错配不会必然报 shape error。记录 align_compute_s 和 align_transport_s 比仅报告 GPU 利用率更有诊断价值。

## 6. State dropout：打破状态到轨迹的捷径

每个训练样本独立决定是否去掉整段状态 token。不是逐维把物理状态置零，也不是删除对应动作标签。图像、指令及当前动作监督仍然保留，推理恢复正常状态。

$$
d_b\sim\mathrm{Bernoulli}(p),\qquad p=0.25\ \text{(公开 LIBERO 示例)}.
$$

可通过只改变工件位置、保持初始机器人状态相同的成对样本检验模型是否依赖视觉。若去掉状态使接触精度下降，应分别报告开放空间抓取和接触阶段，不要把总体均值当成每种工序都改善。状态不足时，视觉遮挡下可能无法推断夹爪真实姿态，因此全量丢弃不是更强正则的必然方向。

## 7. 跨本体动作统一：坐标是数据契约

论文的绝对 EEF 状态使用统一机器人中心坐标（单臂基座，双臂基座中点，前左上）；动作块每一步都相对同一个当前时刻，而不是前一预测步。令 p、R 已表达在该坐标系：

$$
\Delta p_j=p_j-p_0,\qquad \Delta R_j=R_jR_0^\top.
$$

$$
\Delta r_j=\operatorname{Log}(\Delta R_j)^\vee.
$$

恢复绝对命令：

$$
p_j=p_0+\Delta p_j,\qquad R_j=\operatorname{Exp}([\Delta r_j]_\times)R_0.
$$

这里是基坐标轴下的旋转增量，不是腕部局部增量。若下游接口期望局部增量，需要换成 R_0^T R_j 并相应处理平移；非交换三维旋转时顺序尤其重要。179° 到 -179° 的欧拉角直接相减是 -358°，正确相对旋转约 2°。若混入 UMI 使用 wrist-local 的数据，必须显式转换；不要在归一化之后才纠正坐标。

论文选择旋转向量逐通道归一化。训练集分位数只能在正确物理单位、同坐标且过滤异常后计算；评测不能参与。重建 action chunk 时必须使用采样起点状态而非新执行状态。接管、重定位或断帧发生时，应切分连续段而非跨越缺口补出动作。

## 8. 规模、实验与可复现边界

论文 Table 1 的约 171 万条、14330 小时是六个来源数据在筛选之前的统计，而不是实际使用量。含模拟和真机示教，不是纯 EGO 数据。作者进行整 episode 下采样与任务过滤，并排除会造成下游评测泄漏的数据；原始录制、工位和任务模板应保留来源 ID。

作者报告持续预训练 36k optimizer steps、全局 batch 4608（两次 microstep）、48 张 H100 约 110 小时，约 5280 GPU 小时。即便四张同型号 GPU 理想线性缩放也约 1320 小时；实际通信、教师和数据加载可能更慢。它不是单人一周完整预训练方案。

论文 Table 6 的 RoboTwin Clean2Random（50 个任务）两阶段对照：

| 空间监督阶段 | 5% 数据 | 25% | 50% | 100% |
|---|---:|---:|---:|---:|
| 都关闭 | 43.8 | 46.5 | 46.1 | 48.2 |
| 仅后训练 | 45.6 | 53.4 | 55.3 | 57.1 |
| 仅持续预训练 | 46.6 | 52.0 | 58.4 | 63.2 |
| 两阶段 | 49.2 | 60.5 | 67.4 | 72.3 |

数字为成功率百分数，来自论文而非本站复测。全量两阶段与全关闭相差 24.1 个百分点，不等于从随机初始化训练的收益。LIBERO-Plus 上 SLA 87.4%，π₀.₅ 84.4%，收益显著依赖评测条件。论文私有八条件平均为 83.8% 对 59.8%，以 object instance 计分，多个物体同一 rollout 并非完全独立，不能视作通用工业成功率。深度 probe 也不能证明隐特征在任意工厂均可恢复绝对深度。

## 9. 从官方 checkpoint 跑通最小闭环

官方 README 提供 uv 环境、训练组件、基础及 benchmark checkpoint 下载命令。建议固定 upstream commit 后执行。以下为官方流程的精简版（未在本站执行）：

~~~bash
git clone --recursive https://github.com/OpenHelix-Team/Spatial-Forcing.git
cd Spatial-Forcing/SLA
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
hf download physical-intelligence/libero --repo-type dataset --local-dir Data/libero_lerobot/
hf download haofuly/SLA_base --repo-type model --local-dir checkpoints/
hf download haofuly/SLA_training_component --repo-type model --local-dir checkpoints/
uv run scripts/compute_norm_stats.py --config-name train_libero_plus
uv run --no-sync scripts/train_align.py train_libero_plus --exp-name=experiments
~~~

下载后检查目录是否满足配置中的 checkpoints/SLA_base/params 和 checkpoints/omnivggt；固定资产和权重哈希。官方示例要求 8 卡，不要直接在 4 卡上照搬 batch512 和 fsdp_devices=8。先确认训练 config 注册、设备可见性、teacher/student layout、batch 可整除及梯度累积逻辑，再用 1 个样本验证前向、梯度和保存恢复。官方 README 中 LIBERO-Plus 评测另需 Python 3.8 环境与 assets，运行 serve_policy.py 后调用 examples/libero/main.py；RoboTwin 和带记忆的 RoboDojo-Real 转至 XPolicyLab 对应 PR。README 提供模型下载路径不等于本次已验证权重可下载或结果可复现。

## 10. 四卡研究方案：验证因果而不是堆组件

先选择 2 个代表任务，保持下游真实动作数据、训练步数、初始化、增强和评测种子一致。A 为基础 π₀.₅，B 为固定层空间蒸馏，C 为多层门控，D 为 C 加状态丢弃。先从任务后训练开始，再用 UMI 有动作数据或 RGB-only EGO 做小规模持续预训练对照。区分 LoRA 挂载位置：若冻结全部视觉和语言路径且仅训练 action expert，辅助 loss 无法改变学生视觉特征；必须验证梯度到达选定视觉表征参数。

EGO RGB-only 是本文提出的扩展方向，不是 SLA 已验证设置。可令来源有效性 a_b 表示存在真实动作标签，使用：

$$
L_{\mathrm{act,m}}=\frac{\sum_b a_b L_{\mathrm{act},b}}{\max(1,\sum_b a_b)}.
$$

纯 RGB 样本只更新对齐分支可到达的学生参数，不把零动作或人体腕轨迹当机器人命令。动作掩码之外还需处理视图缺失和教师可信度。可先缓存冻结教师 patch 特征，但缓存 key 必须含图像内容、标定、增强及教师版本，否则在线裁剪后的学生特征与缓存失配。

建议记录：每来源有效样本数、有效 patch 比例、两种损失及共享参数梯度范数、门值饱和率、训练吞吐、显存、depth probe 以及固定起点/目标位移/遮挡下的闭环成功率。保留失败类型与执行耗时，配对比较各任务而非只比较均值。达到可重复收益再扩展数据。

## 11. CPU 合成验证

运行本站独立脚本：

~~~bash
python3 examples/sla_spatial_contract_cpu.py
~~~

验证 sigmoid 非归一化门控、每样本掩码聚合、跨视图空间插值隔离、动作锚点及旋转 wrap、几何内参变换。它不是官方模型推理或训练，只是防止最基本的接口错误。代码与图示由本站编写。未验证真实相机标定、GPU 梯度、实际教师输出、checkpoint 或闭环成功率。

继续学习：[π₀.₅ 视觉蒸馏与梯度几何](../pi05-vision-distill-gradient-geometry/)；[A³-VLA 区域监督](../a3-vla-attention-alignment/)；[EgoStudy 数据扩展](../../../04-human-to-robot/egostudy-scaling-action-components/)。
