---
title: "Open-HumanWAM：5万小时 EGO 规模化预训练、IDM 注意力、视觉目标与 49D 动作源码审计"
description: "2026-11 Open-HumanWAM 一手论文、官方 Open-HumanWAM 分支、HF 数据卡与 GoalWAM 训练/推理源码；详解 50K 子集、注意力与动作契约、视觉目标、48步动作、RTC、训练配置及复现边界"
sidebar:
  order: 5
---

> **核对日期：2026-11-22。** 原始报告：[arXiv:2611.06817（2026-11-04 提交，官方 11-05 公布）](https://arxiv.org/abs/2611.06817)；[项目主页](https://dagroup-pku.github.io/open-humanwam/)；**[实际官方代码分支 Open-HumanWAM](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM)**；[50K 数据卡](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K)；[GoalWAM 发布目录](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM/humanwam_goal)。**注意**：互联网上部分摘要把它写成 Open-HumanScale；项目和可运行源码当前使用 **Open-HumanWAM**。官方 README 的旧链接曾指向不存在的 `open-humanscale` 分支，实测仓库实际分支是 `Open-HumanWAM`。以下明确标记「作者报告」「当前代码」「教学推导」「待复现」，不将文中指标当成本手册复现结果。

# 从“人类 EGO 数据很多”到“模型确实受益”：一个可审计的训练路线

上一篇 [HubEgo-100K](../../02-data-engineering/hubego100k-rgb-imu-curation-training-contract/) 解决 RGB/IMU 和元数据的清洗契约，但没有机器人动作真值。本篇把研究问题前移到**如何规模化预训练、如何用小量目标本体数据后训练、如何不把训练目标和在线可见输入搞混**。Open-HumanWAM 的独特价值不仅是“50K 小时”，而是同时公开实验子集、注意力变体、跨域动作映射、RoboTwin 后训练及 Goal 模型推理接口。

![Open-HumanWAM 系统架构：纯人类与人机混合两条路线](/Embodied-AI-Roadmap/diagrams/open-humanwam-system.svg)

*图 1｜本站原创源码导读图。左边为 Vanilla 的人类预训练→机器人适配；右边为 Goal 的人机混合+视觉目标。浅色框代表训练期监督，不是线上可直接使用的控制标签。*

## 1. 三种“规模”绝不混用

| 名称 | 官方当前说明 | 真正的研究用途 |
| --- | --- | --- |
| HumanNet 总体规划 | 约 1,000,000 小时的人类视频资源计划 | 不意味着全部已经公开，也不是本论文预训练所用量 |
| HumanNet Ego | 已发布约 200,000 小时第一人称子集 | 数据大盘，**不是** Open-HumanWAM 实际 50K 训练样本数 |
| **Open-HumanWAM-50K** | **50,000 小时，约 35.9M 选中 clip** | 由 HumanNet、EgoSuite-Open100K、EgoDex、Gen-HumanEgo、EgoVerse 筛选而来 |
| 作者 IDM 放缩实验 | 100/500/1K/5K/20.5K/50K 小时 | 固定/约束模型与机器人后训练协议，研究人类预训练覆盖率效应 |
| 机器人后训练 | RoboTwin 50 个任务，Vanilla 的探索实验每任务 50 条 | 不能用总人类时长替代实际机器人示教数量 |
| 真实机器人混合预训练 | 固定 14,618 h 机器人数据，加入 0/15,901/45,938 h 人类数据 | 属于 Goal 的**另一套**放缩实验，不等于 Vanilla 子集 |

【一手核对】[数据卡](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K) 报告总计 31,396,799 个物理 episode，多个 clip 可能指向同一 episode。数据以 **LeRobot v3** 存放：MP4、Parquet、元数据和原始来源映射；各个 `part-*` 是独立数据集，不能把所有 `episode_index` 当成全局唯一键。

**强制数据键**：`(source, dataset_path, episode_index, start_frame, end_frame)`。同一个录制/物理 episode 的多条 clip 不允许同时进入训练和测试集。数据卡明确给出 100h、500h、1000h、5000h、10000h、20500h、50000h 七个 subset 索引。每一行的 `end_frame` 为**开区间**：

$$
I=[f_{\mathrm{start}},f_{\mathrm{end}}),\qquad
N_I=f_{\mathrm{end}}-f_{\mathrm{start}}.
$$

若 `start_frame=100,end_frame=132`，只能取 100 至 131 的 32 帧。不能改用 `range(start,end+1)`，否则跨 episode 或损坏时间窗口。**数据集虽然有公开卡片，但文件访问仍要求 HF 账号接受条件；当前完整数据下载/预训练载荷上传状态需按下载当日复核**。基础实验应先只下载一个 `part-*` 和子集索引验证解码，再考虑全量。

### 1.1 原始视频为什么不能直接等价机器人动作

源数据能有手腕轨迹与语言标注，但手部估计的坐标、尺度、遮挡与开合量可能不稳；视频也不直接给目标平台关节、夹爪命令与接触力。对不同采集来源必须保留 `source`、相机定义、手部可见度、手型/腕部可信度和处理版本；必要时把低可信动作设为无效监督，同时仍可保留 RGB 用于视觉/世界预测。

推荐的最小数据卡：

| 字段 | 训练前校验 | 不满足的处理 |
| --- | --- | --- |
| `source_clip_id`, `recording_id` | 具备稳定源标识、可关联去重 | 不用于跨域泛化评测 |
| `camera_intrinsics`, `T_world_camera` | 方向与物理量单位已定义 | 不监督米制 3D 动作 |
| `hand_visibility`, `motion_confidence` | 可审计且阈值有版本 | 对应通道 mask，不直接删除时间帧 |
| `timestamp`, `fps` | 单调且逐传感器明确 | 无法可靠对齐则整段隔离 |
| `license`, `consent`, `provenance` | 已满足原始来源限制 | 不得进入可分发/可训练池 |

## 2. HumanWAM-Vanilla：预训练/中间训练/后训练各在学什么？

### 2.1 模型数据流

Vanilla 继承视频生成骨干与动作扩散/流生成专家，将当前视频、文字、状态、手部运动序列编成条件，学习未来视频潜变量及动作。官方 [配置基类](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_vanilla/configs/_base_/model_ia.yaml) 选择 `Wan-AI/Wan2.2-TI2V-5B`，视频 DiT 配置为 30 层、宽 3072；动作 DiT 为 30 层、宽 1024；两者都只是配置级参数，**不是** Goal 模型同一套 36 层 4096 维 Transformer。Vanilla 人类预训练中的手部动作监督存在质量分层，不能由“模型含 Action DiT”推断数据全是高精度机器人动作。

- **IA / Isolated Attention**：较强隔离视觉与动作令牌，利于建立干净对照，但信息互通受限；对应 [IA 实验配置](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM/humanwam_vanilla/configs/scaling_ia) 中的相对动作。
- **IDM / Inverse-Dynamics Attention**：通过视频前后状态约束动作表征的关系；50K 主线 [IDM 配置](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_vanilla/configs/scaling_idm/pretrain_50000h.yaml) 选择 `absolute_camera_frame`，不是直接输出机器人关节速度。
- **Mid-train（EgoPro）**：目标转为兼容头戴+腕部多视角的 W0 模型。官方 [midtrain.yaml](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_vanilla/configs/midtrain.yaml) 设置 10k steps，腕部可作为**条件输入**，未来视频预测只针对头相机。并且 README 明确 EgoPro 的 dataset/processor 完整绑定尚未就绪。
- **Robot post-train**：在 RoboTwin 命令、动作、观测上优化，可配合数据增强，最终 RoboTwin 配置与早期 12k-step 无增强探索配置**不是同一实验协议**。

### 2.2 时间因果与损失示意

令视频潜变量 `z` 为来自 VAE 的编码，`a` 为标定后动作序列，`c_t` 为当下已知 RGB、历史状态和语言。联合训练可写成下面的**教学抽象**（不是声称官方将所有项简单求和）：

$$
\mathcal L_{\mathrm{total}}
=\lambda_v\mathbb E\|v_\theta(z_s,s,c_t)-u_z^\star\|^2
+\lambda_a\mathbb E\big[M\odot\|g_\psi(a_s,s,c_t)-u_a^\star\|^2\big].
$$

其中训练期未来帧/动作产生**监督标签**，部署时只能给模型历史和当下条件：

$$
c_t=\{o_{\leq t},\ell,\mathrm{state}_t\},\qquad
\hat a_{t:t+H-1}=\pi_\theta(c_t,\xi).
$$

对线性 flow 路径（便于检验方向，不等同于公开模型的所有 scheduler 细节）：

$$
x_s=(1-s)\epsilon+s x_1,\qquad
u^\star=x_1-\epsilon,\quad 0\leq s\leq1.
$$

**必须核对 scheduler 朝向**：Goal 官方 [sampler.py](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/backends/native/sampler.py) 使用带 flow shift 的 **10 步二阶 UniPC**，不是照搬 π₀.₅ 的 4 步 Euler；若按另一个模型的推理朝向修改 `sigma`，即使张量形状正确也会生成错误动作。纯 CPU 示意不能证明作者实际 GPU 数值一致。

## 3. HumanWAM-Goal：为何需要单独理解 49D/64D？

![HumanWAM-Goal 的历史采样、动作维度和前缀冻结契约](/Embodied-AI-Roadmap/diagrams/open-humanwam-detail.svg)

*图 2｜本站原创输入输出契约图。末端目标图像属于线上输入；训练阶段从 episode 终帧构造目标图像的流程必须在评测时避免真值泄漏。*

Goal 使用混合人类/机器人经验，推理时接受：`instruction`、`goal_images.head`、`images.head`，可选左右腕部 RGB、先前历史、`observation_timestamp`、机器人 `state`、RTC 的 `prev_chunk`。官方接口位于 [inputs.md](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/docs/inputs.md)，必须先读接口再做 Adapter。

| 输入输出 | shape 或单位 | 因果/语义 |
| --- | --- | --- |
| `images.head`、可选 left/right | HWC uint8 RGB，目标/当前/历史 | 当前和时间早于当前的历史可见 |
| `goal_images.head` | RGB 目标图像 | 来自合法目标设定；不能偷看当前测试 rollout 的真实终态 |
| `history` | 按时间从旧到新，时间戳早于当前 | 不允许用之后采样的“历史” |
| 物理 `state` | `[49]`，带坐标/夹爪定义 | 只传当前已知状态；各域 mask |
| 内部模型维度 | `64` | 官方 `config.py` 的 `action_dim=64`；不等于实际 64 个可执行通道 |
| 动作输出 | `[48,49]` | `actions` 为绝对命令；`relative_actions` 对当前状态相对化 |
| `action_valid_mask` | `[48,49]` 布尔 | 无效维为 0，不进入控制 |
| `prev_chunk` | `[N,49]` 已规划未执行的**绝对**命令 | 可选 RTC，转换到**当前状态**坐标再冻结前缀 |

### 3.1 49D 按语义逐通道解包

官方 [布局定义](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/docs/inputs.md) 使用 Python 左闭右开切片：

| Slice | 语义 |
| --- | --- |
| `0:23` | 本体相关关节通道；含关节夹爪 7、15 |
| `23:26` | 底盘位置二维 + 航向角 |
| `26:29`、`29:33`、`33` | 左手末端 xyz、xyzw 四元数、夹爪 |
| `34:37`、`37:41`、`41` | 右手末端 xyz、xyzw 四元数、夹爪 |
| `42:45`、`45:49` | 头相机位姿，供坐标转换；**在动作输出中禁止预测** |

从源码看 `SharedEEFLinear` 将本体专属 **23D joint** 与共享 **19D (=3 chassis+8 left+8 right)** 分开映射，剩余 **22D** 内部 padding 保持无效。因此：

$$
D_{\mathrm{physical}}=49=23+19+7,\qquad
D_{\mathrm{model}}=64=23+19+22.
$$

这里 `7D` 相机信息是物理输入末尾，并不等于扩展出来的 `22D` padding。对任何有效动作行都要求：

$$
M_{t,j}=0\quad(j\in\{42,\ldots,48\}),\qquad
\hat a_{t,j}=0\quad\text{if }M_{t,j}=0.
$$

如果双臂只启用 14 个关节与夹爪控制量，不能凭“49维接口”就在数据集中人工补 35 个零并标为有效，那会让模型学习错误的恒定控制目标。

### 3.2 相机坐标、四元数和夹爪方向

Goal 在 `robot.json` 中声明 base/head_camera 坐标以及 `gripper_space`。有效四元数必须是 xyzw、单位范数；对于左臂从世界转换到头相机的点：

$$
\mathbf p_C=\mathbf R_{CW}(\mathbf p_W-\mathbf t_{WC}),\quad
\mathbf p_W=\mathbf R_{WC}\mathbf p_C+\mathbf t_{WC}.
$$

坐标求逆方向错一次，位置看似平滑也可能偏移几厘米甚至被映射至相反方向；旋转同样需要严格采用定义好的变换顺序。

特别容易混淆夹爪：通用 `closedness_0_100` 是 **0 开、100 闭**，而 RoboTwin 配置 `command_openness` 是 **0 闭、1 开**，不能直接线性数值同向复制。RoboTwin 的后训练代码显式保留 command openness，不把它误认为米制 aperture；具体由 [robot.py](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/processing/robot.py) 和 checkpoint 的 `robot.json` 决定。

### 3.3 归一化的统计边界

官方 Goal RoboTwin 后训练采用训练集上每维 q01/q99（比例边界）：

$$
\tilde a_j=2\frac{a_j-q_{01,j}}{q_{99,j}-q_{01,j}}-1,\qquad
a_j=q_{01,j}+\frac{\tilde a_j+1}{2}(q_{99,j}-q_{01,j}).
$$

在 `q99=q01` 的常数/无效通道必须另行处理，不可除零；官方 [normalization.py](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/posttrain/src/goalwam_train/normalization.py) 有分组有效性与边界约定。统计只从 RoboTwin `clean_50` 后训练数据计算（其中末端 pose 可复用预训练共享边界），**不得读测试 Random 或 test goal cache**。

论文的目标是跨本体共享运动语义，不代表所有硬件动作都天然兼容。物理控制端还必须有 joint limit、碰撞、夹爪行程、安全停止与时间戳校验。

## 4. 48 步预测，为什么部署不一定执行 48 步？

Goal 配置 `horizon=48`，官方 RoboTwin 评测 [session.py](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/robotwin/session.py) **每查询执行 32 步**，并使用约 0.5 s / 2.0 s 的历史帧，而不是“48 步预测 = 每次都运行 48 步”。

$$
H_{\mathrm{pred}}=48,\qquad H_{\mathrm{exec}}=32,\qquad
H_{\mathrm{unused}}=16.
$$

若模型在 30 Hz 环境输出，则每次执行的理论物理窗口为：

$$
\Delta T_{\mathrm{exec}}=\frac{32}{30}\ \mathrm{s}\approx1.067\ \mathrm{s}.
$$

这是教学上的名义时间，不包括实际推理延迟、控制闭环误差及重规划。如果启用 RTC，前一次未执行部分作为 absolute `prev_chunk` 输入，模型会将其转为当前状态下的相对前缀并设为不可加噪：

$$
N_{\mathrm{prefix}}\leq H-1,\qquad
m^{\mathrm{noise}}_t=0\ (t\leq N_{\mathrm{prefix}}).
$$

**不是**把先前 `relative_actions` 直接塞入 `prev_chunk`，更不能将模型预测 48 步全部作为已执行的状态更新。验证对抗样例：`state_t\) 改变后，若未变更的 absolute `prev_chunk` 被解释成 relative，则端点命令必须被判错。

## 5. 论文放缩规律，什么结论能用到工厂数据规划？

### 5.1 对数线性只是被观测区间内的经验拟合

IDM Scene-OOD 100h→50K 的作者数值为 49.1%→73.2%（**经过机器人适配评估**）。若用简化教学两点拟合：

$$
S(H)=\alpha+\beta\log_{10}\frac{H}{100},\quad
\beta=\frac{73.2-49.1}{\log_{10}(50000/100)}\approx8.93.
$$

它只能解释“每增加一个十倍量级的预训练时长，在该端点拟合下对应约 8.93 个百分点”，**不允许外推**到 500K h 得到“保证成功率”；训练与评测协议、数据质量、覆盖分布、模型容量会改变斜率，更不意味着每多采 1 小时都带来同等收益。

### 5.2 覆盖率的价值不等于更多重复轨迹

作者在相同小时数做过 **scaled diversity** 与 **matched diversity** 比较。关键控制变量是任务与场景覆盖数，不是仅比较文件大小。更高质量的实验设计：

$$
\Delta_{\mathrm{diversity}}=
S(\mathrm{more\ tasks/scenes}\mid H)
-S(\mathrm{fixed\ tasks/scenes}\mid H).
$$

实际工业数采应额外比较：相同 1000h 下，单工位重复示教 vs 跨工件/光照/遮挡/操作者/任务阶段分层采集；保持机器人下游示教数、网络、训练 step 和评测 seed 一致，否则无法归因。

### 5.3 作者数字、公开代码数字有版本差异

| 实验 | 作者项目主页/论文 | 代码分支 README（11-22 审计） | 该如何引用 |
| --- | --- | --- | --- |
| Vanilla 最终 RoboTwin | Clean 92.08、Random 68.88（平均 80.48） | 同值；README 注释现有评测 50 episodes/task，论文写 100 | **不可声称完全同协议** |
| Goal 零样本 RoboTwin | robot-only 11.48 → hybrid 13.94 | 13.94 | 不含目标任务后训练 |
| Goal 论文精修后训练 | Clean 92.70、Random 84.60，平均 88.65 | **另一发布快照** Clean 92.66、Random 86.50，平均 **89.58** | **分别标记实验快照，不择优拼表** |
| 实体 20 任务零样本 | robot-only 20.0 → +45.9K human 30.5 | 代码 README 同量级 | 作者实验；本站未复测 |

特别是项目主页中 *Goal 的 88.65%* 与当前源码 README 中 *89.58%* 同时存在，它们不是四舍五入就能解释的同一组数。**不得把 89.58 写成论文当初的固定数值，也不能用不同版本的 Clean/Random 计算对比增益。**

## 6. 源码逐文件审计：从表到训练，再到执行

| 一手文件 | 关键细节 | 最容易导致的复现错误 |
| --- | --- | --- |
| [50K HF 数据卡](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K) | `dataset_path/episode_index/start/end/source/task/hand`，每 part 独立 | 片段归属、源录制泄漏 |
| [Vanilla 实验索引](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_vanilla/README.md) | IA/IDM、diversity、100-50K 子集，12k 探索与 24k 最终 | 不同设置乱比较 |
| [Vanilla 50K 配置](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_vanilla/configs/scaling_idm/pretrain_50000h.yaml) | 50K 的 `corpus`、global batch 8192、累积16步 | 只看 per-GPU batch |
| [Vanilla pretrain 基类](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_vanilla/configs/_base_/pretrain_ia.yaml) | `data.train._target_: ???`，`processor: ???` | **不能原样直接完成端到端预训练** |
| [Vanilla midtrain 配置](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_vanilla/configs/midtrain.yaml) | 10k，EgoPro 绑定尚待公开 | 错用 RoboTwin 替代 EgoPro |
| [Goal 模型实现](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/backends/native/model.py) | text tower + generator，SharedEEF 23+19，视频/动作桥接，RoPE | 把 Goal 当成普通 pi05 Action Expert |
| [Goal config.py](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/config.py) | 64D 内部、48 horizon、10 UniPC、36 层 4096 hidden | 直接套 Vanilla 16D |
| [Goal observation.py](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/processing/observation.py) | goal→history→current，valid-mask、prefix/noisy-mask | 非因果历史、padding 误训 |
| [Goal posttrain data.py](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/posttrain/src/goalwam_train/data.py) | LeRobot v3 物化、录制终帧目标、80% cache/20% 终帧、历史 dropout | 评测终帧泄漏、时序越界 |
| [Goal posttrain README](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/posttrain/README.md) | 30fps `clean_50`、36k steps、FSDP2、global batch 256 | 把个人 H100 显存推断成必能复现 |

**特别审查**：Goal `goal_cache.tar` 评测缓存应只含从初始头相机图像和任务文字生成、经人工审阅的目标图，且按初始观察/任务/seed 严格匹配。它不是轨迹真实终帧缓存。若测试直接把真实最终成功图像输入，会引入未来信息。官方提供 [Goal cache 检查工具](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/robotwin/goal_cache.py)，必须保留 cache miss 失败，而不是自动替换一条容易成功的测试样本。

## 7. 复现路线：先纯 CPU 合同，后最小子集和目标任务

### 阶段 A｜无 GPU，当天可验证的前置条件

先运行本站 [纯 CPU 契约脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/open_humanwam_contract_cpu.py)：

```bash
python3 examples/open_humanwam_contract_cpu.py
# 检验区间、源录制泄漏、49D mask、q01/q99、RTC 前缀、放缩外推边界
```

这是可复现的**接口合成验证**，不是官方 checkpoint 推理，也不测物理空间重建精度。

### 阶段 B｜只检查官方模型接口（须配算力）

```bash
git clone --branch Open-HumanWAM https://github.com/DAGroup-PKU/HumanNet.git
cd HumanNet/humanwam_goal
bash scripts/setup_environment.sh /path/to/native-env
source /path/to/native-env/bin/activate
# 提前确认 Hopper、CUDA 12.8、Python 3.13、GLIBC 和依赖版本
python examples/make_sample.py
goalwam infer --checkpoint "$GOALWAM_CHECKPOINT" \
  --sample examples/sample/observation.json --output outputs/actions.npz
```

该步骤严格对应当前官方 README；设置 `GOALWAM_CHECKPOINT` 前需从 [模型集合](https://huggingface.co/collections/DAGroup-PKU/open-humanwam) 下载对应模型。官方原生推理要求 Hopper + 特定 cuBLASLt；用户规划的 RTX 6000 Ada **不能直接视为满足作者原生 bit-exact 环境**。可用计算资源、模型显存和数据许可均需实测。

### 阶段 C｜目标任务的后训练与评测（官方开放路径）

```bash
cd HumanNet/humanwam_goal
# 官方 RoboTwin clean_50 源数据、模拟器和模型权重需先单独取得
python -m goalwam_train.convert --raw-root /path/to/robotwin_raw \
  --output /path/to/robotwin_clean_50 --workers 8
python -m goalwam_train.normalization --data /path/to/robotwin_clean_50 \
  --pretrained /path/to/HumanWAM-Goal-Base \
  --output /path/to/outputs/robot.json
torchrun --nproc-per-node 8 -m goalwam_train.train \
  --pretrained /path/to/HumanWAM-Goal-Base \
  --data /path/to/robotwin_clean_50 \
  --goal-cache /path/to/train_goal_cache \
  --output /path/to/outputs
```

以上是**官方命令的复写**，不是已执行记录。其默认 8 GPU、36,000 steps、global batch 256。若只有 4 卡，需重新推导等效 batch/累积/激活检查点与实际速度，不能擅改卡数后宣称完全复现。其原生 RoboTwin 评测要求目标缓存、50×100 seeds×Clean/Random 的**10,000 次**实验，不要只测 10 个容易成功的 seed。

### 阶段 D｜Vanilla 50K 原始预训练：当前不可声称全通

README 展示：

```bash
cd HumanNet/humanwam_vanilla
bash scripts/train.sh --config configs/scaling_idm/pretrain_50000h.yaml
```

但它继承的 [`pretrain_ia.yaml`](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_vanilla/configs/_base_/pretrain_ia.yaml) 仍含 `???` 的数据加载器、处理器、统计路径；公开 README 也称预训练 loader/部分实验处理器尚在整理。**这条命令是官方配方入口，当前不是即拿即跑的可用性承诺**。需要先用一个 part、明确模型权重、动作转换和统计量把端到端 100h 配方打通，才扩到 1K/5K/50K。

## 8. 最小消融矩阵，优先回答研发真正需要的问题

| 实验 | 固定项 | 只变化一项 | 评价指标 |
| --- | --- | --- | --- |
| R0 控制基线 | 同一目标训练集、seed、模型 | 不用人类预训练 | 任务成功率、完成时间、接管率 |
| R1 数据量放缩 | 同一训练协议、测试源外集合 | 100 / 500 / 1000 / 5000 h | Scene-OOD 成功率 vs log(小时数) |
| R2 数据多样性 | **同小时数**、机器人示教数和模型 | 多工件/多人/多场景 vs 单工位重复 | OOD 和 worst-group |
| R3 伪动作质量 | 同 EGO 视频和总时长 | 全监督/置信度 Mask/仅未来视觉监督 | 梯度污染率与策略成功率 |
| R4 Goal 防泄漏 | 固定目标任务、初态与采样种子 | 合法初态条件目标 vs 违规终帧目标（**仅诊断，不报泛化成绩**） | Goal 依赖性、泄漏检测 |
| R5 49D Adapter | 同 checkpoint | head_camera↔base、gripper 方向、mask | roundtrip、动作越界率 |

每组至少保存：来源 clip manifest hash、train/test source ID、原始-转换 action 变换/逆变换误差、回放可视化、单位测试、模型 checkpoint/评测脚本版本。单个任务成功率建议报告重复 seed 和置信区间，而不是只挑最好 checkpoint。

**工业落地建议**：若拥有大量工厂 EGO/UMI 数据与很少机器人示教，先建一个 100-1000h 的**数据正确性与泛化基准**，比较动作伪标签 vs action-free 视觉目标预训练；随后再评估 Vanilla/Goal 的模型投入。这既能区分“数据量效应”和“数据质量效应”，也避免在官方预训练数据 loader 尚不完整时贸然配置超大规模 GPU 实验。

## 9. 已知局限与待验证清单

1. **开源状态不等于全链路复现**：Goal 的推理与 RoboTwin 后训练代码可见；Vanilla 的全量 50K 数据配方还存在未绑定项；数据载荷/权重的可下载性需实时确认。
2. **动作不通用**：49D 是带模型配置的 Shared EEF 容器，接口里的本体 mask、49→64 内部投影、夹爪方向、相机坐标转换都必须对接目标平台。
3. **结果归因受任务分布制约**：RoboTwin、Scene-OOD、真实工厂各有不同操作难度，成功率不跨协议迁移；越多 EGO 数据不一定越好。
4. **训练成本高**：视频世界模型预训练、动作后训练、视频解码与 50K 数据治理不是四张工作站 GPU 轻易完成的“周末 demo”。
5. **控制仍需额外机制**：模型输出 action chunk 不自动提供失败检测、接触力闭环或安全恢复；仍需执行状态识别、人工纠正和回流策略。

**阅读顺序**：先 [无动作人类视频共训 SPAR](./ego-guidance-spar-actionfree-source-guide/) → 本篇 Vanilla 与 Goal 的数据和模型接口 → [HiFi-UMI-2K 动作契约](../../02-data-engineering/hifi-umi2k-action-video-contract/) → [π₀.₅ × LIBERO 微调](../vla/pi05-libero-data-loss-lora-source-guide/)。这四篇覆盖“人类数据质量→预训练目标→跨域动作表示→机器人闭环”的主线。

## 来源与核验记录

- [论文 arXiv:2611.06817](https://arxiv.org/abs/2611.06817)（2026-11-04）与[官方项目图表/版本数值](https://dagroup-pku.github.io/open-humanwam/)。
- [官方 Open-HumanWAM 分支](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM)（2026-11-22 审计；不是不存在的 open-humanscale 分支）。
- [HF 50K 数据卡/许可/索引字段](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K)；[Vanilla 实验配置](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM/humanwam_vanilla/configs)。
- [Goal 源码输入](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/docs/inputs.md)、[模型投影](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/backends/native/model.py)、[后训练说明](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/posttrain/README.md)、[RTC 布局](https://github.com/DAGroup-PKU/HumanNet/blob/Open-HumanWAM/humanwam_goal/src/goalwam/processing/observation.py)。
- 本站仅执行标准库 CPU 合成契约，并在 CI 做构建/渲染/线上页面校验；**没有跑过作者权重、下载全量 50K、GPU 训练或 10k 轮 RoboTwin**。
