---
title: "HubEgo-100K：万小时工业 EGO 数据的采集治理、RGB–IMU 对齐与 VLA 共训"
description: "审计 2026-11 新发布的 HubEgo-100K v1.0 论文和官方 HF 数据卡：72,754 episode、五存储表、原生 MCAP 与帧级 LeRobot、时钟及 IMU 单位陷阱、场景/录制防泄漏、无动作人类视频用于视觉与世界预测预训练"
sidebar:
  order: 12
---

> **更新与证据界限（2026-11-21）**：本教程依据 [论文 arXiv:2611.21938（2026-11-16 发布）](https://arxiv.org/abs/2611.21938)、[作者发布的 Hugging Face 数据集及字段说明](https://huggingface.co/datasets/HubLabs/HubEgo-100K)、[LeRobot 官方库](https://github.com/huggingface/lerobot)、[MCAP 格式](https://mcap.dev/)复核。作者提供的是**数据集与使用示例**，不是已公开的端到端 π₀.₅ 策略训练实现。本站提供两张原创架构/时间图及 [纯 CPU 合成契约](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/hubego100k_alignment_split_contract_cpu.py)；**没有下载受访问条件约束的真实视频、跑 GPU 训练或复现作者未报告的机器人成功率**。

本篇定位：连接 [工业 MCAP→LeRobot→π₀.₅ 数据合同](./mcap-lerobot-v3-temporal-action-contract/)、[HiFi-UMI-2K 动作及六视角解码](./hifi-umi2k-action-video-contract/) 和 [Ego Guidance/SPAR 人类视频无动作共训](../01-foundations/world-models/ego-guidance-spar-actionfree-source-guide/)。研究目标不是“多采 1 万小时就更强”，而是：**怎样把真实工业工人的多样操作记录转换成不泄漏、可追踪、能支撑 VLA 训练的视觉经验。**

![HubEgo-100K 五类物化数据、质量门控及两条训练路径](/Embodied-AI-Roadmap/diagrams/hubego100k-system.svg)

## 1. 先认清数据：10,124 小时不等于有 10,124 小时机器人控制标签

论文题目带有 100K，但**首版 v1.0 只释放了 10,124 小时**，是向 100,000 小时目标推进的第一阶段。区分三种数量：

| 层级 | 论文 v1.0 数值 | 训练含义 |
| --- | ---: | --- |
| 视频 | 10,124 h / 1,093,194,072 帧 | 1080p H.264、输出 30 fps |
| Episode | 72,754 条，最长 30 min、中位 4.9 min | 一个 episode 是过滤后的连续片段或长片段切块 |
| 原始录制 | 41,688 次 | 同一录制可切出多个 episode；切分必须保留同源关系 |
| 采集人员 | 12,227 | 评测应划分不重叠人员 |
| 环境 | 15,548 | 定义为“人员 × 场所类型”，**不是**物理工位 ID 或同等数量独立厂区 |
| 任务内容 | 56 种任务、30 个工作类别 | 35 类手部动作族，18 个技能组 |
| 传感器 | 每帧 6D IMU + 原生传感器流 | 其中 6D 是三轴加速度、三轴角速度，**不是**手部或末端 6-DoF 位姿 |
| 原生存储 | 43.5 TB 视频、1.29 TB 传感器流 | 全量物化不适合先于抽样与 QC 进行 |

论文报告约 83% 的小时数来自工作相关场所；94.7% 采样帧可见至少一只手、76.3% 可见双手、96.9% 时间在任务内。这些**由视觉语言模型估计**，不是对所有帧人工逐一标定的真值。数据卡中的“至少 95%”是概述性取整，涉及定量实验应使用论文精确值及原始聚合口径。

### 三种常被混淆的监督

- **可直接观测**：视频、时间戳、手机运动传感器读数、录制元数据；其可靠性受采样完整性影响。
- **由模型推断**：动作族、技能组、任务描述、步骤说明、手可见率、on-task 比例；需要对语义准确性做抽样复核。
- **根本没有提供**：每个视频帧对应的机器人工具位姿、夹爪开合、基座/关节控制量、接触力、成功失败真值、相机标定内外参。

不能把帧级 observation.state 的六维传感器值传入 π₀.₅ 的 Action Expert，当作末端状态或动作监督。相机头部惯性只刻画头部运动，其与双手和物体运动之间的关系存在遮挡、尺度歧义与非刚体差异。

## 2. 五类存储与主键：为什么看起来都叫 LeRobot，却不能直接训练

官方 v1.0 的 mono/ 根目录有相互独立的五类数据：

| 物理文件/表 | 关键字段 | 读取注意 |
| --- | --- | --- |
| videos/observation.images.main/ | 每 episode MP4 | 30 fps 输出；并非每帧原始 60 fps |
| data/ | timestamp, observation.state, episode_index, frame_index | Parquet 帧行；6 float32 是加速度和陀螺仪 |
| mcap/ | /imu/accel, /imu/gyro, /imu/motion, /video/frames 等 | 完整原生时间序列；可含 mag、heading、baro |
| catalog.parquet | episode_index、recording_id、contributor_id、video_path、质量字段 | 构建 split、定位视频与 MCAP 的权威索引 |
| meta/annotations.parquet（及 subtasks） | description、task_instruction、action_family、difficulty 等 | 机器生成的**episode 级**文本，不是真实时刻动作轨迹 |

图的连线全部由 episode_index 对齐；同一原录制的不同 episode 通过 recording_id 追溯，同一采集员通过 contributor_id 追溯。因此：

$$
\operatorname{join}(V,F,M,C,A)=
\{(V_e,F_e,M_e,C_e,A_e)\mid \mathrm{episode\_index}=e\}.
$$

**不要用 MP4 文件名或数组的第 n 行代替主键**。若来自同一长录制的片段跨训练集、验证集，两侧可能共享同一工具、环境和连续动作，导致 OOD 评估虚高。合并其他 EGO/UMI 数据源时，外部键应加 dataset/revision/source 前缀，避免不同数据集的整数 episode_index 撞号。

官方提供 parquet 目录和 MCAP TAR 内的 member/offset/bytes 索引。按需读取单条 episode 的传感器文件可以利用归档偏移；从整段 MCAP 直接顺序扫描以取一个 episode 可能放大 I/O。**标注与视频都要使用固定 HF revision**，不应让第一次构建的 Parquet 与下次更新的 MP4 混配。

### 两种量纲不同的“状态”

帧级 observation.state 的布局是：

$$
s_t=\left[a_x,a_y,a_z,\omega_x,\omega_y,\omega_z\right]^{\mathsf T}\in\mathbb R^6,
$$

其中加速度统一为 m/s²（包含重力），角速度为 rad/s。原生 MCAP 则保留手机采样**原始单位**：iOS 加速度可能是 g，Android 可能已经是 m/s²，频道元数据记录其单位。安全转换需要：

$$
a_{\mathrm{m/s^2}}=
\begin{cases}
9.80665\,a_g & \text{原始频道单位为 g}\\
a_{\mathrm{raw}} & \text{原始频道单位为 m/s^2}
\end{cases}
$$

不要对已经转换的帧级 observation.state 再乘一次重力常数；也不要只依据设备品牌猜单位，必须读通道元数据。

## 3. 从手机录制到可训练数据：官方九阶段治理与隐性偏差

论文第 3.2 节给出的顺序是：原始采集 → 头戴方式判别 → VLM 密集时间分段 → 手部可见性过滤 → 隐私检测和裁切 → 片段形成 episode → 帧间隔/漂移与同步完整性检查 → 任务类别重平衡 → 语义标注与打包。该流程值得工业规模采集系统模仿，但其数据分布也会因此改变。

**原始多样性 vs 筛选多样性**：过滤无手帧、离任务片段及隐私内容，提升动作相关视觉密度，却会损失手暂时离开视野、找工具、思考及失败恢复等长流程状态。为了评测机器人长任务的异常恢复能力，工业自采数据应保留未处理的原始事件元数据；训练切片另成版本，不可用筛选数据覆盖原件。

**手的“可见” vs 手的“可用”**：模型在约每秒一帧的样本上判断 left/right/both/no hand/unclear，可能忽略瞬时遮挡、接触时刻和工具细小动作。若对精细装配、插接、拧紧任务筛选，可另做手指/物体关键点追踪、遮挡率与动作可识别性人工复核，不能只用整体 hands_share 判定训练有效。

**隐私与质量不是可互换的过滤**：作者按隐私检测删去带人脸、儿童、敏感文字的区间，并有额外时间边界；这不证明所有其他隐私风险都消失。工业场景可能包含屏幕、工艺参数、图纸与工装标识。引入外部数据前要确认授权范围，内部采集需要独立脱敏策略、访问审计和不可逆脱敏版本。

**缺失信号应进入统计**：论文说明 4 个 episode 注释空、22 条缺少 setting type、209 条缺少时间段标签；此外部分 episode 缺少磁场和气压通道。即使训练样本能被加载，也不能在模型条件中假装所有字段都完整。

## 4. 核心数学：RGB 帧时间与 IMU 原生时钟怎样接起来

![头戴视频与原生 IMU、插值和严格在线因果的区别](/Embodied-AI-Roadmap/diagrams/hubego100k-detail.svg)

定义视频第 t 帧的真实采集时间为 $\tau_t$，原生 IMU 在 $\sigma_i$ 时刻采样。论文说明将传感器时间相对同一 episode 第一视频帧表达，并在帧表中进行线性插值：

$$
\widehat z(\tau_t)=(1-\alpha)z_i+\alpha z_{i+1},\quad
\alpha=\frac{\tau_t-\sigma_i}{\sigma_{i+1}-\sigma_i},
\quad \sigma_i\le\tau_t\le\sigma_{i+1}.
$$

这里 $z$ 可以是三轴加速度或角速度。必须先验证严格单调采样时间、左右邻域都存在、间隔不过大和频道单位统一；否则应输出无效 mask，而不是偷偷使用离得很远的样本。

### 一个常被忽略的因果陷阱

**离线对齐与在线输入不是一回事。** 论文公布的逐帧插值利用前后两个原生样本，适合离线特征生产。但在实时策略于 $\tau_t$ 做预测时，右侧的 $\sigma_{i+1}$ 可能尚未到达；此时将插值后的数值当成“截至当前可见”的状态，就是**未来样本泄漏**。在线合约可使用最新历史样本的 zero-order hold、仅历史样本估计或额外可观测延迟：

$$
\widehat z_{\mathrm{causal}}(\tau_t)=
z_{\max\{i:\sigma_i\le\tau_t\}},
\qquad
\max_i(\tau_t-\sigma_i)\le\delta_{\max}.
$$

这不代表历史保持优于插值；而是两种任务需对应不同的数据可用性契约。用作未来状态监督时可以选离线插值，用作当前在线感知时必须有明确的可用时间。

### “30 fps 输出”不表示时间轴每 33.33 ms 都完美

论文报告 5.5% 的 episode 至少存在一个超过标称间隔 1.5 倍的帧间隔，部分从 60 fps 重编码的片段有重复帧，视频的真实时间长度与帧数/30 有漂移；中位绝对漂移 0.017 s、95 分位约 0.155 s。故：

$$
\Delta_t=\tau_{t+1}-\tau_t,\quad
q_{\mathrm{gap}}=\frac{\sum_t\mathbf1[\Delta_t>1.5/30]}{N-1}.
$$

训练样本应按真实 timestamp 检查窗口，在动作用途做时间插值时需先确认事件发生在同一录制连续段。不能把 frame_index/30 作为唯一时刻；也不能将重复视频帧视为独立的新动作信息。

### 为什么惯性无法替代手部米制轨迹

手机 IMU 测的是头部载体比力与角速度。即使完成重力对齐和积分，位置误差也随时间积累；单个头部 6D 序列不足以观测左右手对物体的独立作用。若缺相机内参及外部尺度参考，则从 1080p 视频恢复手部米制位置仍有不确定性。**本数据并未发布可直接使用的标定相机位姿或双手工具 TCP 轨迹**。应将其定位为视觉语义/时序预训练源，不是 HiFi-UMI 20D 机器人动作集的直接替代。

## 5. 动作语义：看懂“一条行动描述”与“动作标签”的差别

作者自动标注的 action_family 是 episode 主导的**类别**；task_instruction 是按流程描述的自然语言指令，长约 35.2 个词，并非每个原子步骤都带起止时刻。两个句子可对应不同的真实手部运动：

- “拧紧螺丝”是语言意图，无法唯一推出工具旋转轴、力矩、接触状态或持续时间。
- “操作钳子”在视频中可见，但不能直接推出夹爪宽度、末端位姿变化或可执行控制序列。

对训练来说，区分任务层级尤为关键：

| 使用方向 | 可以提供的真实监督 | 必须另外补充 |
| --- | --- | --- |
| 视频表征预训练 | 当前/未来 RGB，多视图无，但可构造时间对比 | 训练目标、负样本策略、数据均衡 |
| VLM 任务识别 | episode 级文字及任务类别 | 高质量标签复核；原子时间边界 |
| 视频世界特征预测 | 时间上后一段视觉特征 | 防泄漏切分；跨视角稳定性 |
| 人类→机器人动作学习 | 视频与动作语义的间接辅助 | 自有机器人动作标签、状态/坐标系、执行反馈 |
| 直接行为克隆 π₀.₅ | **无法单独训练** | 同动作空间、连续时间、可控物理动作与有效 mask |

若需要逐帧原子动作标签，可参考本网站的 [EgoActionBench 多轨时间线与标注评价](./egoactionbench-dense-atomic-actions-source-audit/)，但不能把 HubEgo 的 episode 文字未经处理冒充密集标注真值。

## 6. 工业落地方案：少量机器人动作 + 大量 EGO 的分源共训

对工业操作，优先建议做一项可以隔离归因的研究：机器人样本 $D_R$ 保留 (image, language, proprioception, action)，HubEgo $D_H$ 仅提供 (video, timestamp, IMU, weak-language)；共享视觉主干 $f_\theta$，使用两个明确分开的训练目标。

**路径 A：单纯视觉表征学习。** 用 EGO 连续帧做对比/预测损失，再在固定机器人数据量上训练相同动作头。对于低算力和新岗位实战，这是风险最小、归因最清楚的一阶段对照。

**路径 B：动作 + 未来视觉共训。** 模仿 [SPAR 的动作与视觉专家分开监督思路](../01-foundations/world-models/ego-guidance-spar-actionfree-source-guide/)，训练机器人动作专家及人类/机器人共同可用的未来表征预测头：

$$
\mathcal L_{\mathrm{joint}}=
\underbrace{\lambda_R \,\mathbb E_{D_R}[\mathcal L_{\mathrm{FM}}(\widehat v,v^*)]}_{\text{只有机器人数据可用}}
+
\underbrace{\lambda_H\,\mathbb E_{D_H\cup D_R}
[\mathcal L_{\mathrm{future}}(\widehat h_{t+\Delta},\operatorname{sg}(h_{t+\Delta}))]}_{\text{两类视频都可用}}
+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}.
$$

以 $\operatorname{sg}$ 表示停止梯度、$\Delta$ 是**视频的物理时长**而不是动作维度。此式是面向 HubEgo 的**建议实验配方，不是作者已经实现的模型**；实际模型使用 DINO/V-JEPA 特征、损失类型和权重需实验确定。样本配比最好以有效 episode/时长/场景作控制变量，不把数小时的 30 fps 重复帧当成与稀缺机器人动作等权的独立监督。

无动作视频上的 $\mathcal L_{\mathrm{FM}}$ 必须严格 mask 掉。简单地给人类样本填全零动作会把动作专家推向停止运动。未来特征目标只能在训练目标支路出现，不能混入推理时的 prefix。训练期间可接入完整未来图像作监督，推理仅输入当前和历史。

### 一个可以执行的三臂对比

固定同一套机器人数据、seed、policy、训练总步数和评测集，比较 **R-only**、**R + 随机 EGO**、**R + 经过 hands/on-task/时间 QC 的 EGO**。三组机器人动作 token 数保持一致，记录 EGO 有效小时数、唯一 contributor 数、唯一任务类型数和 GPU-hour。为了研究“多样性是否重要”，在等量 EGO 小时下再比较 **集中少数环境** 与 **跨环境覆盖**，避免把数据量变化误判为数据多样性收益。

建议指标：每个任务 success / intervention / time、未见工具和环境的成功率、视觉表征 frozen linear probe、按任务类别分层结果、置信区间。**截至 v1.0 论文主要贡献是数据发布与数据属性测量，尚无本数据训练 π₀.₅ 获得成功率提升的论文实测结论**；相关方法证据来自论文综述所引用的其他工作。

## 7. 切分与 QC：真正决定实验是否可信

遵循论文建议优先构建 contributor-disjoint train/val/test；对同一人员的所有 recording_id 和 episode_id 一次性划组，不能在创建 action/video 窗口之后再随机划分。若有同一物理站点的其他人员共用工位，当前字段未必能识别，应增加自有 site_id 或使用更严格的工位级留出。

$$
G_{\mathrm{train}}\cap G_{\mathrm{val}}=\varnothing,\quad
G_{\mathrm{train}}\cap G_{\mathrm{test}}=\varnothing,
\quad G=\{\mathrm{contributor\_id}\}.
$$

对人类视频，我们建议在 Parquet 层先出一张**可审计的 manifest**，至少保留：

| 字段组 | 必填证据 | 异常处理 |
| --- | --- | --- |
| 溯源 | dataset revision、episode_index、recording_id、contributor_id | 键冲突直接拒绝 |
| 视频 | video_path、原始 FPS、输出帧数、重复帧、间隔 gap、漂移 | 质量分层，计入有效小时 |
| 传感器 | 原生单位、频道、时间偏移、缺失范围、读取来源 | 无单位不可做量纲混合 |
| 语义 | action_family、task_instruction、标注来源、复核状态 | 低置信样本只用于无语义目标 |
| 训练窗口 | split、窗口起止时间、可见输入的最大采样时刻、目标未来时刻 | 同源越界和未来泄漏拒绝 |

**效率与风险**：将 catalog/annotations 两表在远端抽样读取，先按同源、任务、字段完整性筛出几千条，再按 episode 下载小样本视频与 MCAP，避免一上来传输 43.5 TB。注意访问需要在 HF 登录并接受数据条件；CC BY 4.0 允许署名使用，但不等于绕过访问表单、保密和企业数据合规。

## 8. 最小数据读取与复现路线：可复制，但要清楚哪些尚未实跑

官方数据卡给出了按需下载三条 episode 的范例。这里采用**不直接拉取全部视频**的改写步骤：

    python -m pip install -U huggingface_hub pandas pyarrow
    hf auth login
    # 登录并在数据集页面接受访问条件后：
    python - <<'PY'
    import pandas as pd
    repo = "HubLabs/HubEgo-100K"
    catalog = pd.read_parquet("hf://datasets/"+repo+"/mono/catalog.parquet")
    ann = pd.read_parquet("hf://datasets/"+repo+"/mono/meta/annotations.parquet")
    assert catalog.episode_index.is_unique and ann.episode_index.is_unique
    table = catalog.merge(ann, on="episode_index", how="left", validate="one_to_one")
    cols = ["episode_index", "recording_id", "contributor_id",
            "task", "frames", "duration_s", "hands_share", "on_task_share"]
    print(table[cols].head())
    print("episodes", len(table), "unique recordings", table.recording_id.nunique())
    PY

该示例依赖实际安装的 pandas/huggingface_hub 的 hf:// 支持及数据访问权限；**本站并未对受限数据实际运行**，遇到权限/版本问题时应停下来对照官方卡片、而不是假装成功。进一步取样可按照 [官方按集下载示例](https://huggingface.co/datasets/HubLabs/HubEgo-100K) 的 meta/episodes parquet 中 data chunk/file 索引下载相应 Parquet 和 catalog.video_path 所指视频。固定 repository revision，再使用兼容的 LeRobot v3 loader。

读取 MCAP 的注意事项：

1. catalog.mcap_tar / mcap_member / mcap_offset / mcap_bytes 是定位原生流的索引，不能把 TAR member byte offset 当作 MCAP 事件 timestamp。
2. /video/frames 的 capture time 是视频帧采集时刻；/imu/accel、/imu/gyro 具有单独采样时刻；转换到统一时间轴后再插值。
3. 对原生 iOS g 与 Android m/s² 做显式单位解析；验证重力方向与静止区间不能仅凭数值大小自动推断 OS。
4. 复现帧表必须对照原生流抽取几段波形，验证 sample-rate 差异、边界外插、重复帧、同步偏移；论文明确**残余视频-IMU 时间偏差没有独立测量**。

先运行本站的 CPU 合成负例（无需账号、视频、GPU），再允许真实数据进入训练：

    python3 examples/hubego100k_alignment_split_contract_cpu.py

脚本验证 5 项：iOS 原生 g→SI 单位、线性插值与在线因果反例、丢帧/重复帧 QC、contributor+recording 防泄漏、无动作样本 loss mask 以及训练划分计数（脚本输出具体 PASS 标志）。**CPU 测试只能证明接口逻辑**，不能证明真实数据质量、相机内参、语义标注精度或策略泛化。

## 9. 该项目最值得借鉴什么，哪些东西不能直接搬

| 可以借鉴 | 需要重新研发或额外验证 |
| --- | --- |
| 多人并行采集＋工作环境多样性优先 | 工业工位和工艺的去重机制、人员/工位权限 |
| 原生 MCAP + LeRobot 帧级衍生物的双存储 | EGO/UMI 空间外参、手-工具 TCP、世界系 SE(3) |
| 可见手、on-task、时间漂移作为 QC 特征 | 密集原子动作和失败恢复语义的人工金标 |
| contributor/recording 追溯及版本化 | 多采集源统一 schema 与训练样本审核 |
| 低成本视频作为视觉预训练数据 | 可执行动作 supervision、机器人基座对齐和稳定评测 |

**关键局限**：单目头戴 RGB 的视觉尺度不确定；未提供完整相机标定；IMU 属于头部而不是工具；机器生成文本缺少原子时间边界和人工全面复核；本版暂无 HubEgo→VLA 的直接训练效果报告；数据源带有受控访问机制。若把它直接当作“1 万小时工业末端动作真值”，将系统性污染训练标签。

## 10. 推荐的学习与实验顺序

第一步，弄清本页中的五存储结构与 episode/recording/contributor 三层键，能口头解释“为何同人同工位不能随机划视频窗口”。第二步，运行 CPU 负例，证明“离线插值不等于在线因果”“原生加速度不保证 SI 单位”。第三步，只下载少数公开获权 episode 做真实数据读写和人工检查。第四步，按照第 6 节固定机器人动作量搭建 R-only / R+EGO / R+QC-EGO 控制实验。第五步，系统性记录未知环境与失败恢复是否改善，并对损失曲线与任务成功率分别归因。

### 原始来源及声明

- [论文：HubEgo-100K: In-the-Wild Egocentric Video of Skilled Work for Robot Learning，2026-11-16，arXiv:2611.21938](https://arxiv.org/abs/2611.21938)：v1.0 采集流程、数据统计、时间同步、质量与局限。
- [作者官方 HF 数据集、数据卡和实际字段描述](https://huggingface.co/datasets/HubLabs/HubEgo-100K)：访问条件、LeRobot v3/MCAP 目录、字段、单集下载示例及 CC BY 4.0 授权。
- [LeRobot 官方代码](https://github.com/huggingface/lerobot)、[MCAP 官方规范](https://mcap.dev/)：读取格式的上游接口；它们不是 HubEgo 作者提供的完整训练代码。
- [Ego Guidance（SPAR）](https://arxiv.org/abs/2611.13685)：无动作 EGO 视频与机器人动作共训的独立论文参考，不应把该论文结果说成 HubEgo 成绩。

**复现级别：论文/数据卡核查 + 自编 CPU 数学合同；数据访问、真实视频-IMU 对齐、GPU 训练、策略评测均未完成。**
