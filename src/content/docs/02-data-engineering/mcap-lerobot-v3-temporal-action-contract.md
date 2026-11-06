---
title: "工业 EGO/UMI 数据生产：MCAP→LeRobot v3→π₀.₅ 的时序与动作契约"
description: "用官方 MCAP、LeRobot v3 与 openpi 源码统一采集时钟、episode、帧查询、SE(3) 动作、质量掩码和训练接口；附两张架构图与无 GPU 反例测试"
sidebar:
  order: 9
---

> **2026-11-06 源码核对；工程教学专题，不是新论文，也不是宣称已部署生产转换器。** 依据 [Foxglove MCAP 格式规范（commit 23d6c627）](https://github.com/foxglove/mcap/blob/23d6c627a27ac089d418d9e7f8cdcef9ed86dac1/website/docs/spec/index.md)、[LeRobot v3 官方文档（commit d490e472）](https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/docs/source/lerobot-dataset-v3.mdx)、[LeRobot Dataset 当前实现](https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/datasets/lerobot_dataset.py) 和 [openpi JAX 数据加载器（固定旧版 215abfb）](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py)。两套源码**不属于同一冻结依赖版本**，直接拼接可能不兼容；本文提出的是待实施的适配数据契约。
>
> 阅读顺序：[EGO/UMI 时空标定](./ego-umi-spatiotemporal-calibration/) → [UMI Vive Tracker 到 TCP 外参](./umi-tracker-handeye-pivot-calibration/) → **本篇：从可信测量到可信监督** → [π₀.₅ LIBERO 数据与 Loss 源码](../../01-foundations/vla/pi05-libero-data-loss-lora-source-guide/)。

# 0. 为什么「可以 Replay」不代表「数据可以训练」？

工业 EGO/UMI 采集典型有：双目/腕部 RGB（30 Hz）、Tracker/SLAM 位姿（60–120 Hz）、夹爪编码器（100 Hz）、机器人状态和控制命令（50–250 Hz）。这些消息**不是同一时刻采集、不是同一坐标系，也不一定有同一动作语义**。把各个数组截到相同长度、按行合并后导出 Parquet，会在视觉上看起来正常，却把监督目标错位。

假设执行端 TCP 横向移动速度为 0.45 m/s，动作标签实际比图像晚 25 ms，位置相差：

$$
\Delta x \approx \|\mathbf v\|\Delta t
=0.45\times0.025=0.01125\,\mathrm m=11.25\,\mathrm{mm}.
$$

这不是训练噪声的抽象讨论，而是同一张图像被配上**另一时刻应执行的动作**。对插销、卡扣、插接等毫米级任务不可忽略。本文目标：建立**能被数据 QC、π₀.₅ 训练与真机回放共同检验**的契约。

![采集事件时间、MCAP 保存、统一时间轴、LeRobot v3、openpi 训练的生产链路](/Embodied-AI-Roadmap/diagrams/mcap-lerobot-v3-system.svg)

*图 1｜原始事件不丢、时钟不混用、episode 不串、动作语义可逆。绿色是已存在的官方库能力；橙色 QC/适配器是本文建议工程组件，不代表已发布可直接调用。*

## 1. 三套接口分别负责什么？不能误把存储格式当数据语义

| 层 | 可靠功能（官方依据） | **不**自动保证的事 |
| --- | --- | --- |
| **MCAP** | 以 Channel / Schema / Message 保存多 topic 时间序列；可分块、索引；每条消息具 `log_time` 与 `publish_time` | 两台设备时间戳同源；外参正确；图像曝光时刻就是写盘时刻 |
| **LeRobot Dataset v3** | 低维 Parquet、相机 MP4、元数据/episode 偏移；按 feature + 时间窗口读取；减少每 episode 文件数量 | pose 的左右乘约定；未观测动作生成；延迟补偿方向；自定义 QC 语义 |
| **openpi π₀.₅ 旧版加载器** | 根据 `action_horizon` 和数据集 FPS 计算 `delta_timestamps`；读取动作序列，再经 transforms 和 norm | 某个 v3 新数据集可被旧版 `lerobot.common.datasets` 直接加载；UMI 的自由手持轨迹已经是机器人 joint action |

MCAP 官方 [Message 记录](https://github.com/foxglove/mcap/blob/23d6c627a27ac089d418d9e7f8cdcef9ed86dac1/website/docs/spec/index.md#message-op0x05) 定义的 `log_time` 是**记录时间**、`publish_time` 是**发布时间**；发布时刻未知可设为 log time。二者都**不是协议自带的相机硬件曝光时间**。正确做法：把设备自带的传感器采集 timestamp/曝光中点/IMU sample tick **作为消息 payload 或明确字段保留**，附上 clock-domain 与该会话对应的时钟变换。不能仅因值看起来是纳秒就认为其时间基相同。

### 1.1 建议保留的原始消息和元数据

`/ego/front_left/image`、`/ego/front_right/image`、`/umi/left/tracker`、`/umi/right/tracker`、`/umi/gripper`、`/robot/state`、`/robot/command`、`/calibration/version`、`/task/events`。这些 topic 名是**本站建议示例**，并非 MCAP 或 LeRobot 强制标准。每条建议有：`source_serial`、`source_seq`、`sensor_time_ns`、`log_time_ns`、`clock_domain`、`calib_id`、`validity`；原始 pose 用 `T_parent_child` 和坐标定义保存，不能仅存 7 个浮点数。

**不要为了转换格式删掉：** 丢帧、重连、失跟踪的事件；真正异常通常位于数据切段和恢复时。原始 MCAP 做不可变归档；中间的 HDF5 可作为算法缓存，最终训练视图以**显式版本**导出为 LeRobot v3 或 openpi 当前支持的格式。

## 2. 时钟统一：先校准设备，后构造样本

设设备 $d$ 内部时钟 $t_d$、统一记录时钟 $t_r$；一次录制段落内允许线性漂移：

$$
\boxed{t_r=a_dt_d+b_d},\qquad
\epsilon_{d,i}=t_{r,i}-(a_dt_{d,i}+b_d).
$$

$a_d$ 表示比例漂移、$b_d$ 表示起始偏移。用同源触发器、LED 同步事件或可信双向时间配对拟合；若设备重连、系统重启、温漂突变，必须**切段重新估计**，不能跨全程硬套一个偏置。验证报告至少给出每段样本量、残差 P50/P95、最大漂移、时间戳倒退/重复次数。

特别注意**因果时刻**：

- RGB 有曝光开始/结束时优先存曝光中点 `t_exp_mid`；滚动快门还涉及逐行曝光，不能以一条单时刻解决全部问题。
- IMU 有采样时间；融合系统返回的姿态也许带滤波延迟或预测，不能当作真实观测时刻。
- 机器人 command 具有**发送、执行生效、状态反馈**三个不同事件；数据标注的 action 必须声明到底对应哪一个。
- MCU 与主机不是同一时钟，不可仅拼接两个 `time.monotonic_ns()` 的结果。

归一化后的锚点 $t$ 应选择**策略实际可见的最后一个观测时刻**。若训练使用未来 $t+\eta$ 才到达的相机帧，离线 Loss 可能很好，在线部署则出现无法复现的数据泄漏。

## 3. 对齐是按时间查询，不是按数组索引

每个策略查询样本 $z_t$ 包含过去或当前可用观测 $o_{\le t}$，以及未来动作监督 $A_{t:t+H-1}$：

$$
z_t=\left(I_{t_c},\,s_{t_s},\,\ell,\,A_{t:t+H-1},\,m_{0:H-1}\right),
\qquad t_c\le t,\quad t_s\le t
$$

其中 $t_c$ 是到达时间满足约束的相机曝光时刻，$t_s$ 是在线可获得的状态时刻。若允许离线**两侧插值**取更精确的 $s_t$，须标记这种“非因果高精度标签”，不能当成部署可获得的策略输入；可用于离线 pose 重建或动作 label 估计。时间查询策略建议：

1. **图像**：只选时间上不晚于观测截止时刻，且曝光帧龄 `age ≤ threshold` 的最新有效帧；相机滞后不能用未来帧补齐。
2. **状态**：在线取上一帧/滤波器预测并记录 age；离线拟合位姿标签时仅在同一有效 tracking segment 内使用两侧插值。
3. **夹爪**：离散开/合、接触位、连续开度需分别明确编码；离散态用阶梯保持，连续物理开度可限速插值。
4. **目标动作**：按**物理控制生效时间**与明确的控制周期取样，缺测生成 mask，不复制下一 episode 的首帧。
5. **边界**：episode、校准版本、时钟段、换设备、分布切分都必须作为硬边界。接近终点不足 H 步时应明确 padding/mask 或丢弃，不能静默跨 episode 拿数据。

对于两时刻的 TCP 位姿，平移可线性插值，旋转应使用单位四元数球面插值：

$$
\mathbf p(\alpha)=(1-\alpha)\mathbf p_0+\alpha\mathbf p_1,\qquad
\alpha=\frac{t-t_0}{t_1-t_0},
$$

$$
q(\alpha)=\operatorname{slerp}(q_0,q_1;\alpha),\quad
\operatorname{slerp}=
\frac{\sin((1-\alpha)\theta)}{\sin\theta}q_0+
\frac{\sin(\alpha\theta)}{\sin\theta}q_1,
$$

其中若 $q_0\cdot q_1<0$，先将第二个四元数取负，选最短路径。$\theta$ 是做过符号对齐后的夹角；$\theta\to 0$ 时回退到归一化线性插值。失跟踪、跳变、$\|q\|\ne1$ 应报错或掩码，不要强行平滑。

![同一事件的曝光、控制生效、不同时间窗口与 episode 边界对齐细节](/Embodied-AI-Roadmap/diagrams/mcap-lerobot-v3-timing-detail.svg)

*图 2｜不能把视觉“最近邻”理解为允许未来观测；动作延迟方向、状态查询模式、episode 边界是三个互相独立的开关。*

### 3.1 延迟是需要辨识的参数，不应按经验拍一个常数

把**命令**与**执行后的实际运动**视为两个时间序列，初步可用相关性估计控制延迟 $\hat\tau$，但快速往复动作、低通滤波、机械间隙可能产生多峰。工程上应以**已知触发阶跃/脉冲**校验，并分解：

$$
\tau_{\rm total}=\tau_{\rm camera}+\tau_{\rm infer}+\tau_{\rm transport}+\tau_{\rm controller}+\tau_{\rm actuator}.
$$

若 action 字段定义为**下发命令**，训练时的索引是命令时间；若定义为**实际 TCP 生效目标**，应对齐执行时间。不能无说明把这两种数据混在一个 LeRobot `action` 字段里。

## 4. Action 到底是什么？三个相似表示对应不同的 Loss

定义机器人 base $B$、工具 TCP $E$：$T_{BE}(t)$ 表示 E 到 B，严格使用列向量左乘。同一物理轨迹至少有三种动作监督方式：

| 名称 | 数学定义（忽略夹爪） | 训练/部署要求 |
| --- | --- | --- |
| 世界/基座**绝对目标** | $T_{BE}(t+k)$ | 已标定统一 B 系；解码需要机器人运动学 |
| **锚定到当前帧的相对目标** | $T_{BE}(t)^{-1}T_{BE}(t+k)$ | 每块动作共同锚在 $t$，不能逐步累计前一增量 |
| **逐步增量命令** | $T_{BE}(t+k-1)^{-1}T_{BE}(t+k)$ | 每步乘接上一步，积分误差会累积 |

上表是标准的 SE(3) 右相对变换形式，**不**意味着每个机器人控制器都要求右乘；若控制接口要求世界系左增量，需另定义。旋转不能简单地做欧拉角逐分量相减，尤其跨 $\pm\pi$ 时。

对连续 6D 相对运动可取李代数：

$$
\xi_k=\log\!\left(T_{BE}(t)^{-1}T_{BE}(t+k)\right)^\vee\in\mathbb R^6,
\quad T_{BE}(t+k)=T_{BE}(t)\exp(\widehat{\xi_k}).
$$

**重要纠错：** LeRobot 的 `action` **只是一个 feature 名称**，官方 v3 容器并不规定绝对 / delta / 末端位姿 / 关节电机命令。HuRo 的下一帧绝对关节位置、LIBERO 既有 delta、UMI 的相对 TCP 向量是三种不同对象。必须在项目 manifest 标明 `action.kind`、`frame`、`units`、`actuation_time`、`gripper_mapping`、`mask`、`robot_model`、`retargeting_version`。

例如 UMI 记录 $T_{WE}^{\rm human}$ 不等于可直接用的机器人 $q_{\rm robot}$；还需 $T_{BW}$、TCP 工具关系、可达性/碰撞/IK 与指爪映射。无机器人 Action 的 EGO 视频即使导出 v3，也**不会凭格式转换自动产生 π₀.₅ 的 Flow Matching 监督**。

### 4.1 为什么说「归一化」也是契约？

对每个机器人/动作定义域采用独立统计 $s$：若用 q01/q99 分位数近似归一化：

$$
\widetilde a_j=2\frac{a_j-q_{0.01,j}}{q_{0.99,j}-q_{0.01,j}}-1,\quad
a_j=q_{0.01,j}+\frac{\widetilde a_j+1}{2}(q_{0.99,j}-q_{0.01,j}).
$$

此处是 **openpi 可选分位数归一化的示意**，不是 LeRobot `meta/stats.json` 的统一强制变换；具体取决于 openpi 模型配置和 transform。维度常数（分母趋零）、统计版本与 gripper 单位需要单独处理。规范是：**训练归一化、部署反归一化和动作 mask 必须引用同一个 manifest/统计版本**。

## 5. LeRobot v3：存储的变更对百万小时级生产意味着什么？

根据官方 [v3 文档](https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/docs/source/lerobot-dataset-v3.mdx)，典型目录：

```text
dataset/
  meta/info.json                 # FPS、features/dtype/shape、path templates
  meta/stats.json                # 全局统计；不要未经转换直接当 openpi norm stats
  meta/tasks.parquet             # 当前 main 文档中的任务映射
  meta/episodes/chunk-000/file-000.parquet  # episode 偏移与元数据
  data/chunk-000/file-000.parquet           # 多 episode 低维样本
  videos/<camera>/chunk-000/file-000.mp4   # 多 episode 的相机帧
```

v3 从“一个 episode 一个文件”变成“**文件中含多个 episode**”；读者根据 **metadata 的 offset** 恢复 episode。这样文件数量更少，适合几十万段采集数据，但容易误判：**视频压缩/Parquet chunk 的物理边界不等于任务边界**；不能靠文件名切训练/验证集。

LeRobot 官方接口示例（**只演示 v3 reader，未在本站运行下载**）：

```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    "owner/dataset-name",
    delta_timestamps={
        "observation.images.wrist_left": [-0.1, 0.0],
        "action": [0.0, 0.1, 0.2, 0.3],
    },
)
s = dataset[42]
# keys/shapes 依创建时 features 而定，不能盲用示例字段
```

`delta_timestamps` 是相对于目标索引时刻的**秒数偏移**，不是“数组索引”。实际 FPS、tolerance、video decoder 的行为与版本绑定。写入须按官方 `add_frame` → `save_episode` → `finalize()` 完成；**不调用 finalize 可能留下未 flush 的 Parquet footer**。v3 `meta/tasks.parquet` 与一些旧版中文博客中的 `tasks.jsonl` 不同，迁移时必须检查已固定版本的真实 schema。

### 5.1 最小训练视图与可扩展侧车

| 训练字段/侧车 | 示例形状或类型 | 来源与防错 |
| --- | --- | --- |
| `observation.images.front` / `wrist_left` | [3,H,W]，每次采样 | 视频帧，原始曝光 metadata 另存 |
| `observation.state` | [D_s] float32 | **机器人**状态或被明确适配的工具状态，不可悄悄混合 |
| `action` | [D_a] float32、chunk 后 [H,D_a] | 监督目标；绝对/相对/关节命令必须可追溯 |
| `timestamp` / `episode_index` | seconds / int | 查询基准、episode 保护 |
| `task` | 文本映射 | 采集工序名称及对象标签 |
| `sample_quality`（自定义） | [H] bool 或枚举 | 丢帧/IK/同步/接触/末端等多维掩码 |
| `provenance.json`（自定义侧车） | manifest | MCAP 哈希、标定 ID、镜像版本、机器人 URDF、时钟拟合 |
| `source_clip_id`（自定义侧车） | string | **源采集片段**级训练/验证隔离 |

自定义侧车**不是 LeRobot 核心格式规定字段**；若接入具体策略，应配置 repack transforms 或自定义 dataset wrapper。不要以“添加列”推断 openpi 模型会使用列。

## 6. 从 v3 到 π₀.₅：版本和动作目标必须分别适配

固定的 [openpi data_loader.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py) 使用旧式 `lerobot.common.datasets.lerobot_dataset` 导入，构造：

$$
\Delta t_k=\frac{k}{f_{\mathrm{dataset}}},\quad k=0,\ldots,H-1.
$$

以 $f=10$ Hz、$H=10$ 为例，target 是 $0,0.1,\dots,0.9$ 秒的一段。它**并不**替我们推断相机-控制器的物理延迟，也**不**校验每个 UMI Action 代表关节角还是 TCP。新版 LeRobot v3 所使用的 API 路径、元数据和读取机制已改变；**不能把“可被 v3 读出”写成“旧版 openpi 无缝支持”**。优先走两条路线之一：

1. **冻结可运行的 openpi / LeRobot 组合**：确认官方 lock/依赖版本，在该版本读取能力支持的数据格式上构造契约；v3 如不兼容，做清晰且可逆的 v3→旧接口导出或数据集 adapter。
2. **显式升级读取层**：让 openpi 的训练 adapter 读取 v3 `LeRobotDataset`，统一键名、tolerance、image decode、episode padding，跑回归测试验证训练/推理一致性；不能仅替换 import。

特别要有 4 项最小回归：`observation` 仅依赖过去；`action` 第 k 帧指向对齐定义的生效时刻；`padding mask` 不跨 episode；`norm_stats/version` 一致且解码可逆。

## 7. CPU 反例：五种假阳性怎样被检测出来？

本站新增独立标准库 Python 合成脚本 [`examples/mcap_lerobot_v3_contract_cpu.py`](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/mcap_lerobot_v3_contract_cpu.py)，不依赖 GPU、MCAP/LeRobot 下载、机器人或相机。运行：

```bash
python3 examples/mcap_lerobot_v3_contract_cpu.py
```

**这是数据契约和负例测试，不是生产转换器。** 脚本执行：①双钟线性拟合与零校正对照；②可见性约束，拒绝“离当前时间最近的未来帧”；③SE(3) 锚定相对动作与逐步 Delta 的区别；④episode 末端 mask，拒绝拼接下一段；⑤按 source clip 防切分泄漏。输出通过标志及数值。没有使用真实录制数据，结果只验证逻辑，不说明本体定位精度。

### 7.1 自己采的 10 段数据如何做真实试验？

**阶段 A（最小闭环）**：锁定左/右 UMI、EGO 双目、手爪、机器人 1 种任务和 10 段采集；存原始 MCAP、相机曝光、Tracker 位姿及版本化外参，建立**事件时间** QC 报告。

**阶段 B（时间轴）**：对每个时钟域拟合 `a,b`、估计控制生效延迟，分别打印 $\mathrm{P95}(|t_{\rm frame}-t|)$、位姿查询年龄、失跟踪比例，标记异常段；人工对照 20 个关键接触事件。

**阶段 C（训练视图）**：为机器人演示建立动作语义 manifest，UMI 经 IK/碰撞过滤之后另建目标动作；只给有效真实/可执行动作计算动作监督。明确生成 v3 与 openpi 已锁定版可读的目标数据形式，做 10 段逐项回放和帧哈希校验。

**阶段 D（验证）**：采集片段级 train/val 隔离，同一工序分别训练「不校时」「仅校时间」「时间+外参+动作/QC」小模型；同时统计离线 loss、闭环成功率、接管率、周期、运动安全。**不**把 GPU 未跑过的方案写成成功率已提升。

## 8. P0 验收门槛、失败模式与局限性

| QC 维度 | 提案级验收约束（**非论文报告值**） | 不达标处置 |
| --- | --- | --- |
| 原始完整性 | MCAP 校验和、schema/seq、topic 清单、缺帧率 | 原始不覆盖；隔离缺损段 |
| 时钟 | 每 clock-domain 独立拟合；P95 与漂移预算记录 | 标记/切段/重新同步 |
| 图像/状态因果 | 推理输入无未来帧；state age 有界 | 无法在线复现者不得作在线输入 |
| 位姿 | 四元数合法、外参版本固定、失跟踪断点不插值 | 标注 mask，不伪造连续轨迹 |
| Action | 明确本体/坐标/命令 vs 测量/相对 vs 绝对/指爪单位 | 不明语义禁止混训 |
| 边界与泄漏 | source clip 分组，不跨 episode 构造 chunk | 重新切分、重建样本索引 |
| 归一化 | 训练集统计、统计版本、逆变换回归 | 拒绝部署 |
| 复现 | 版本锁定、hash、脚本、训练/真机单独报告 | 不宣称完整复现 |

最易忽略的两个局限：**(1) 时钟对齐不是空间标定**——正确时间也无法救错误 $T_{WE}$；**(2) 转成 v3 不是将人变成机器人**——自由手动作仍需运动学约束与真实控制标签，相关性学习不能替代可执行性评测。机器人动态执行延迟会随负载、轨迹、控制器改变，单一静态 $\tau$ 只能作为初始近似。

## 9. 权威来源与可追溯边界

- [MCAP 官方规范：Message、时间字段、Chunk/Index](https://github.com/foxglove/mcap/blob/23d6c627a27ac089d418d9e7f8cdcef9ed86dac1/website/docs/spec/index.md) 与 [Python reader API](https://mcap.dev/docs/python/mcap-apidoc/mcap.reader)：事实是记录格式与查询行为，不包含任何工业 UMI 校时算法。
- [LeRobot v3 官方文档与迁移说明](https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/docs/source/lerobot-dataset-v3.mdx) 和 [源码入口](https://github.com/huggingface/lerobot/blob/d490e472a5d1a207cfe0255caaaf74b6d66e4656/src/lerobot/datasets/lerobot_dataset.py)：事实是 v3 的分块存储、元数据、Reader 与 finalize；演示数据并非本站采集。
- [openpi 固定版本数据加载器](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py) 与 [LIBERO 数据配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)：事实是历史版本的 `delta_timestamps` / transforms；与 v3 的适配层要实际开发、测试。
- [LeRobot 官方动作表示说明](https://huggingface.co/docs/lerobot/v0.6.0/action_representations)：区分锚定相对与逐步 Delta；公式中的 SE(3) 表示和 QC 阈值是本站工程推导，**没有声称这些阈值是论文定论**。
