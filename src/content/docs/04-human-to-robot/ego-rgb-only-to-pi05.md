---
title: "只有 RGB 的 EGO 如何训练 π₀.₅：源码、监督构造和完整闭环"
description: "从 openpi compute_loss 源码解释无动作样本为何不能直接训练 Flow Matching，给出视觉自监督、3D 伪动作、UMI 桥接、EgoEngine 四条路线和 EGO/UMI 工厂验证步骤"
sidebar:
  order: 6
---

> **核验日期：2026-10-27。定位：源码核查 + 可实施设计方案，非本站已跑通的 GPU/真机实验。** 阅读前先看 [三篇 Human-to-Robot 总纲](./human-robot-complete-route/)、[Ego-Pi](./ego-pi-deep/)、[EgoEngine 官方代码审计](./egoengine-code-reproduction/)、[HumanEgo](./humanego/) 和 [UMI-Bridge](./umi-bridge/)。
>
> **一句话结论**：纯 RGB EGO 数据只有观测，缺失 Flow Matching 要监督的未来动作。现有 openpi 的 π₀.₅ **不能把纯 RGB 视频直接丢入标准训练器，便自动学到动作**。必须先选择一种显式监督构造策略，或修改训练目标，让视频只更新视觉/时序表征而不假装存在机器人动作。

![RGB-only EGO 向 π₀.₅ 提供有效监督的四条路线](/Embodied-AI-Roadmap/diagrams/ego-rgb-only-training-routes.svg)

*图 1｜A：仅训练视觉/语义；B：人手运动形成新动作域；C：UMI 作为动作锚；D：生成可执行机器人轨迹。四者的数据成本、监督位置、可执行性不同，不应混称「EGO 直接做 VLA 后训练」。*

## 0. 先从一个具体场景出发

你录了 1 小时「工人把不同颜色零件放入两个料框」的头戴 RGB，摄像机 30 fps。视频包含：人看见了哪个零件、按什么顺序拿、放在哪里、什么时候抓取、是否完成。但如果没有 6DoF 手腕位姿、手指动作、机器人的状态或动作，**视频不包含一个可直接当作机器人 joint/EE action 的向量**。

训练目标有两个不同层次：

1. **知道做什么**：看图判断当前是哪类零件、下一步先抓哪个、什么是完成状态。RGB 很适合。
2. **知道怎样执行**：机器人在时刻 \(t\) 应发送哪个夹爪控制量、末端增量、关节位置。光有 RGB 无法保证唯一确定它，尤其受遮挡、尺度和本体差异影响。

Ego-Pi 的重点是**把人类示范里的新规则迁移到已经掌握基础操作的机器人**，而不是证明无动作 RGB 自动变成机器人控制器。官方项目页明确提到其机器人已拥有基础低层操作能力，人类数据负责组合与规则。[Ego-Pi 作者项目](https://egopipaper.github.io/)。

## 1. 直接审计 openpi：为什么报错或伪标签会误导？

一手入口：

- [openpi 项目 README](https://github.com/Physical-Intelligence/openpi)：当前公开训练实现说明，π₀.₅ **仅支持 Flow Matching head 的训练与推理**。
- [JAX π₀ / π₀.₅ 核心](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)：查 <code>compute_loss</code>、<code>embed_prefix</code>、<code>embed_suffix</code>。
- [训练 DataConfig](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)：查 <code>action_sequence_keys</code>、<code>data_transforms</code>、<code>model_transforms</code> 和 normalization。

### 1.1 真值动作出现在损失构造的哪里？

官方 JAX <code>Pi0.compute_loss</code> 的主要输入是 <code>observation</code> 和 <code>actions</code>。令动作块 \(A\in\mathbb{R}^{B\times H\times D}\)，高斯噪声 \(\epsilon\) 同维，官方约定：

$$
x_\tau=\tau\epsilon+(1-\tau)A,\qquad
u_\tau=\epsilon-A.
$$

网络接收 \(x_\tau,\tau,o\)，输出 \(v_\theta\)，优化：

$$
\mathcal{L}_{FM}=
\mathbb{E}_{o,A,\epsilon,\tau}
\left[\frac{1}{HD}\left\|v_\theta(x_\tau,\tau,o)-(\epsilon-A)\right\|_F^2\right].
$$

**重要实现细节**：上述 \(HD\) 平均是讲解总标量的等价概念；当前源文件自身先对每个动作 token 的最后一维求 mean，返回形如 \([B,H]\) 的损失，训练器后续再聚合。该代码抽样 \(\tau\) 使用 Beta(1.5, 1) 并限制到接近 \((0,1)\) 的区间，而不等价于教学中常见的均匀采样。别把公式里的 \(t\)（噪声时间）错认为视频的帧时间。

**极简数值演算**：若某一维动作 \(A=0.8\)，噪声 \(\epsilon=0.2\)，\(\tau=0.3\)，那么 \(x_\tau=0.62\)，应预测的速度 \(u_\tau=-0.6\)。若 A 根本不存在，这两个值都无法构造。为不存在的动作填零，则模型会训练成「输出零动作」，不等于在无监督学习。

### 1.2 图像前缀参与计算，为什么不能说「视觉也有 Loss」？

在 <code>embed_prefix</code> 里，图像经 SigLIP/PaliGemma 视觉组件编码，文本经 token embedding 进入前缀；动作与流时间进入动作专家后缀。虽然动作损失的梯度可以通过注意力影响前缀中的可训练权重，**这不等于当前训练器还有独立的图像重建、下一帧预测、文本生成 CE Loss**。

如果希望纯 RGB 样本也训练视觉表征，需要**自行加辅助任务、正确构造可用标签、连接 optimizer，并验证梯度确实回传**。仅在 Dataset 里加一个 image 字段不会自动产生监督。

### 1.3 三个错误做法与反例

| 错误操作 | 为什么不成立 | 应怎样处理 |
| --- | --- | --- |
| 无动作样本的 actions 填零 | FM 会把假零动作当 GT，污染动作分布 | 该样本不计算动作 FM；额外构造可验证的其他监督 |
| 把人的关节角直接写进机器人的关节维 | 人/机器人的坐标轴、关节顺序和运动学不同 | 单独做人类动作域，或严格重定向后才称机器人动作 |
| 把手腕图像光流当末端 3D 位移 | 像素位移混有头部运动、深度变化、遮挡 | 先校准相机、估计 3D 位姿及可信度，再转换坐标 |

### 1.4 Mask 为什么必须拆成两种？

<code>valid_frame_mask</code> 表示一帧能否追踪；<code>action_supervision_mask</code> 表示是否**真的有某个动作维度的有效监督**。两者不能共用。

对于不同机器人动作容器，单独引入 \(M\in\{0,1\}^{B\times H\times D}\) 可以定义**研究性扩展**：

$$
\mathcal{L}_{FM,masked}=
\frac{\sum_{b,h,d} M_{bhd}\,[v_\theta-u]_{bhd}^2}
{\max(1,\sum_{b,h,d}M_{bhd})}.
$$

这个 masked loss **不是**上面查阅的原版 JAX <code>compute_loss</code> 已实现的功能，需自己修改并核对训练器聚合方式。无动作 RGB 样本 \(M=0\) 时，其 FM loss 应被跳过，仍需另一条监督分支才能产生梯度。这里的分母保护只避免除零，不使全零样本 magically 成为有效训练数据。

## 2. 四条路线的真实监督在哪里产生？

| 路线 | EGO 最小输入 | 新建什么监督 | 能更新什么 | 还依赖什么 |
| --- | --- | --- | --- | --- |
| A 视觉/语义辅助学习 | RGB+时间戳 | 阶段、目标图、顺序或时序预测任务 | 视觉/条件表征或辅助头 | 自己改训练器；最终还需机器人动作数据 |
| B 人体 3D 动作伪标签 | RGB+相机/手/物体 3D 估计 | 人类手腕/物体中心相对动作 | human embodiment 动作支路或跨域表示 | 可靠标定、追踪置信度、跨本体训练基础 |
| C UMI 动作锚定 | EGO RGB + 配对 UMI | 已知夹爪开合、末端 6DoF 与视觉配对 | 视觉-动作潜空间，后续机器人策略 | UMI 同步录制、机器人数据或可执行性验证 |
| D EgoEngine 机器人化 | RGB + 3D 重建/仿真资产 | \(\tilde I^r_t,\tilde A^r_t\) | 真正的 robot-domain FM / imitation policy | 几何、仿真、动作重定向、视觉转换、质检 |

**不存在免费的转化**：从一段普通 RGB 推出完整双臂高频动作，需要观测中原本没有的 3D 几何、机器人运动学、先验或附加标签。模型可以估计，但要将**估计误差、是否可执行、监督置信度**一起记录。

## 3. 路线 A：只有 RGB，不追求立即预测机器人动作

### 3.1 你究竟能学什么？

在固定的采集任务中构造「当前画面 \(I_t\)」与「将来画面 \(I_{t+\Delta}\)」的配对。例如料框中的零件数量发生变化、抓取阶段从接近进入抬起，或视频片段的时间顺序。可以做：

**阶段分类**：标签 \(y_t\in\{\text{approach,grasp,move,place,release}\}\)，定义

$$
\mathcal L_{\mathrm{phase}}=
-\sum_{c} \mathbf 1[y_t=c]\log p_\phi(c\mid I_{t-k:t}).
$$

**未来视觉表征预测**：以固定视觉教师 \(f_T\) 抽未来帧特征，预测头从当前条件估计目标：

$$
\mathcal L_{\mathrm{future}}=
\left\|g_\phi(f_\theta(I_{t-k:t}),\ell,\Delta)
-\mathrm{stopgrad}(f_T(I_{t+\Delta}))\right\|_2^2.
$$

**顺序/成功事件监督**：给相邻视频片段配上阶段推进或者是否完成标签。对于「先装 A 后装 B」，需要同时看物体状态变化与任务文字，不应只让模型根据背景区分前后。

上述都是**工程上可构造的候选损失，不是 openpi π₀.₅ 官方现有 RGB-only 训练入口，也不是三篇论文逐项采用的固定配方**。

### 3.2 为什么它不会直接训练动作专家？

如果辅助分支只连接视觉编码器/新头，其梯度沿

~~~text
L_future / L_phase
    → 辅助头
    → 视觉特征融合层（如解冻）
    → 视觉 backbone（如解冻）
~~~

传播，不自动通过原版 <code>Pi0.compute_loss</code> 更新动作速度头。要让策略受益，必须在之后**用有 action 的机器人样本继续微调**，验证动作成功率是否超过只用机器人数据的基线。视觉指标提升不能替代这一验证。

### 3.3 视频切片与泄漏防范

按采集 episode 或工位划分训练/验证/测试，**不能相邻帧随机分到不同集合**。同一动作几十帧几乎重复，这样划分可能使视觉预测分数虚高。未来视觉监督只允许在训练时使用 \(I_{t+\Delta}\)；推理时不应把真值未来帧当条件。

## 4. 路线 B：重建人类 3D 动作，再做 Human-to-Robot Transfer

[HumanEgo 官网](https://humanego-ai.github.io/) 清楚说明其结构化数据依靠 Aria MPS 的 SLAM、手部 3D 姿态和同步 RGB；它描述的是**从 raw RGB 经过上游感知重建**的管线，不是 FM 不需要动作标签。Ego-Pi 也并非直接把只有像素的视频作为没有动作的 FM 样本。

### 4.1 EGO 的坐标和帧同步从哪里来？

定义 \(T_{A\leftarrow B}\) 将 B 系坐标表示转换到 A 系。原始 EGO SLAM 帧为 S，桌面固定 marker 定义世界系 W，相机系为 C，手系为 H，物体系为 O。若估计 \(T_{W\leftarrow S}\)、 \(T_{S\leftarrow C}(t)\) 和 \(T_{C\leftarrow H}(t)\)，则

$$
T_{W\leftarrow H}(t)=
T_{W\leftarrow S}\,T_{S\leftarrow C}(t)\,T_{C\leftarrow H}(t).
$$

对于桌面 AprilTag 做世界锚定，**需要测的是固定 marker 世界坐标、相机内参（鱼眼必须按正确模型）、相机与 IMU 外参、相机与手的关系**。不能把 VIO 每次启动的随机世界原点等同桌面机器人基座原点。

**时间误差**：图像 \(I_t\) 与姿态 \(\hat T(t+\delta)\) 若不同步，接触瞬间会出现错误监督。若手速为 \(0.5\,m/s\)，偏差 \(20\,ms\) 就意味着约 \(10\,mm\) 纯平移误差，这是由速度×时间得出的示例量级，尚未计姿态与跟踪误差。

### 4.2 相对物体动作比绝对轨迹更可迁移

定义手相对物体的姿态：

$$
T_{O\leftarrow H}(t)=
T_{W\leftarrow O}(t)^{-1}T_{W\leftarrow H}(t).
$$

构造人类交互伪动作的一种**研究性设计**：

$$
a^{h}_t=
\left[
\mathrm{Log}\!\left(T_{W\leftarrow H}(t)^{-1}T_{W\leftarrow H}(t+\Delta)\right),
g_t,
\mathrm{Log}\!\left(T_{O\leftarrow H}(t)^{-1}T_{O\leftarrow H}(t+\Delta)\right)
\right].
$$

前三项分别是腕部 6D 增量、估计的抓握/释放代理 \(g_t\)、物体参考系中的相对运动。**这不是任何官方论文逐字公开的标准 action schema**；它只是明确表示如何把需要监督的量从数据中构造。真实使用必须指定旋转 Log 的参考系、单位、时间间隔、夹爪语义，以及哪些维度被 mask。

相对量能减少坐标系差异，不会自动解决人手/双指夹爪的接触物理差异。比如人可以用五指搓动螺母，双指夹爪可能做不到。

### 4.3 别把「人类动作」直接写成「机器人动作」

对于人体示范 $A^h$，选择**human embodiment** 输入标签，只监督对应 human 动作头/适配器；机器人动作头用已有 \(A^r\) 数据监督。联合损失概念：

$$
\mathcal L_{\mathrm{joint}}=
\lambda_r\mathbb E_{D_r}\mathcal L_{FM}^{robot}
+\lambda_h\mathbb E_{D_h}\mathcal L_{FM}^{human}
+\lambda_s\mathcal L_{\mathrm{semantic}}.
$$

必须用域标识、不同的归一化/输出映射处理 \(\mathbb R^{D_h}\) 和 \(\mathbb R^{D_r}\)。若固定 openpi action container 32 维，两域分别映射到容器并 pad；**pad 只解决张量大小，不代表物理等价，缺失维不能当真实动作评价**。上述联合目标同样是研究性扩展；官方 openpi README 并没有交付可直接复用的 Ego-Pi 人/机器人完整混合训练配置。

### 4.4 谁能承担这个路线？

若团队目前只有未标定单目 RGB，而且每帧手部大面积被遮挡，这条路线的质量风险最高。若已经有双目鱼眼+IMU 与固定 AprilTag，或已提供 UMI Wrist 6DoF，则可以优先尝试世界系轨迹平滑、相对位姿和接触阶段。**一个高质量 3D 小数据集可能比十万小时不知尺度的 RGB 更适合第一轮动作闭环**。

## 5. 路线 C：让 UMI 充当 EGO 与机器人之间的动作锚

这里研究的问题不是「UMI 数据是不是 EGO 数据」，而是：当同一动作同时被头部和腕部相机看到，且腕部夹爪有位姿/开合时，能否让 Head EGO 特征学到与真实末端动作相关的表征？

建议采集

~~~text
同一 Episode ID
├── Head EGO RGB + timestamp + 相机6DoF
├── Wrist UMI RGB + timestamp
├── UMI 夹爪 world 6DoF + openness + button
├── 任务文本 / 阶段 / 质量标签
└── 相机-夹爪外参 / 标定版本
~~~

先训练腕部视觉→动作辅助头，再用跨视角配对约束 Head 视觉编码出动作相关信息。例如设 \(z_H,z_W\) 为 head/wrist 特征：

$$
\mathcal L_{\mathrm{align}}=
-\log\frac{\exp(\operatorname{sim}(z_H,z_W^+)/\tau_c)}
{\sum_j\exp(\operatorname{sim}(z_H,z_{W,j})/\tau_c)}.
$$

需包含**同一时刻同一动作正样本与不同动作负样本**，否则对齐可能只学到同一桌面、同一操作者或同一背景。这是 InfoNCE 类研究性候选损失，不是引用 UMI-Bridge 的原版精确公式。

然后将这种表征用于有真实机器人动作的后训练；如果没有 robot action，至少还能用 UMI 末端轨迹做动作空间可行性映射与仿真验证。详见 [UMI-Bridge 教学解读](./umi-bridge/)；其具体代码开放状态需进一步核验。

## 6. 路线 D：EgoEngine 把人类视频变成配对机器人数据

EgoEngine 的输出不只是 robot-looking video，而是**视频与动作同一时间索引的可执行配对**：

$$
\tilde D_r=
\left\{(\tilde I_t^r,\tilde s_t^r,\tilde A_t^r,\ell,t)\right\}_{t=1}^T.
$$

公式用于表达训练数据接口，非原论文逐字数据类型。其工程管线：

~~~text
原始 EGO RGB
  → 人手/物体/相机估计
  → 物体/机器人几何 + 相机轨迹（prepared input）
  ├─ 动作分支：重定向 → Replay → MPC → 必要时 RL
  └─ 视觉分支：人体区域处理 → 机器人渲染 → 遮挡合成
  → 按同一参考轨迹与时钟合并 RGB + action
  → HDF5 质检
  → 机器人策略训练 → rollout
~~~

**复现边界**：其[官方开源仓库](https://github.com/GaTech-RL2/EgoEngine-CoRL2026)公开 prepared-input 开始的部分命令，原始 RGB 的完整重建感知链路和下游完整 π₀.₅ 训练并非一键全开放；本站已有 [逐命令、文件和 Replay/MPC/RL 审计](./egoengine-code-reproduction/)，不要重复声称把这套流程全部跑通。

**关键数据质量门槛**：生成动作要过 FK/碰撞与末端可达性检查，视频与动作要过 timestamp/遮挡与抓放事件一致性检查，失败窗口要拒绝或降权。不应仅因为合成视频视觉漂亮就标记为高质量数据。

## 7. 数据契约：先让数据管线不会骗你

推荐保存两个层级：

**L0 原始层（不可覆盖）**：

~~~json
{
  "episode_id": "sort_parts_0001",
  "source": "ego_rgb",
  "task_text": "sort red and blue parts",
  "camera_serial": "ego_stereo_v1",
  "camera_model": "fisheye",
  "calibration_version": "calib_2026_10",
  "rgb_paths": ["frames/000000.jpg", "frames/000001.jpg"],
  "image_timestamp_ns": [1000000000, 1033333333],
  "split_group": "fixture_a_operator_01",
  "robot_action_available": false
}
~~~

**L1 派生监督层**（输出型式建议；具体与训练平台 schema 适配）：

| 键名 | 形状或定义 | 有效性 |
|---|---|---|
| <code>camera_T_world</code> | 逐帧 4×4，实际变量名注明方向 | VIO quality、marker anchor |
| <code>hand_T_world</code> | \([T,2,4,4]\) | 逐手置信度 |
| <code>object_T_world</code> | \([T,N_o,4,4]\) | object visible、track confidence |
| <code>grasp_event</code> | \([T,2]\) | 接触事件人工抽检 |
| <code>human_pseudo_action</code> | \([T,D_h]\) | 每维 mask + provenance |
| <code>robot_action</code> | \([T,D_r]\) | 不存在则**不创建伪零值** |
| <code>action_source</code> | human_estimate / umi / robot_real / robot_synth | 数据来源清楚 |
| <code>action_valid_mask</code> | 与对应动作同形 | 缺维不参与 loss |
| <code>sync_error_ms</code> | 每 episode + 分位统计 | 接触段单独统计 |
| <code>label_confidence</code> | 明确定义打分来源 | 不能等同真实性 |

如果实际只采视频，则 L1 可以只有阶段、目标帧和检测特征，绝不能因数据 schema 要求机器人动作就创建假标签。若训练平台要求固定 <code>actions</code> 字段，应该**按数据源路由到不同 Dataset/Trainer**，而不是把假动作送进默认 Pi0 trainer。

### 7.1 三种「0」必须严格区分

1. 真实动作恰好为 0：**有效监督**，可以计算 loss。
2. 图像不可见或手势跟踪失败：**标签缺失**，必须 mask。
3. 此样本没有机器人动作：**监督任务不适用**，不能填 0 强行计算动作 loss。

### 7.2 坐标与同步的验收顺序

1. 相机内参：重投影误差、鱼眼边缘误差与画面畸变；必须单独看周边区域。
2. 相机/IMU 外参：动态旋转与时延；只标静态相机不足以约束 IMU 关系。
3. 世界锚：固定 marker 重复测量世界原点稳定性，世界坐标和机器人基座外参分开估计。
4. Hand/Object：跟踪漂移、失跟比例、接触帧误差。
5. Robot action：重定向后的 FK 误差、关节限位、碰撞和控制频率。
6. 同步：对物体释放、抓取接触的事件做人工抽样检查；不能仅凭视频总帧数相同判断已同步。

## 8. 一个两阶段「先跑通再扩规模」的实操 Demo

**建议任务：桌面按颜色分拣零件**。EGO 人类演示包含新规则，机器人仅提供「拿起零件→放入指定料框」基础动作演示。这样能区分**规则迁移成功**与**单纯视觉分类成功**，也较容易人工复位。

### 阶段一：不用 GPU 验证数据和监督（建议 1–2 周的里程碑，不是耗时承诺）

**Step 1｜固定采集条件**：记录双目鱼眼参数、IMU 时间戳、AprilTag 布局、RGB 图像、工件和任务版本。至少制作少量不同位置、不同光照、不同操作者的 episode。

**Step 2｜把原始视频按 Episode 可视化**：在视频上同步绘制 hand/object tracks、阶段标签和时间轴。没有 3D 追踪则先用阶段标签完成路线 A；不要冒充得到动作监督。

**Step 3｜抽检 10 条质量**：记录世界系原点的漂移、失跟帧比例、末端反投影误差、抓取/释放事件时间偏差。数据不合格停在采集/重建，不要继续训动作。

**Step 4｜先验证机器人动作数据接口**：用已有 LeRobot/机器人轨迹搭建真实 action chunk，确认 action 单位、归一化、控制器映射；在小数据上 overfit 到低误差，再做 rollout。优先验证 openpi 默认的数据预处理链是否适配本体。

**Step 5｜数据可行性闸门**：若只能得到 RGB+阶段标签，走 A；如果可信手腕 3D，走 B；如果有配对 UMI，优先 C；如果有数字孪生与仿真资源，评估 D。没有 action 的训练样本应隔离于 robot FM。

### 阶段二：带动作数据后的 π₀.₅ 后训练

建议顺序：

~~~text
baseline_0: 原始机器人数据 → π0.5 LoRA/Full fine-tune → 受控 rollout
experiment_A: robot baseline + EGO 视觉辅助目标（需改代码）
experiment_B: robot baseline + 带动作伪标签 human 域样本
experiment_C: robot baseline + UMI 跨视角动作表征
experiment_D: robot baseline + 经可执行验证的 synthetic robot demos
~~~

每组都固定机器人演示数量、训练 tokens/steps、evaluation random seed、场景拆分。**同场景训练收益和新场景迁移收益必须分开报告**。先完成 baseline，避免任意新方案的收益来自训练步数多或刚好评测了更简单的布局。

### 8.1 JAX / openpi 最小源码改造定位

这是**阅读代码的工作单**而非未经验证的补丁：

| 目标 | 从哪里开始查 | 新增逻辑 | 核验标准 |
| --- | --- | --- | --- |
| 接机器人真值 | <code>training/config.py</code> 的 DataConfig | LeRobot schema、action_sequence_keys、data transforms | action 与图像时间一致 |
| π₀.₅ FM | <code>models/pi0.py</code> 的 compute_loss | 沿用官方 action/noise/time 构造 | 单条 robot 样本能算有限 loss |
| RGB only | 新 auxiliary trainer / batch router | 不调用没有动作的原始 compute_loss | 无动作样本不参与 FM |
| Human pseudo action | 独立 Dataset + normalizer | embedding 域标签/掩码/动作头适配 | 验证 pad 维不参与监督 |
| 多教师或语义目标 | 在 prefix 上加 head 并明确连接梯度 | 未来语义或视觉表示目标 | 梯度非零且不破坏机器人 BC |
| 真机部署 | policy inference + 本体控制 | 反归一化、限幅、频率与缓存 | action semantic 与控制端一一对应 |

官方 openpi README 报告 LoRA 微调显存需求 **大于 22.5 GB**、全量训练 **大于 70 GB**（其参考环境），真实所需会随模型和配置变化；实际部署前按 checkpoint 的 config 与 GPU 余量重新测。不要因为只有 RGB 就误认为显存需求消失。

### 8.2 离线检查与真正闭环有什么区别？

离线必须有：Loss 曲线、动作/伪标签分布、有效 mask 比例、世界系反投影、训练/测试样本互斥。仿真必须有：动作可达率、碰撞率、任务完成率。真机必须有：急停、速度限制、工装隔离、初始化一致、任务成功率和置信区间、干预次数、接触异常。

**单独报告数据转化漏斗**：原始 EGO 小时 → 有效同步小时 → 有 3D 追踪小时 → 可重定向小时 → 仿真可执行小时 → 真实策略可提升的小时。最后一项不能由前面数据量直接推导。这样才能知道数据工厂应该提升采集量、定位精度还是动作生成有效率。

## 9. 一个深入的训练梯度问题

设模型的 Prefix 参数为 \(\theta_P\)，动作专家为 \(\theta_A\)。只使用机器人 FM 时

$$
\nabla_{\theta_P}\mathcal L_{FM}^{r}
\quad\text{和}\quad
\nabla_{\theta_A}\mathcal L_{FM}^{r}
$$

可能都非零（依冻结策略而定），因为动作查询视觉上下文；但对 **RGB-only 的未标注样本**，原始动作 loss 不存在。如果引入视频损失 \(\mathcal L_{video}\)，联合目标写成

$$
\mathcal L=\mathcal L_{FM}^{r}
+\lambda_v\mathcal L_{video}^{h}
+\lambda_x\mathcal L_{cross-domain}.
$$

新增任务的梯度应明确注入哪几层。**如果两种监督在同一参数上方向冲突，简单加权可能同时伤害动作与视觉能力**。应按任务统计共享前缀参数梯度余弦，以及基础 robot-only 成功率是否退化；必要时冻结部分视觉主干、分阶段训练或隔离投影头。这里是面向实际训练稳定性的工程建议，不意味着 Ego-Pi 原论文采用上述三项梯度方案。

## 10. 最终决策：哪条路最值得优先做？

在已经有 EGO 双目/IMU、UMI 腕部位姿与夹爪开合、可用真实机器人的团队里，**优先级建议 C → B → A（并行）→ D（后续）**：

**先用同步 UMI 建立可信动作锚和简单 robot baseline**，因为这最快能发现数据平台真正缺什么；把 Head EGO 的阶段/视觉子目标和 human 3D pose 做辅助训练；只有在明确目标任务可复用、资产齐全时才投资高成本机器人化视觉生成。

若实际上只有大量 RGB、暂时无法采 UMI 与机器人数据，就先做 A（任务语义/时间段标注/目标帧）。这会改善数据资产，但**不要对外汇报「已经训练得到可部署 π₀.₅ 动作策略」**。

## 11. 直接核对的一手资料（按验证层级）

- [openpi README：π₀.₅ Flow Matching 头开放范围与资源](https://github.com/Physical-Intelligence/openpi)
- [openpi π₀/π₀.₅ JAX loss、Prefix/Suffix 源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)
- [openpi 数据训练接口](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
- [Ego-Pi 官方项目：已有基础技能、50:50 数据混合、58D 双手交错](https://egopipaper.github.io/)
- [Emergence of Human-to-Robot Transfer：预训练多样性条件](https://vision.ics.uci.edu/papers/emergence-of-human-to-robot-transfer-in-vision-2026/)
- [HumanEgo：Aria SLAM、手部轨迹、Interaction-Centric Tokens](https://humanego-ai.github.io/)
- [EgoEngine 作者项目：机器人视觉+可执行动作双分支](https://egoengine.github.io/)
- [EgoEngine 已公开的 prepared-input 代码](https://github.com/GaTech-RL2/EgoEngine-CoRL2026)
- [UMI-Bridge 原论文](https://arxiv.org/abs/2609.18232)

**阅读/复现状态**：本章公式中保留的 openpi FM 是核对源码的实际形式；标注为**研究性设计、候选损失、数据契约、示范实验**的部分均是可验证建议，不是作者官方训练配方；站点尚未在用户设备上执行本章提出的新策略微调或真机 rollout。
