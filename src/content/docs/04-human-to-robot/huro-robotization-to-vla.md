---
title: "HuRo 深度实战：EGO 视频机器人化、LeRobot 与 VLA 预训练"
description: "CoRL 2026 HuRo 十阶段源码审计：相机/手部、IK、视觉合成、动作语义、数据契约、训练评测及工业 EGO/UMI 复现"
sidebar:
  order: 10
---

> **核对时间：2026-11-03。** 一手来源：[论文 arXiv 2609.10706](https://arxiv.org/abs/2609.10706) / [作者实验项目](https://3587jjh.github.io/HuRo/) / [官方仓库](https://github.com/3587jjh/HuRo) / [十阶段代码说明](https://github.com/3587jjh/HuRo/blob/main/pipeline/README.md) / [Parquet 和 LeRobot 精确字段](https://github.com/3587jjh/HuRo/blob/main/examples/README.md) / [机器人配置](https://github.com/3587jjh/HuRo/blob/main/configs/README.md) / [环境与许可](https://github.com/3587jjh/HuRo/blob/main/setup/README.md)。
>
> **真实发布边界**：机器人化 Pipeline 于 2026-09-11 开放；仓库 README 的 **HuRo Dataset 仍标为 Coming soon**。作者开源的十阶段脚本**不包括完整的 VLA 预训练/真机 rollout 一键复现**。本站的 CPU 验证脚本只验证数学与数据语义，没有执行 GPU/Isaac Sim/真机实验。

# HuRo：从人类 EGO 视频生产机器人训练对

![HuRo EGO 视频机器人化系统流程](/Embodied-AI-Roadmap/diagrams/huro-system.svg)

*图 1：本站根据作者 Pipeline README 重绘，展示人体标注（蓝）、IK 动作转换（橙）、视觉合成（紫）及下游 VLA（绿）。下游训练属于论文实验、**不是**开源 Pipeline 的第 11 个官方 stage。*

## 0. 先确定技术路线和边界

想让“只有 RGB/双目鱼眼视频的 EGO 数据”变成可训练 VLA 的样本，必须解决两种 domain gap：

**Observation gap**：训练视频看到人的手，但机器人部署时看到机械臂/手。**Action gap**：人手 21 关键点和 MANO 姿态不等于机器人关节命令，也未保证动力学与接触可执行。HuRo 用同一片段同时生成目标本体观察 $\tilde o_t$ 和目标本体动作 $\tilde a_t$，构造 $(\tilde o_t,\ell,\tilde a_t)$。

| 路线 | 人类侧监督形式 | 机器人动作从哪里来 | 应用位置 |
| --- | --- | --- | --- |
| [Ego-Pi](./ego-pi-deep/) | 人类与机器人各自带动作监督 | 两个 embodiment 共同微调 | 大底座上的新规则迁移 |
| [Human-to-Robot Emergence](./human-robot-emergence-deep/) | 人手 3D 位置 | 底座多本体知识 + robot 数据 | 预训练多样性门槛 |
| [EgoEngine](./egoengine-code-reproduction/) | 重建的几何/相机/交互 | 任务级物理仿真与轨迹优化 | 高保真目标任务示范 |
| **HuRo** | 普通第一视角视频，先估手和相机注释 | **大规模 IK 重定向并生成可执行关节表示** | **规模化 robotized 预训练** |
| [WING](./wing-ego-spectral-transfer/) | RGB 潜在交互结构 | 机器人侧真实动作 | RGB 辅助引导动作学习 |

**不能把“robotized action”说成已经通过真机动力学验证的安全动作**：几何重定向解决的是运动学表征，可执行性还必须检查接触、关节限制、碰撞与控制器实际跟踪性能。

## 1. 论文核心结果与实验证据

作者报告对 **5 个人类数据源**进行转换，共约 **630K episodes / 142M frames / 1,317h（30fps）**，来源帧数占比 EgoDex 55%、EgoVerse 27%、Ego4D 10%、Ego10K 6%、EPIC-Kitchens 2%。

在四项 ALLEX 真实机器人任务（苹果拿放、杯子堆叠、方便面杯交接、微波炉装载）中，**预训练后还进行下游真机微调**：整体成功完成率随 HuRo 数据规模从 51.5% 升到 80.3%，OOD 从 34.9% 升到 72.2%。数值是[作者项目页面](https://3587jjh.github.io/HuRo/)的实验，**不是本站的独立复现、也不是工业流水线部署指标**。

两个特别重要的消融：

- **视觉转换是否必要**：同来源、相同动作监督下，仅 10% overlay 的版本 OOD 59.5%，完全无 overlay 55.7%；显示观察域对齐确有收益，但不等于 overlay 比例越高越好。
- **是不是只用视觉特征就够了**：视觉通路预训练但重建动作头，弱于用 robotized observation + retargeted action 的端到端预训练。说明可迁移的不仅是图像语义，还有连续动作条件映射。
- **是否只是帧数更大**：单来源 EgoDex 78.9M 帧，ID/OOD 63.9%/54.2%；多来源 Mixed 50% 71.1M 帧，ID/OOD 78.2%/69.8%。数据多样性（动词、物体、组合）有不可替代的贡献。

这些比较要在论文对应的相同任务协议内解读；不同评测表格不能随意混成一个排行榜。

## 2. 官方十阶段源码：逐级输入输出与排错

[官方 pipeline/README](https://github.com/3587jjh/HuRo/blob/main/pipeline/README.md)将任务分成 **1–7 人体域标注**、**8–9 本体转换**和 **10 训练数据写盘**：

| # | 真实脚本 | 后端与功能 | 关键输出 |
| --- | --- | --- | --- |
| 1 | `stage1_annot_intrinsics.py` | DroidCalib，AnyCalib 备用：相机内参/畸变 | `fx,fy,cx,cy,xi,H,W,model` |
| 2 | `stage2_annot_contact.py` | 100DoH Faster R-CNN：手检测、左右猜测 | 逐帧手框、side/conf |
| 3 | `stage3_annot_contact_refine.py` | BoT-SORT：手身份时序一致性 | 稳定左右手分配 |
| 4 | `stage4_annot_hand.py` | HAWOR：21 点 3D、MANO、mask | 相机坐标系手姿 |
| 5 | `stage5_annot_extrinsics.py` | DROID-SLAM、MoGe-2、GeoCalib | 米制、重力对齐的 cam-to-world |
| 6 | `stage6_annot_narr.py` | ViTDet-H、Qwen3.5-9B | 分割操作片段和语言指令 |
| 7 | `stage7_annot_inpaint.py` | ViTDet-H、SAM2、ProPainter | 人臂 mask / 修复背景 |
| 8 | `stage8_robot_retarget.py` | PyRoKi IK/JAX：机器人重定向 | `state_qpos`、EEF、误差诊断、cam-to-base |
| 9 | `stage9_robot_overlay.py` | Isaac Sim 渲染目标机器人的手臂 | overlay 视频、`overlay_valid` |
| 10 | `stage10_lerobot_convert.py` | 数据转换器：每片段一个 episode | **LeRobot V2.0**，视频+state/action+语言 |

原始视频并不需要人工裁剪：Stage 6 会找操作片段，也可能因检测到三人以上等规则丢弃样本。Stage 1 标定失败的 clip 可能写空 JSON；有些 Stage 对**跳过**的 clip 也写 `.done` marker，**不能把 done marker 数量当有效视频数量**。

### 2.1 为什么双目鱼眼数据一定要保存内参与曝光时间戳？

视频中的像素不等于三维空间坐标。在去畸变针孔模型下：

$$
\lambda\begin{bmatrix}u\\v\\1\end{bmatrix}
=K\begin{bmatrix}X_c\\Y_c\\Z_c\end{bmatrix}.
$$

对固定深度的横向估计，若像素误差为 $\delta u$，近似：

$$
\delta X\approx\frac{Z_c}{f_x}\,\delta u.
$$

**教学例子**：$Z_c=0.6$m、$f_x=400$px、$\delta u=4$px 约产生 6mm 横向误差。这还没计入手追踪、SLAM、相机/世界对齐和时间偏移的误差。工业 EGO 应继续保留原始分辨率、硬件时间戳、IMU、左右镜头内外参和标定版本，30fps/256 短边只是模型推理前处理，不是原始数据存档标准。

### 2.2 世界/机器人坐标链必须完整

定义 $T_{a\leftarrow b}$ 是 **b 系到 a 系**的齐次变换，列向量左乘。HuRo 的 `cam_pose` 是 cam-to-world（OpenCV 光学相机轴，米制/重力对齐），`cam_pose_base` 是 cam-to-robot-base。若机器人基座 b 和世界 w 的变换可知：

$$
\mathbf p_b(t)=T_{b\leftarrow w}\,
T_{w\leftarrow c}(t)
\begin{bmatrix}\mathbf p_c(t)\\1\end{bmatrix}.
$$

**不要**把 `T_world_cam`、`T_cam_world`、`T_base_cam` 三者直接通过改变量名混用。机器人 base 为 x 前、y 左、z 上；OpenCV 相机为 x 右、y 下、z 前。HuRo 的左右 MANO wrist 坐标轴也不同：左手腕的 +x 沿 wrist→finger，右手沿反方向；双手 +y 指向手背、+z 指向拇指侧。自研 UMI 若输出 Tracker Pose，必须明确原点、手掌轴、TCP 外参，否则 IK 结果会出现系统性偏移。

### 2.3 动作重定向原理：优化的是几何匹配而不是数值映射

人体 21 个关键点经帧坐标变换、相机尺度估计后形成参考目标 $\hat p^h_{t,j}$；机器人关节 $q_t$ 决定每个指尖/腕链接位置 $\operatorname{FK}_j(q_t)$。

一个**本站教学等价问题（不是作者原版完整 loss）**：

$$
\min_{q_{1:T}}\sum_{t,j}w_j
\left\|\operatorname{FK}_j(q_t)-\hat p^h_{t,j}\right\|_2^2
+\lambda_s\sum_t\|q_{t+1}-2q_t+q_{t-1}\|_2^2
+\lambda_l L_{\mathrm{limit}}(q).
$$

损失分别控制几何对应、时序平滑、关节限位。几何可达不意味着接触稳定，实物工况仍需检测自碰撞/环境碰撞和控制力学。

官方 Stage 8 的诊断列 `retarget_cost`, `err_tip_mm_*`, `err_palm_mm_*`, `err_local_dir_*`, `err_ddq`, `err_cam_pos_mm`, `err_cam_rot_deg`，以及 `state_mask_left/right`（某侧无目标）都应在中间 Parquet 保存。这些**不自动保存在最终 LeRobot 训练数据里**，因此要做 QC sidecar。

![HuRo 数据坐标、下一状态监督与 QC 边界](/Embodied-AI-Roadmap/diagrams/huro-detail.svg)

*图 2：以 Stage 10 源码的 next-state 监督为准：右边新 Episode 的首帧不能成为左边 Episode 的 action target。*

## 3. 这篇最容易复现错的 Action 契约

[作者 examples/README.md](https://github.com/3587jjh/HuRo/blob/main/examples/README.md#the-lerobot-dataset)明确：ALLEX 的最终数据集（V2.0）采用：

| 键 | 形状/含义 | 工程危险点 |
| --- | --- | --- |
| `observation.state` | 48D 当前关节角 $q_t$ | 顺序由 `meta/info.json` 决定 |
| `action` | 48D **下一帧绝对关节角** | **不是** Delta；尾帧重复 |
| `observation.state_eef_left/right` | 各 9D，腕位移3+旋转6D | 相对 robot base |
| `action_eef_left/right` | 各 9D，下一帧腕姿 | 时间上同步往后移 1 帧 |
| `observation.state_cam_frame` | 9D 相机在 base 系的位姿 | 注意 cam-to-base |
| `observation.images.zed_left` | 机器人化视频 | 必须对齐同一时刻动作 |
| `task_index` / 语言 | segment 的操作描述 | 指令标签按片段，不是整个原视频 |

文档特别规定旋转 6D 是旋转矩阵的**前两行**，不要不核查目标训练工具箱就按“前两列”解析。

$$
a_t=\begin{cases}
q_{t+1},&t<T-1\\
q_{T-1},&t=T-1.
\end{cases}
$$

因此如果下游 VLA 的 action 是 $\Delta q_t$ 而不是绝对命令，就必须显式转换：

$$
\Delta q_t=a_t-q_t,\qquad
q_{t+1}=q_t+\Delta q_t.
$$

**Chunk 边界**：在长度为 $K$ 的预测序列里，动作是未来 $[q_{t+1},\dots,q_{t+K}]$，但超出 Episode 末尾必须使用有效 mask、重复或填充（由目标模型约定）处理，不能接到下一条 EGO 视频的起始动作。

### 3.1 LeRobot V2.0 与 v3.0 不可只改文件夹名

HuRo Stage 10 按 `<clips>_lerobot/<robot>/<HxW>` 保存单一分辨率的视频数据集。**作者 Stage 10 不写 `meta/stats.json`**，需依照目标训练任务、数据混合方式和 train split 的归一化配方另行生成。

2026 年 LeRobot [v3.0 官方文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)强调多个 Episodes 共享 Parquet/MP4 分片、元信息存索引及 offsets；这是另一个版本的存储契约，不应直接拿 HuRo V2 文件夹冒充 v3。必要时使用对应版本的官方数据迁移工具并重新检查视频抽帧、分片索引和 Episode 边界。

### 3.2 工厂数据平台应额外存质量和溯源信息（本站建议，非 HuRo 原版）

```json
{
  "origin": "EGO_factory",
  "source_clip_id": "shiftA_clip_0007",
  "robot_name": "industrial_dualarm",
  "calibration_version": "rig_v1",
  "action_semantics": "next_absolute_joint_position",
  "fps": 30,
  "frame_convention": "robot_base_xforward_yleft_zup",
  "ik_tip_error_mm_p95": null,
  "overlay_valid_ratio": null,
  "time_sync_offset_ms": null,
  "data_license_review": "PENDING",
  "qc": "UNREVIEWED"
}
```

`null` 表示必须真实测量，**不能因例子里提供 schema 就当作达标**。严格禁止把训练/测试源视频拆出的不同片段随机分到两边，避免同一场景、同一次操作的视觉泄漏。

## 4. 按真实源码执行的复现流程

### 4.1 环境/硬件前置条件

官方 [setup/README](https://github.com/3587jjh/HuRo/blob/main/setup/README.md)要求：Linux、CUDA 12.8 对应驱动、至少 **24GB NVIDIA 显存**。Stage 9 的 Isaac Sim RTX 光线渲染**需要 RT Cores**；A100/H100 **没有 RTX RT Cores，不能运行这一路渲染**。作者还明确提示 Isaac Sim 5.1 与高于 R580 的驱动分支存在启动崩溃。对你准备的云算力，**带 RT Cores 的 L40S/RTX 4090 适合前端渲染，H100 可留作 VLA 训练**，但是否能跑必须按 CUDA/驱动/Isaac Sim 组合实测。

此外，Stage 9 的 `OMNI_KIT_ACCEPT_EULA=Y` 是**主动接受 NVIDIA Omniverse 许可**。请先阅读许可，再自行设置；本站不替用户接受。项目自身代码 Apache-2.0，但部分模型/数据依赖禁止商业使用，**工厂数据资产项目须先审查 THIRD_PARTY_NOTICES**。

### 4.2 官方环境与输入示例（命令已对照 README）

```bash
git clone https://github.com/3587jjh/HuRo HuRo
cd HuRo
git submodule update --init --recursive
./setup/setup.sh
./setup/download.sh
conda activate huro
# MANO 右/左模型需按 setup/README 手动获取并放入指定路径
# Qwen3.5-9B 可能于首次 Stage 6 才自动下载
# 阅读 NVIDIA Omniverse EULA，确认后自行设置其接受变量
./run_pipeline.sh 2>&1 | tee run.log
python examples/read_parquet.py examples/clips_chunked
python examples/load_lerobot.py examples/clips_lerobot/allex/192x342
```

`./run_pipeline.sh` 顶部有 `INPUT_DIR`, `GPUS`, `ROBOT`, `FIRST_STAGE`, `LAST_STAGE`，示例默认有两条 clips。每个 Stage 支持断点恢复，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline/stage1_annot_intrinsics.py \
  --input_dir examples/clips --part 1/1 --no_tqdm
```

输入应为第一视角 MP4，官方假设 30fps、短边 256px、30秒至30分钟。转换命令（注意这是推理前处理，不替代原始 MCAP 备份）：

```bash
ffmpeg -i raw.mp4 \
  -vf "fps=30,scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)'" \
  -c:v libx264 -crf 18 -an clips/raw_30fps_256.mp4
```

### 4.3 替换成你的工业双臂：不只是改一行 YAML

[作者配置文档](https://github.com/3587jjh/HuRo/blob/main/configs/README.md)说明：加入目标机器人的 URDF 与 Mesh；在 `configs/<robot>.yaml` 中明确 `left_arm/left_hand/right_arm/right_hand`、关节 home pose、左右 MANO Wrist frame、camera optical link 和 21 点到机器人 finger link 的映射；执行：

```bash
python configs/check_frames.py myrobot
```

这一步用于发现 wrist/optical 坐标轴错误。**Stage 8/9** 再用你的本体跑，**Stage 10 还要在 `_LEROBOT_TARGETS` 加目标机器人的关节顺序和输出布局**。当前仓库配置仅有 ALLEX。你的 UMI 若是两指平行夹爪，就应该用 **腕 6DoF + 夹爪开合** 的 target space，而不是把机械夹爪虚构成 20 维五指动作。

HuRo 的官方代码目前**不自动读取**已有 EGO/UMI 标注。若你已有双目外参、世界系 SLAM、UMI 位姿，必须转换成 Stage N−1 期望的 Parquet 字段并添加 `.done` marker，才可从 Stage N 继续；切勿直接复制错误的 `.done` 来“通过”代码分支。

### 4.4 分阶段验收清单：数据准入不是只数文件

- **Stage 1–5**：输入视频数、标定成功数、丢帧和重力尺度异常、左/右手身份互换、手 21 点 P95 误差（若有标注 GT）。
- **Stage 6–7**：分割出的操作时长、VLM 指令抽样核对、mask 对物体的误删、inpaint 前后物体是否移动。
- **Stage 8**：IK 可达率、指尖残差、越关节限位、自碰撞、运动平滑；保留每条训练样本的诊断。
- **Stage 9–10**：`overlay_valid`、视频时间戳与 `action[t]` 对应、训练集只包含合法数据、绝对动作是否正确加载。
- **Policy**：先小样本 overfit 验证训练数据能被模型消费，再仿真 closed-loop，最后限速真机评测。

作者官网实际诊断：相机位置 ATE 中位 4.47mm、手部21点误差中位 20.4mm、IK 指尖残差中位 21.2mm；仅 **62.5% 轨迹**满足“无 >1° 关节越限”，约 **55.2% 轨迹**满足抽样帧无非抓取自碰撞。可见“视觉上看起来正确”不等于真实机器人就能安全复现。

## 5. 本站新增可运行的 CPU 契约验证

[examples/huro_contract_cpu_demo.py](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/huro_contract_cpu_demo.py) 不需要 GPU，独立执行：

```bash
python3 examples/huro_contract_cpu_demo.py
```

它包含以下独立的 synthetic assertions（**不是 HuRo 官方推理/训练结果**）：

1. `T_world_camera`→`T_base_world` 组合、齐次变换逆的一致性；
2. 48D `action[t]=q[t+1]` 与尾帧 repeat，绝对/增量转换往返；
3. 不允许跨 episode 的下一帧作为 Action，EEF label 也同步往后一帧；
4. 反例注入：overlay无效、IK残差超限、非单调帧时钟应拒收；
5. 训练/测试按 `source_clip_id` 分组，不让同一原视频不同片段泄漏。

**全部 PASS 才说明入门的训练数据语义未混淆**，不能解释为 21 点定位、真机抓取或官方 pipeline 已复现。

## 6. 如何将成品数据真正训练进 π₀.₅？

**注意：HuRo 仓库开源的是前端数据生产流程；以下属于本站设计的后端路线，不是作者发布的现成 openpi training config。**

```text
LeRobot V2 机器人化 Episode + QC sidecar
          ├─ episode/clip/scene 分组拆 train-val-test
          ├─ 联合校验 camera_frame、robot base、关节顺序
          ├─ action adapter: absolute q[t+1] → 模型需要的动作语义
          ├─ only train split 的 normalization stats
          └─ 图像、任务、状态和未来 K 个动作的 temporal sampler
                                  ↓
                  VLA 初始化（pi0.5 / ACT / DP）
                                  ↓
             robotized 预训练或继续训练 + checkpoint
                                  ↓
             少量真实本体 demo 微调 + 安全真机闭环
                                  ↓
                 失败数据 / HITL 回流 / 对照评测
```

若采用线性桥的 Flow Matching，令 $A$ 为已经转换到模型动作空间的 action chunk，$\epsilon\sim\mathcal N(0,I)$：

$$
x_\tau=(1-\tau)\epsilon+\tau A,\qquad
u^*=A-\epsilon.
$$

一个可解释的**工程有效维度 mask（不代表 openpi 默认实现）**对应：

$$
\mathcal L_{\mathrm{FM}}=
\mathbb E\left[
\frac{\sum_{t,d}m_{td}(v_\theta(x_\tau,\tau,o,\ell)_{td}-u^*_{td})^2}
{\max(1,\sum_{t,d}m_{td})}
\right].
$$

$m_{td}$ 应区分有效关节维、有效未来时间步和伪动作质量，不应无条件把无 tracking 的帧当零动作标签。在 π₀.₅ 的具体代码中，是否 mask padding 取决于真实训练实现，**不能因本文数学上引入 mask 就声称官方会自动屏蔽所有 padding**。

## 7. 有因果解释力的消融，而不是一个最好看的成功率

| 实验 | 固定条件 | 变量 | 回答的问题 |
| --- | --- | --- | --- |
| A No PT | 相同目标机器人真实示范/评测种子 | 无 HuRo 预训练 | baseline |
| B Visual-only | 相同 HuRo 来源视频 | 仅沿用视觉预训练 | 动作监督是否关键 |
| C No overlay | 同源+相同 retarget action | 人手视觉 vs robotized 视觉 | 域差对 OOD 的贡献 |
| D Full HuRo | 同样的动作数量和微调预算 | robotized obs + action | 联合机器人化收益 |
| E source-diversity | 帧数量相近 | 单源 vs 多源 | 新组合是否来自多样性 |
| F QC ablation | 同样训练预算 | 过滤 vs 不过滤动作伪标签 | 质量门槛的价值 |

至少报告任务成功数/总数、OOD 位置与物体变化、平均时延、接管、碰撞和按源场景切片结果。**不能只看离线 action MSE 就宣称获得更强闭环策略**。

## 8. 数据资产与经济性：一万小时原视频不等于一万小时训练动作

作者[项目成本剖析](https://3587jjh.github.io/HuRo/)：单 RTX 5090 的 15min 视频，EPIC-Kitchens 完整管线约 143.1min（9.5×），Ego4D 约 120.1min（8.0×）。源视频切出 manipulation segments 的比例分别为 **17.3% 与 6.9%**；**97–98%** 是进入 segment **之后**的后处理保留率，不能错认为原视频 97% 都可用。五来源原始约 3298.6h → 最终约 1316.8h。

数据工厂应按如下数量守恒：

$$
H_{\mathrm{valid}}=H_{\mathrm{raw}}\cdot
r_{\mathrm{manip}}\cdot r_{\mathrm{pipeline}}\cdot r_{\mathrm{industrialQC}}.
$$

**纯教学算例**：原视频 10,000h，操作片段率0.10、转换率0.97、工业 QC率0.8，则有效约 776h；这**不是对你的团队产能的推测**。你的 EGO/UMI 应实时统计实际每一阶段产出比例和原因；对未通过质量门的样本不要直接混进训练。

## 9. 给 EGO/UMI 工业闭环的落地次序

**P0（这一周）**：20条固定工位 EGO 视频，附相机标定、IMU、UMI 真值轨迹。验收视频到手21点、world/base 变换、时间同步和任务分割。保存原始 MCAP（绝不在原始层覆盖或降采样）。

**P1（下一步）**：准备目标机器人 URDF + 夹爪动作代理，先跑 retarget 输出对比 UMI/TCP，记录关节和指尖残差。对 **不可靠动作标签**拒收，比较无 overlay 与完整 overlay 的视觉质量与几何误差。

**P2（训练闭环）**：形成机器人标准 LeRobot Episode（明确 action absolute 或 delta），与真实本体示范共同训练。固定基准任务和测试集，跑 A–F 对照；安全真机限速、碰撞限值、人工急停和 HITL 回流作为最终验收。

## 10. 失败定位表与许可红线

| 现象 | 应查内容 | 绝不能做 |
| --- | --- | --- |
| `stage1` 后什么都没有 | 内参 JSON 是否为空；短视频是否缺相机运动 | 将 `.done` 当作成功 |
| 左右手漂移 | 手检测 side、跟踪 ID、遮挡 | 左手 q 拼到右手动作 |
| IK 看着顺但抓取无效 | wrist-frame、tip residual、接触/碰撞 | 只看关节平滑就签收 |
| overlay 黑屏 | RT Core、R580 驱动、Vulkan ICD、Isaac EULA | 用 H100 生跑 Isaac RTX stage9 |
| loss 飞起 | q[t+1] vs delta、norm stats、关节单位 | 直接把 absolute 喂给 delta policy |
| OOD 失败 | 单源/多源、场景泄漏、视觉合成 | 用 ID 单一成功率代表泛化 |

**开源和商用风险**：HuRo 自身 Apache-2.0，但 [THIRD_PARTY_NOTICES](https://github.com/3587jjh/HuRo/blob/main/THIRD_PARTY_NOTICES.md) 与 README 明确第三方依赖限制**此 pipeline 不可商用**。工厂应用要替换有许可问题的模型与数据、重新审核素材及隐私；**不能因为 GitHub 可 clone 就用于生产线商用**。

**本教程的复现状态**：已核对论文作者项目、公开十阶段仓库、schema 和 setup；已编写可运行的**教学数据契约脚本**。但尚未跑过全量作者 pipeline、GPU 训练和 ALLEX/自有工业机器人真机验证。下一篇建议衔接 [EGO/UMI 时空标定](../../02-data-engineering/ego-umi-spatiotemporal-calibration/) 与 [EgoEngine 代码复现](./egoengine-code-reproduction/)。
