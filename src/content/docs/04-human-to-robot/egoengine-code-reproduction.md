---
title: "EgoEngine 官方代码复现：Prepared EGO → 配对数据 → VLA"
description: "官方仓库逐模块复现：视觉合成、Replay/MPC/RL 动作优化、时间对齐、HDF5、训练闭环边界"
sidebar:
  order: 5
---

> **2026-10-26 一手源码核验**：本教程基于 [GaTech-RL2/EgoEngine-CoRL2026](https://github.com/GaTech-RL2/EgoEngine-CoRL2026) 的 [README](https://github.com/GaTech-RL2/EgoEngine-CoRL2026/blob/main/README.md)、[INPUTS](https://github.com/GaTech-RL2/EgoEngine-CoRL2026/blob/main/docs/INPUTS.md)、[REPRODUCIBILITY](https://github.com/GaTech-RL2/EgoEngine-CoRL2026/blob/main/docs/REPRODUCIBILITY.md)、[MODE_SWITCHING](https://github.com/GaTech-RL2/EgoEngine-CoRL2026/blob/main/docs/MODE_SWITCHING.md) 与 [RL_RECIPE](https://github.com/GaTech-RL2/EgoEngine-CoRL2026/blob/main/docs/RL_RECIPE.md)。命令是作者提供的实际入口；**本网站尚未在 GPU 或真机执行实验**。

## 1. 为什么这份开源代码对 EGO 研究特别重要？

EgoEngine 从人类第一视角视频生成两个互相对齐的监督：机器人视角观察与机器人的可执行动作。然而，**当前仓库不是从任意原始 RGB 自动恢复物体模型并训练完整 VLA 的“一键系统”**；它从 prepared input 开始。FoundationPose 与早期感知重建不在本次代码发布的可复现范围内。后端的视觉与动作双分支、数据对齐、导出 HDF5 可调用。

### 1.1 原论文与代码的边界矩阵

| 部分 | 官方开源现状 | 最终能得到什么 |
| --- | --- | --- |
| 原始人类视频 → 物体 6D 轨迹、重建、去除人体背景 | 此版本不包括完整前端感知流程 | 需要另外完成准备数据 |
| Prepared 场景/相机轨迹/重定向验证 | 有命令与六个示例 | 可校验数据完整性 |
| 合成机器人视觉观察 | 有命令；三个 Aria Mustard 案例输入齐全 | 渲染前景、遮挡 Mask、合成视频 |
| 动作可行性优化 | 有 Replay→MPC→RL 实现 | 通过完整校验的 EEF+手指轨迹 |
| RGB+动作对齐导出 | 有 HDF5 导出 | 监督机器人策略的数据对 |
| 完整下游 π₀.₅ 训练/真机策略 benchmark | 不在本次官方测试入口中 | 需要自行适配与评估 |

原 README 很明确：“visual replay” 与 “action refinement” 保持独立，视觉渲染采用原 RB-Y1 retarget 命令，动作优化采用 floating-wrist XHand EEF 控制；**不能把这两套数据误认为同一个关节状态**。

## 2. 一次真正可以按步骤尝试的 CPU 验收

准备 Python 3.10、OpenGL 可用的 Linux/macOS。完整命令来自官方 README：

    git clone https://github.com/GaTech-RL2/EgoEngine-CoRL2026.git
    cd EgoEngine-CoRL2026
    python3.10 -m venv .venv-visual
    source .venv-visual/bin/activate
    python -m pip install -r requirements/visual-tested.txt
    python -m pip install --no-deps -e .
    egoengine list
    egoengine validate
    egoengine demo --case mustard_1

无显示的 Linux GPU 主机需要按作者文档设置 MUJOCO_GL=egl。注意这条流程不训练策略，只输出**prepared 参考 EEF 动作与新合成的视觉资料**。

### 2.1 为什么必须从 mustard_1 开始？

| 样本 | 帧数 | 再生成视觉与数据 |
| --- | ---: | --- |
| Aria mustard_1 | 93 | 完整 |
| Aria mustard_2 | 105 | 完整 |
| Aria mustard_3 | 108 | 完整 |
| TACO Brush | 209 | 背景输入不全 |
| TACO Cut | 210 | 原始 retarget 在预备输入下 mask 为空 |
| TACO Scrape | 228 | 背景输入不全 |

这里是**官方 prepared 数据文件**，不是六个任务都已经从原始人类视频完成端到端重建。作者特别提示 TACO 的历史预览视频不能冒充新产生的训练对。

### 2.2 解剖输入与输出

Prepared 数据具备相机轨迹、scene.xml/scene_eq.xml、物体与机器人 mesh、task_info.json、trajectory_kinematic.npz、retarget.json、相机内参、预先去人体背景的视频。NPZ 包含 qpos[T,nq]、qvel[T,nv]、ctrl[T,nu]，其中 ctrl 为**仿真控制目标**。

CPU demo 的关键输出：

    outputs/mustard_1/
      render/scene.mp4
      render/foreground.mp4
      render/visibility_mask.mp4
      render/rby1_commands.npz
      render/render.json
      visual.mp4
      visual.json
      dataset.hdf5

其中 render 是重新生成的场景/机器人可视化，visual 是合成观察，dataset.hdf5 是原参考 EEF controls 和观察的配对数据。这个阶段叫“准备数据复现”，不叫“VLA 训练成功”。

## 3. 物理动作分支：Replay → MPC → RL 的内在原因

视频里手抓住物体、物体移动到桌面的轨迹，并不等于机器人能在自己的关节范围和动力学下安全完成。

**重要概念**：原始参考轨迹只是提议；动作分支是**通过物理仿真和接触反馈筛选及修正提议**。

    状态 S_i + 人类运动参考
          ├─ Replay（便宜）：未来两块是否可行？
          │     └─ 成功 → 仅提交当前一块
          ├─ MPC（中等成本）：未来两块是否可行？
          │     └─ 成功 → 仅提交当前一块
          ├─ RL（昂贵）：未来两块是否可行？
          │     └─ 成功 → 仅提交当前一块
          └─ 全部失败：恢复边界状态，拒绝失败的部分轨迹

官方默认一块 **20 控制步**，检查未来 **两块**（two-chunk plan）后只提交当前 **一块**（one-chunk execution）。如果下一块不可行，当前块即使成功也会被拒绝：这是避免短视控制的关键。后续边界重新从 Replay 试起，不会永久停留 RL。

在官方示例的仿真时钟下：sim_dt=0.01s，ctrl_dt=0.05s，20 控制步对应 100 physics steps。20 是动作控制块长度，100 是内部积分步数，二者不能混淆。

### 3.1 形式化理解物体跟踪

设人类示范物体在世界系的位姿为 \(T^h_O(t)\)，机器人控制使仿真物体达到 \(T^{sim}_O(t)\)。教学用目标可表达为：

$$
e_t=\mathrm{Log}\left[(T^h_O(t))^{-1} T^{sim}_O(t)\right] \in \mathbb{R}^{6}.
$$

可使用 \(e_t^\top W e_t\) 评价平移和旋转偏差，再结合碰撞、关节限位、时序平滑和不稳定接触。这是**原理抽象**，不是官方公布的全部 RL reward。官方常用物体跟踪终止阈值为 **0.05m 平移、1rad 旋转**；这些阈值用于仿真筛选，不是保证真机精度。

### 3.2 仓库里的实际算法预算

默认 MPC：8 samples、2 iterations。RL：1024 parallel worlds、全局共享 1500 PPO iterations，每个窗口至多 300 iterations。后一个窗口可继承已接受 RL 状态下的 actor/critic/normalizer/optimizer，不等于每个窗口重训从零开始。RL 使用 70 维每手观测（当前关节、指尖、物体位姿、参考指尖、参考控制等），**不是只把 RGB 喂给 PPO**。

官方独立 CUDA 环境命令：

    python3.12 -m venv .venv-action
    source .venv-action/bin/activate
    python -m pip install -r requirements/action-tested.txt
    python -m pip install -e .
    egoengine doctor
    egoengine action --case mustard_1

调试基线：

    egoengine action --case mustard_1 --methods replay

注意：实际 GPU 运行结果可能随采样和环境不同。README **没有证明所有 TACO 场景完整成功，也没有证明切换策略的 physics step 成本必定比 full-horizon RL 低**。发布新的效率数字前要执行官方 benchmark，并保留所有失败窗口的计算量。

## 4. 时间对齐：任何“人类 EGO → 机器人数据”必须先过的关

用 \(T_{A\leftarrow B}\) 表示从 B 坐标到 A 坐标的齐次变换，则

$$
T_{W\leftarrow G}(t)
=T_{W\leftarrow C}(t)\,T_{C\leftarrow G}(t).
$$

视觉 source、background 和 robot renderer 必须对应同一矫正视角与参考帧 clock。鱼眼 EGO 图像要先正确建模相机内参或转换到与 render 一致的针孔模型，不能直接把原始鱼眼像素用于同样的透视投影。

同一时间帧 \(i\) 的合成 RGB 和优化动作必须共享 task/case/trajectory reference index。判断同步是否可靠不能只看帧数相同，还应检查夹持接触时刻、物体位置、手腕坐标变换和动作的物理意义。

## 5. 合并两个分支：HDF5 是什么结构？

GPU action 完整成功之后，官方配对命令是：

    egoengine export --case mustard_1 --video outputs/mustard_1/visual.mp4 \
      --trajectory outputs/mustard_1/action/trajectory_mjwp_mixed.npz \
      --visual-source original_retarget_render

产出的 schema 是 **egoengine.prepared.v1**。重要字段（位于 data/demo_0 下）：

| 名称 | 说明 | 用于 VLA 的注意点 |
| --- | --- | --- |
| actions | floating wrist EEF actuator targets + XHand joints | 不是万能真机控制指令 |
| obs/qpos、obs/qvel | 仿真 state，含物体 | 不等于真实机器人测量 |
| obs/joint_pos | action simulator 关节状态 | 校验维度、单位 |
| obs/agentview_image | 视觉合成 RGB，默认 256×256 | 视觉编码器预处理 |
| obs/rby1_joint_commands | 渲染用的原机器人 joint commands | 与 EEF 标签不同 provenance |
| obs/camera_to_world | 相机位姿 | 检查矩阵方向 |
| camera_intrinsics | 随分辨率重缩放的内参 | 检查像素变换 |
| timestamps | 参考时钟秒 | 与动作帧严格对齐 |

验收条件：图像数 = 动作数 = 时间戳数；数组 finite；动作轨迹 success=true；数据字段含义在整个数据集一致；输入相机与仿真 FK 的回投影符合场景几何。

## 6. 如何真正喂给 π₀.₅？还有哪些必须补做？

EgoEngine **不等于 openpi 格式**。需要一个数据适配器，把 RGB、控制、状态和文本转为 VLA 的 observation/action chunk；随后才有数据混合、模型微调和闭环评估。

    EgoEngine HDF5 (图像+动作+状态+时间)
                  ↓
       action-space / 相机 / 时间 / mask 适配
                  ↓
      openpi observation + language + action chunk
                  ↓
        π₀.₅ Flow Matching 微调
                  ↓
          仿真 rollout / 实机安全评测

对一般连续动作 \(A\in\mathbb R^{H\times D}\)，在选定“噪声到数据”的时间方向下：

$$
x_\tau=(1-\tau)\epsilon+\tau A,\quad
v^*=A-\epsilon,\quad
\mathcal L_{FM}=\mathbb E\|v_\theta(x_\tau,\tau,o,\ell)-v^*\|_2^2.
$$

**这只是 VLA 训练机制示例，不是说 EgoEngine 官方当前已提供 PI0.5 训练脚本。** action-space 中的 EEF 平移、旋转和手指关节必须转换成当前机器人策略真正使用的控制语义；必要时使用 [Ego-Pi 的交错 29D 双手编码](./ego-pi-deep/) 作为参考。

### 6.1 如果你现在只有人类 RGB，state/action 怎么处理？

**只有 RGB、没有可靠 hand action**：不能直接对动作 expert 做标准有监督 Flow Matching，因为缺少目标动作 \(A\)。可以训练视觉语义、子任务、状态变化或辅助未来预测；也可以先用手部跟踪/SLAM/UMI 获得位置与动作。

**RGB + hand 3D pose / UMI 开合度**：有 human embodiment 动作后可以尝试共训练，但人类状态和机器人 state 并非同一物理对象，必须建立域标签、标定与缺失 mask。不能因为模型输入需要 state 就随意填零并把它当作真实 state。

**EgoEngine 合成 video + 可执行动作**：可以构建机器人监督样本，但需先验证该动作在目标机器人动力学/控制器下可行，不能把模拟通过当作真机通过。

## 7. 一份真正可以判定“训练闭环跑通”的验收表

| 阶段 | 验收动作 | 成果证据 |
| --- | --- | --- |
| Prepared 资产 | egoengine validate | 控制、scene mesh、视频、K 和轨迹校验日志 |
| CPU 视觉分支 | demo mustard_1 | 新建 scene/foreground/mask/visual，93 帧 |
| CUDA 动作分支 | action mustard_1 | 已接受的完整轨迹、模式切换日志与追踪误差 |
| 配对监督 | export | HDF5 schema 和时间戳检查 |
| VLA 训练 | 使用自己的 openpi/BC adapter | 权重、loss、数据划分和离线验证 |
| 自主 rollout | 仿真与安全真机 | 机器人动作真实自主产生，成功/失败/干预 |

只通过前四行叫 **EgoEngine 部分数据引擎复现**；必须通过后两行才叫完整训练闭环。本站**只完成文档与源码审计，不报告不存在的实测复现结果**。

## 8. 工业迁移路线：如何复用自研 EGO + UMI？

从固定桌面夹爪任务开始：用标定好的双目鱼眼、AprilTag 桌面世界原点、UMI 腕部姿态、夹爪开合构建统一世界系轨迹，再输入有物体 mesh 和真实机器人 URDF 的 MuJoCo prepared 场景。先复制本仓库的输入契约和两分支拼接，不急于改训练模型。若要覆盖未知物体、柔性物体或手指接触，必须另外研究物体重建、接触建模和本体动力学。

**先保证监督数据真实、对齐、可执行，再谈 human-only 数据规模。**

一手资料：[论文](https://arxiv.org/abs/2606.12604) · [代码](https://github.com/GaTech-RL2/EgoEngine-CoRL2026) · [复现声明](https://github.com/GaTech-RL2/EgoEngine-CoRL2026/blob/main/docs/REPRODUCIBILITY.md) · [输入规范](https://github.com/GaTech-RL2/EgoEngine-CoRL2026/blob/main/docs/INPUTS.md)。
