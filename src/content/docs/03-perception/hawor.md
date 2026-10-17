---
title: HaWoR：从第一视角恢复世界坐标中的手部运动
sidebar:
  order: 3
description: 相机坐标手、EGO-SLAM、运动补全与坐标变换
---

**论文**：[HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](https://arxiv.org/abs/2501.02973)（CVPR 2025） · **官方项目**：[HaWoR](https://hawor-project.github.io/) · **源码**：[ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR) · **状态**：论文方法解读，未在本站验证其推理结果。

## 1. 为什么单帧 HaMeR 不够？

你低头看自己的手，虽然手相对于头戴相机几乎不动，但身体移动后，手在工厂世界坐标里可能移动了几米。单帧手部估计器通常只输出相机局部坐标下的相对形状，而机器人学习需要连续稳定的空间运动信息。

**HaWoR 的核心设计：手的局部三维恢复与相机世界轨迹分开估计，再统一到世界坐标。** 另有时序补全模块处理手短暂出画/遮挡。

## 2. 数据流

```text
EGO 视频 ─┬─→ Camera-space Hand Reconstruction ─┐
          └─→ Adaptive Ego-SLAM ─→ Camera Pose ─┤
                                                  ↓
                                      World-space Hand Motion
                                                  ↑
                                 Motion Infiller（补缺失帧）
```

这里最重要的接口不只是 3D keypoints，还有相机内外参、坐标系约定和时间对应。若这几个不对，生成的世界轨迹即使看起来平滑，也可能物理上错误。

## 3. 三维坐标变换推导

设 $T_{WC}(t)$ 为相机到世界坐标的齐次变换，$P_C(t)$ 为相机坐标中的手部关键点，则：

$$
\begin{bmatrix}P_W(t)\\1\end{bmatrix}
=T_{WC}(t)\begin{bmatrix}P_C(t)\\1\end{bmatrix}.
$$

这个变换依赖**尺度和方向正确的 $T_{WC}$**。若 SLAM 输出的是 $T_{CW}$，必须先求逆；如果相机自身的 SLAM 世界系在每次录制时重置，需要用录制起点或场景锚点做归一化。不能直接将不同 session 的绝对 xyz 混在一起训练。

## 4. 为什么需要 Ego-SLAM 和 Motion Infiller？

头戴相机往往会频繁转头，场景中还会出现大片动态前景（自己的手、工件或其他人），导致传统静态场景假设受挑战。HaWoR 提出自适应 EGO-SLAM 路线，用于在这些运动情况下恢复相机轨迹。

当手短暂出画时，单帧检测器没有图像证据，逐帧算法可能直接丢弃输出。Motion Infiller 使用时间上下文恢复缺失段。但必须区分：**插值/补全结果是模型预测，并不等于遮挡时刻的真实动作真值。** 在高精插接场景，无法用模型补全轨迹代替高精采集验证。

## 5. 评测怎么做？

至少关注世界坐标手部误差、相机轨迹 ATE/RPE、手在出画期间的连续性、不同场景的失败率。若只算经过对齐的 3D MPJPE，可能无法体现世界坐标漂移。还应比较“相机姿态误差引起的手部绝对误差”与“手部局部姿态估计误差”。

## 6. 能否直接成为 Robot Action？

不能。HaWoR 输出的是 **Human Hand Motion**，机器人所需的是满足关节限制、碰撞、工具中心点与接触约束的动作指令。可以把世界坐标手部轨迹作为中间表征，但仍需要 Human-to-Robot 映射和动作物理可执行性检查。

最小工程验证：在同一批 EGO 视频上，对比普通单帧手部估计+通用 VIO 与 HaWoR 的 world-space 手部轨迹；分别记录遮挡段漂移、绝对轨迹误差、处理速度和失败段。官网和开源仓库提供代码、模型和演示指南，但训练代码是否完整开放需单独检查。

来源：[论文](https://arxiv.org/abs/2501.02973) · [项目页](https://hawor-project.github.io/) · [官方仓库](https://github.com/ThunderVVV/HaWoR)。
