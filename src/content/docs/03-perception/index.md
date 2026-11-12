---
title: 03 感知与重建
sidebar:
  order: 1
---

本章专门回答人类操作视频中的“**谁在动、在哪里、以什么相机/世界坐标运动、标注能否用于训练**”。从二维检测到三维手形，从相机系到世界系，中间有多次不可省略的标定与可观测性判断。

## 推荐学习顺序

1. [HaMeR：Transformer + MANO 的单目三维手部重建](./hamer/)：理解形状、姿态、弱/透视相机与 2D→3D 不适定性。
2. [HaWoR：跨时序恢复世界空间手部运动](./hawor/)：理解相机运动、手部运动先验与缺失帧处理边界。
3. **[EgoHand 2026 源码级教程：DINOv3 + SOMA-25，2D/3D 两分支、相机几何、SLAM 与质检](./egohand-dinov3-soma-camera-qc/)**：从官方 NVIDIA IsaacCapture 推理入口与 NPZ 字段出发，推导焦距与深度关系、2D 关节约束的侧向平移求解，附两张原创技术图和 CPU 几何/帧 ID 负例。
4. [MINT](./mint/)：补充手部结构与时序技术，重点对照输入和监督条件。

**工程延伸**：学习 [EGO/UMI 时空标定](../../02-data-engineering/ego-umi-spatiotemporal-calibration/) 后，再读 [MCAP → LeRobot v3 的时序与动作合同](../../02-data-engineering/mcap-lerobot-v3-temporal-action-contract/)；3D 手重建不等于机器人动作监督，也不自动获得真实接触力。

> **2026-11-12 更新**：EgoHand 论文与官方代码已核对；此站新增合成 CPU 验证，尚未执行官方 GPU 模型推理、精度复测或真机验证。