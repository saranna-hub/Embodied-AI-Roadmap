---
title: 02 具身数据工程
sidebar:
  order: 1
---

机器人模型性能离不开数据：观测、状态、动作、时间对齐和可重复的质检。

建议先读 [UMI](./umi/)：它展示了在人类操作时捕获与机器人控制相关的信息，并解释为什么传感器同步和动作表示必须在训练前解决。

本领域将继续覆盖 EGO 数据集、采集手套、触觉、LeRobot/MCAP、QC 与训练数据清洗。

## 专题：EGO/UMI 工位定位与训练数据质量

从设备对比继续读 [双目鱼眼＋IMU＋桌面 AprilTag：时空标定、TCP 世界系位姿、训练标签质检](./ego-umi-spatiotemporal-calibration/)。本篇从 Kalibr、鱼眼 IPPE/PnP、工具外参、时间偏差到 VLA Action Chunk，附两张流程图与可执行合成反例；与 [HiFi-UMI](./hifi-umi-deep/) 和 [MonoEgo](./monoego-deep/) 配合学习。

## 专题：双 Vive Tracker 的工具外参与 TCP 标定

继续阅读 [UMI Tracker 手眼 AX=XB、Pivot TCP 标定与双手 Replay 坐标链](./umi-tracker-handeye-pivot-calibration/)：解释 OpenVR 坐标与时钟、rigid eye-in-hand 的使用边界、Pivot 的可观测性、双手工具独立外参，并用 CPU 合成契约验证接口方向。与上文的相机/IMU 时空标定互补。


## 2026-11-06 新增：MCAP → LeRobot v3 → π₀.₅ 的工业数据契约

**[工业 EGO/UMI 多模态数据工程：设备时钟、曝光时刻、SE(3) 动作语义、v3 Episode、openpi 适配与训练质检](./mcap-lerobot-v3-temporal-action-contract/)**。承接上述 EGO/UMI 外参与时间校准，重点区分 MCAP log/publish time 与真实采样时间、避免未来视觉泄漏、处理不同频率的状态和机器人命令；附两张技术图、独立 CPU 负例验证脚本和从 10 段采集到真机评估的实验路线。**v3 容器不等于旧版 openpi 直接兼容，也不自动从人类视频产生机器人监督。**
