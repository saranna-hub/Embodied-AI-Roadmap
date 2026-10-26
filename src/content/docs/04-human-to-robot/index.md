---
title: 04 Human-to-Robot
sidebar:
  order: 1
---

**人类动作可以被看见，但不一定能被机器人执行。** 这里重点研究动作空间、接触关系、机器人本体约束、视角差异以及数据质量。

第一篇是 [Ego2Robot](./ego2robot/)：看看动作重定向、机器人外观合成和多层质检怎样构成一条训练数据生产线。

## 从人类视频到 VLA 闭环：三篇论文路线

建议按 [三论文统一训练闭环](./human-robot-complete-route/) → [迁移涌现](./human-robot-emergence-deep/) → [Ego-Pi](./ego-pi-deep/) → [EgoEngine](./egoengine-deep/) 阅读。分别回答**底座何时能迁移**、**怎样混合微调**、**怎样从视频构造可执行机器人示范**。教程区分官方方法与教学替代实现。

### 实际代码复现入口（新增）

**[EgoEngine 官方代码逐模块复现：Prepared EGO、Replay/MPC/RL、HDF5、接入 π₀.₅](./egoengine-code-reproduction/)** —— 有可执行命令、输入输出字段、帧时钟与坐标系校验，清楚区分已发布的代码路径、上游未开放模块和本站尚未执行的 GPU/真机实验。另见 [Ego-Pi 的数据规模与子任务消融核查](./ego-pi-deep/#6-按论文原始实验记录逐项复盘2026-10-26-补充)。
