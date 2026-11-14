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

## RGB-only EGO 训练 π₀.₅：新增源码级路线

**[只有 RGB 的 EGO 如何训练 π₀.₅：监督构造、四条路线与端到端验收](./ego-rgb-only-to-pi05/)** —— 逐行核对 openpi `compute_loss` 的必要动作标签、分析视频自监督/人类 3D 伪标签/UMI 动作锚/机器人化数据生成各自能监督哪些参数，并给出世界系标定、同步、动作 mask、数据契约和真机消融方案。特别强调「只有像素」不等于「可以直接计算 Flow Matching 动作损失」。

## 新增：WING——RGB-only EGO 的交互潜动作与频谱迁移

**[WING 深度教程：相机运动解耦、WING-LAM、DCT 与门控 VLA 训练](./wing-ego-spectral-transfer/)** —— 衔接上面的 RGB-only EGO 路线，讲清为什么“头动引起的光流”不能当机器人的动作、如何蒸馏 RGB-only latent、为何只用低频作动作引导；附两张机制图和独立 CPU 合成契约验证（作者原版代码尚未公开）。


## 2026-11-02 新增：EgoLAP——用运动语言让人类数据真正监督 VLA

**[EgoLAP 全流程深度教程：Language Action、Motion-level Reasoning、MoT 三段 Mask、梯度隔离、GRPO 和 CPU 契约验证](./egolap-language-action-reasoning/)**。与 Ego-Pi/人机迁移涌现、WING、EgoEngine 的区别是：它用**共享的运动语言**与**物理 why 理由**监督 VLM，同时让人类与机器人连续动作都监督 Flow Matching 动作专家，且推理**不生成理由文本**。已有两张图和独立标准库 Python 验证脚本；作者代码/权重仍未公开，不可声称原版训练已复现。

## 2026-11-03 新增：HuRo——EGO 人类视频规模化机器人化与 VLA 预训练

**[HuRo 十阶段源码级深度教程：人手追踪/SLAM、机器人 IK、Isaac Sim 视觉叠加、LeRobot V2 与 π₀.₅ 动作空间接入](./huro-robotization-to-vla/)**。配有两张独立流程图、关键几何和 Flow Matching 数学、代码可执行入口、训练消融与 CPU 数据契约脚本。重点纠错：官方 Stage 10 的 48D `action[t]=q[t+1]` 是**下一帧绝对关节位置**；`stats.json` 不会自动生成；全量 630K 数据在官方 README 仍标明 Coming soon。附非商用依赖及 RT Core 硬件限制，不假称已复现 GPU/真机训练。


## 2026-11-14 新增：ACE-Ego-0 人机联合预训练的数据可信性

[ACE-Ego-0 源码级教程：相机系 EEF、URDF 形态、物理时间动作块、可靠性加权 Huber、22D 论文与 23D Common23 差异](./ace-ego0-camera-morphology-reliability/)——依据原论文与官方 SFT/main 源码，提供两张原创 SVG、CPU 数学/动作接口合同、工业 EGO/UMI 五组消融建议。**当前作者公开范围不包括完整 449-mixture 预训练和人类 EGO 辅助损失实现，不声称完整复现。**
