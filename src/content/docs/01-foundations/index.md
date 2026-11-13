---
title: 01 具身基础模型
sidebar:
  order: 1
---

这个领域关注机器人策略的学习方法：模仿学习、强化学习、VLA、世界模型。

**先读**：[Diffusion Policy](./imitation/diffusion-policy/) 理解条件动作分布，再读 [π₀](./vla/pi0/) 看 VLM 如何与连续动作生成器结合。

### VLA 模型家族

后续依次建设 π、RDT、GR00T、RT、OpenVLA，逐篇核对技术演进与版本关系。已发布的教程以文章正文为准，没有内容的系列不显示为“已完成”。

### π₀.₅ 微调与训练闭环（源码级）

已新增：[π₀.₅ × LIBERO：数据契约、动作空间、Flow Loss、LoRA 与闭环评测](./vla/pi05-libero-data-loss-lora-source-guide/)。先读 [π₀.₅ 联合注意力源码](./vla/pi0-joint-attention-code-audit/)，再沿 LeRobot→变换→归一化→动作 chunk→训练损失→推理反变换追踪。附两张技术图与可在无 GPU Linux 上运行的合成验证脚本；**不代表已完成 GPU/真机复现**。

### 2026-11-07 新增：StableVLA 信息瓶颈视觉 Adapter 与鲁棒性

[StableVLA 源码级精读：DINOv2/SigLIP、通道 Gram + Sigmoid、双流融合、L1 Action 与真实污染评测](./vla/stablevla-ib-adapter-robustness/)；对照 ICML 2026 正式论文、作者源码与权重，附两张原创图和 CPU 数学契约。特别说明 StableVLA 不是 π₀.₅ 的 Flow Matching 实现，向 openpi 移植属于研究方案而非官方已验证模块。


### 2026-11-09 新增：π₀.₅ 视觉蒸馏、DINOv3 教师与梯度几何

[π₀.₅ 视觉蒸馏与梯度几何深度教程](./vla/pi05-vision-distill-gradient-geometry/)：以官方 openpi 源码为参照，设计**训练期单教师、推理期无教师**的视觉残差 Adapter；推导 Flow Matching + Feature Loss、共享梯度角度和动作优先的单向投影，对照基线 LoRA、Adapter、教师、梯度投影四组控制变量；附两张原创机制图和标准库 CPU 合成契约脚本。该路线为**研究性方案**，不能误认为 Physical Intelligence 官方已经支持或本站已完成 GPU/真机复现。

### 2026-11-11 新增：EgoWAM 世界动作模型的三种未来监督

[EgoWAM 深度教程：HPT 双头架构、未来 DINO 特征、相机补偿 3D Flow、Flow Matching 与实际开源训练配置](./world-models/egowam-world-targets-hpt-cotraining/)：以 2026-10-13 论文 v2 和 Georgia Tech 官方仓库为一手依据，区分 DINO 未来目标与 DINO 输入 stem，提供两张原创系统/算法图及纯 CPU 数学合同验证；GPU 训练、真机和 π₀.₅ 移植尚未复现。


### 2026-11-13 新增：Ego Guidance / SPAR（无动作 EGO 训练 VLA）

[SPAR 论文与开源源码级教程：双 Expert、Flow/Regression、Ego4D 双质量门、时间对齐、LIBERO-Plus OOD 评估](./world-models/ego-guidance-spar-actionfree-source-guide/)：核对 2026-11-10 新论文和 llong-cs/EgoGuidance 官方训练实现，新增两张原创图和标准库 CPU 合同测试。真实双臂测试 SPAR 20/50、π₀.₅ 22/50；本站尚未独立 GPU 复现。
