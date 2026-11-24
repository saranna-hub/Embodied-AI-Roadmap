---
title: 05 控制与执行
sidebar:
  order: 1
---

本章专注策略输出到真实控制信号的闭环：异步推理、动作块调度、运动学、轨迹优化、双臂执行及异常恢复。

## 已发布的深度教程

- [RTC：π₀.₅ 异步动作块、软掩码 VJP 与训练期前缀条件化](/Embodied-AI-Roadmap/05-control/rtc-inference-training-time-action-conditioning/)（2026-11-10）：NeurIPS 2025 RTC + 后续 Training-Time RTC、LeRobot 固定源码核验、两图、CPU 时序与 loss 契约。讲清模型推理延迟如何映射到真实机器人动作。

- [PACT-VLA：提交前新观测、FlowRes 自一致性与异步动作块](/Embodied-AI-Roadmap/05-control/pact-vla-precommit-flowres/)（2026-10-30）：论文公式、π₀.₅/LIBERO 实验参数、CPU 合成契约测试、真机异步部署边界。

- [FlashVLA：流式 Flow Matching 动作解码、分块因果注意力和异步执行](/Embodied-AI-Roadmap/05-control/flashvla-streaming-flow-action-decoding/)（2026-11-04）：官方源码逐函数、N=4/C=10 训练张量、Cold-start、动作队列、CPU 数学契约和 LIBERO/RoboTwin 复现入口。建议先读 π₀.₅ 再与 PACT-VLA 对照。

- [FlowDPG：无需 ODE BPTT 的 Flow Policy Critic 梯度蒸馏与双臂长程任务后训练](/Embodied-AI-Roadmap/01-foundations/vla/flowdpg-bptt-free-critic-gradient-posttraining/)（2026-11-08）：CoRL 2026 公式与原表核查、IQL 双 Critic、SARM 阶段奖励、离线到在线回流、两图和 CPU 数学契约。适合在掌握 Flow Matching 后研读。

建议先读 [π₀.₅ LIBERO 数据/损失源码核验](/Embodied-AI-Roadmap/01-foundations/vla/pi05-libero-data-loss-lora-source-guide/)，再理解训练完成后如何让动作块在真实物理时间内可靠执行。

其余运动学、力控和多臂协作章节待逐项补充。

- [τ₀-VLA：高层子任务决策、失败记忆纠错、预演搜索与低层40D动作契约](/Embodied-AI-Roadmap/01-foundations/world-models/tau0-vla-ttc-memory-search-source-audit/)（2026-11-16）：与RTC不同，专门解决长流程子任务选择及执行结果回写；基于官方论文与开源文件逐项审计，标识尚未完整开放的TTC编排。

- [InteractVLA：几何引导、软时间匹配与可审计失败恢复](/Embodied-AI-Roadmap/05-control/interactvla-soft-time-geometric-guidance/)（2026-11-24）：冻结模型的输入梯度、命令位姿映射、兼容性拒绝、版本化候选执行、两张原创图与 CPU 合同。
