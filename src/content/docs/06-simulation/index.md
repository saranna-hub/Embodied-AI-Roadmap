---
title: 06 仿真与评测
sidebar:
  order: 1
---

本专题研究具身模型如何建立可信的仿真闭环、统计成功率、拆解泛化与部署成本，而不只看离线 Loss 或单一排行榜。

## 从 π₀.₅ 的 LIBERO 训练到真正的泛化评测

新增深度教程：[MINERVA：54 万参数解决 LIBERO，为什么不等于 VLA 泛化？](./minerva-libero-capacity-and-generalization/)。

这篇专门对照 [MINERVA（2026-09-03）](https://arxiv.org/html/2609.03715)、[官方 LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 与 LeRobot 文档，讲透轻量视觉策略 CNN/FiLM/关键点、Flow Matching vs L1、预测 Chunk 长度与实际执行步数、Temporal Ensembling、任务 ID 置换、LIBERO-Plus 鲁棒性和 2000 次 rollout 评测协议。附两张教学图和 [无需 GPU 的 CPU 契约脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/minerva_eval_contract_demo.py)。

**按这个顺序阅读**：[π₀.₅ 数据→Loss→LoRA](../01-foundations/vla/pi05-libero-data-loss-lora-source-guide/) → [MINERVA 评测与泛化](./minerva-libero-capacity-and-generalization/) → [PACT-VLA 推理时序](../05-control/pact-vla-precommit-flowres/)。

后续专题继续建设 MuJoCo / Isaac Sim / Isaac Lab、Sim2Real、工业机器人任务基准与真机评测设计。每篇区分公开论文结论、本站合成测试和实际 GPU/仿真/真机复现状态。
