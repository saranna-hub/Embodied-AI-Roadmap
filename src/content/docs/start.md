---
title: 阅读指南
sidebar:
  order: 1
---

建议遵循 **观察 → 表示 → 动作 → 执行 → 验证** 的顺序学习，而不是先记住所有模型名称。

1. **已经会 PyTorch**：先读 [Diffusion Policy](./01-foundations/imitation/diffusion-policy/) 理解动作分布，再读 [π₀](./01-foundations/vla/pi0/) 中的 Flow Matching。
2. **做机器人数据工程**：先读 [UMI](./02-data-engineering/umi/)，理解坐标系、时间戳、Action Chunk 如何形成训练样本。
3. **关注 EGO 感知**：先读 [HaMeR](./03-perception/hamer/)，再转到 [Ego2Robot](./04-human-to-robot/ego2robot/)，理解“能估计人手”不等于“能执行机器人动作”。

### 一篇文章的阅读顺序

先读任务示例和结构图，再看张量与公式，最后检查实验和源码链接。每篇涉及作者结果的地方都会明确标注“论文报告”；没有运行过的命令不会写成“已复现”。
