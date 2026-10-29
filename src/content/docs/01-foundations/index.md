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
