---
title: 预备知识｜具身智能从零入门
sidebar:
  order: 0
---
# 从零建立具身智能的算法基础

本专区不要求计算机科班背景。先理解**张量、参数和梯度**，再学 **Transformer / Attention**，最后掌握 **视觉编码、Flow Matching 和机器人动作**。

![知识学习路线](/Embodied-AI-Roadmap/diagrams/prerequisites-map.svg)

| 顺序 | 课程 | 学完能做什么 |
| --- | --- | --- |
| 1 | [张量与模型训练](./tensors-training/) | 理解 [B,50,32]、Linear、Loss、梯度 |
| 2 | [Transformer 深度教程](./transformer/) | 计算 Q/K/V、Attention、Mask 和 KV Cache |
| 3 | [视觉编码与 VLM](./vision/) | 把 RGB 变成视觉 Token，理解 PaliGemma |
| 4 | [Flow Matching 与动作](./flow-matching/) | 从训练公式推导 π₀ 的推理积分 |

学习方式：每篇都先看**生活类比**，再看**数字手算**、**模型图**、**公式**和**能运行的 Python**。读到不懂的术语就回到本专区。读完再看 [π₀ 原理与实现](../01-foundations/vla/pi0/)。

课程中简化的实现只为讲清原理，不等于已复现大型模型。