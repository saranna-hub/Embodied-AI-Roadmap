# π 系列：从 π₀ 到开放世界机器人策略

这组文章以 Physical Intelligence 的 π 系列为主线。按“先理解架构、再理解训练变化、最后跑通代码”的顺序阅读。系列文章不按发布日期简单堆砌，也不把不同版本误写成同一种模型。

| 顺序 | 文章 | 重点 | 状态 |
| --- | --- | --- | --- |
| 01 | [π₀：结构、动作生成与训练](./pi0-deep-dive.md) | PaliGemma、action expert、flow matching、action chunk、实现 | **技术解读已写；未做本地训练复现** |
| 02 | π₀-FAST：为什么要把连续动作离散化 | FAST tokenizer、自回归生成、与 flow matching 的取舍 | 待写 |
| 03 | π₀.₅：开放世界泛化 | knowledge insulation、跨环境训练、与 π₀ 的输入/训练差异 | 待写 |
| 04 | π₀.₆：从经验中改进策略 | 后训练、数据回流与策略优化 | 待核验后写 |
| 05 | openpi 代码实战 | 数据格式、微调配置、推理服务、评测与排障 | 待实机/环境验证 |

## 三条贯穿问题

1. **视觉、语言与动作怎么交互？** 为什么 π₀ 不直接让 VLM 回归一个动作？
2. **连续动作怎么生成？** flow matching 的向量场、噪声插值、训练目标、推理积分和 action chunk 的关系是什么？
3. **模型怎样用于新机器人？** Action space、坐标系、归一化、相机视角、数据分布、控制频率如何影响微调与部署？

## 来源

- [π₀ 原始论文](https://arxiv.org/abs/2410.24164)
- [π₀ 官方技术介绍](https://www.pi.website/blog/pi0)
- [openpi 官方代码](https://github.com/Physical-Intelligence/openpi)
- [Physical Intelligence 官方研究日志](https://www.pi.website/blog)

文中区分**论文原始设计**与 **openpi 仓库当前实现**。后者会持续更新，代码细节应以所引用的仓库版本为准。

## 架构面试补充

- [π₀ 架构深挖与 15 道面试追问](./pi0-interview.md)：从模型输入输出、Attention Mask、Flow Matching、缓存到训练与真机部署。
