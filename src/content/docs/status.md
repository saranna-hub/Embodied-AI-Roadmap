---
title: 内容建设进度
sidebar:
  order: 4
description: 前四大领域的已完成页面、待撰写论文和复现状态
---

**范围**：本阶段建设具身基础模型、具身数据工程、感知与重建、Human-to-Robot。第五、六领域保留分类入口，不把空目录写成已完成课程。

### 标记规则

- **已上线解读**：已经在这个网站存在可阅读的技术正文；不等于完整实验复现。
- **待深读**：已列入选题，但要先确认论文、版本和开源资料。
- **需核验**：来自之前调研的名称/版本/规模口径，正式发布前须重新核对原始来源。

## 01 具身基础模型

| 工作/专题 | 状态 | 入口 |
| --- | --- | --- |
| π₀ | 已上线解读 | [模型架构、Flow Matching](./01-foundations/vla/pi0/) |
| Diffusion Policy | 已上线解读 | [动作扩散](./01-foundations/imitation/diffusion-policy/) |
| GR00T N1 | 已上线解读 | [双系统 VLA](./01-foundations/vla/gr00t-n1/) |
| RDT-1B | 已上线解读 | [Diffusion Transformer](./01-foundations/vla/rdt-1b/) |
| π₀-FAST | 已上线解读 | [动作频域 Tokenizer](./01-foundations/vla/pi0-fast/) |
| π₀.₅、π₀.₆ | 待深读 | — |
| RT-1、RT-2、OpenVLA | 待深读 | — |
| ACT | 已上线解读 | [Action Chunking Transformer](./01-foundations/imitation/act/) |
| 强化学习基础 | 待深读 | — |
| Ego4WAM | 已上线解读 | [EGO 数据 scaling](./01-foundations/world-models/ego4wam/) |
| InternW0、IronMind、AtomEgo、Zeva-Ego、HABILIS | 需核验 | — |
| Dyna、Gen、世界模型其他家族 | 需核验成员的技术继承关系 | — |

## 02 具身数据工程

| 工作/专题 | 状态 | 入口 |
| --- | --- | --- |
| 原始 UMI | 已上线解读 | [UMI 采集与动作对齐](./02-data-engineering/umi/) |
| PolyUMI、HIL-UMI、Open-AoE | 需核验 | — |
| EgoDex、EgoVerse、Ego4D、Ego-Exo4D | 待深读 | — |
| 工业数据集 Egocentric-10K、IndEgo、EgoLab | 需核验数据口径 | — |
| 数据手套与触觉：UVTA、DexTouch-WM、Touch2Robot | 需核验 | — |
| LeRobot、MCAP、HDF5、RLDS | 待工程教程 | — |

## 03 感知与重建

| 工作/专题 | 状态 | 入口 |
| --- | --- | --- |
| HaMeR | 已上线解读 | [单帧三维手](./03-perception/hamer/) |
| HaWoR | 已上线解读 | [世界坐标手部运动](./03-perception/hawor/) |
| MINT | 已上线解读 | [相机与双手联合恢复](./03-perception/mint/) |
| WiLoR、Dyn-HaMR、ACE-Ego-Hand | 需核验 | — |
| Ego-Exo4D-HM、HandMVNet、EgoForce | 需核验 | — |
| ORB-SLAM3、VINS-Fusion、Basalt、OpenVINS | 待专题 | — |
| Depth Anything、物体 6DoF、HOI 与 Affordance | 待专题 | — |

## 04 Human-to-Robot

| 工作/专题 | 状态 | 入口 |
| --- | --- | --- |
| Ego2Robot | 已上线解读 | [Robot-format 数据合成](./04-human-to-robot/ego2robot/) |
| UMI-Bridge | 已上线解读 | [动作锚定潜变量](./04-human-to-robot/umi-bridge/) |
| HumanEgo | 已上线解读 | [物体中心交互](./04-human-to-robot/humanego/) |
| EgoWild2Dex、DexRoam、EgoHumanoid-V2 | 需核验 | — |
| FlashDexRetarget、DexAgent、RAPID、Praxis | 需核验 | — |
| λ₀ / HumanVerse-500、Whole-Body UMI | 需核验 | — |
| Object-centric / Contact-centric 技术比较 | 待专题 | — |

### 复现状态

目前这些页面均为**文献与源码层面的技术解读**。在本仓库尚未记录对应 GPU 环境、训练日志、评测输出之前，任何文章都不应宣称“已完整复现”。

### 已知工程限制

- **Markdown→网页构建**已提交，但 GitHub Pages 的最终构建结果需检查 GitHub Actions。
- **KaTeX**已配置并接入数学语法检查，尚需检查线上真实渲染。
- 按前四大领域增量撰写，未写完的论文明确保留在清单，不自动生成空泛“已完成”文章。
