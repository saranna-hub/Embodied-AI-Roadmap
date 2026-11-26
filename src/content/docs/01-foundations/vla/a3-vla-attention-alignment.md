---
title: "A³-VLA：交互伪标签、注意力质量与视觉表征后训练"
description: "从接触时刻的图像变化到区域注意力下界、对比监督以及 π₀.₅ 接口改造"
sidebar:
  order: 31
---

> 2026-11-26 核查：[论文 v2（11月10日）](https://arxiv.org/html/2611.06447v2)、[SAM2 官方仓库](https://github.com/facebookresearch/sam2)、[openpi PyTorch 基础实现](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)。论文检查表提及匿名代码仓库，本次未定位可访问的 A³-VLA 专用源码及权重。以下将作者方法与独立教学推导、工程建议分开，后两者不是作者实现的逐行复刻。

## 1. 为什么动作损失不等于视觉能力

假设工位左边一直是待抓件，右边一直是空盒。训练期间只要把夹爪移到左边，就可以降低动作误差。调换工件位置后，模型可能继续执行训练集中的轨迹。即使训练损失很低，也不代表模型学到了指令指定的对象。

A³-VLA 的核心是利用已有示教中的交互时刻建立视觉伪监督，再通过两个不同的梯度路径改变模型：区域注意力约束控制“看哪里”，区域语义对比约束控制“看到了什么”。它不在推理阶段引入新的视觉检测器。

![离线标注与训练部署路径](/Embodied-AI-Roadmap/diagrams/a3-vla-system.svg)

图1：本站原创技术示意。橙色为训练时可使用的未来帧，蓝色为策略当前输入，绿色为实际部署路径。

以条件策略为例，令 $o_t$ 包含多视角图像、语言和状态，$a$ 为未来动作块。openpi 的时间方向为噪声端 1 到数据端 0：

$$
x_z=(1-z)a+z\epsilon,\quad u=\epsilon-a,\quad \epsilon\sim\mathcal N(0,I).
$$

$$
L_{\rm FM}=\mathbb E\|v_\theta(x_z,z,o_t)-u\|^2.
$$

该目标并没有要求某个目标物体的 patch 获得较高注意力。监督区域的意义是给已有表示附加归纳偏置，不是替换行为克隆。若原动作本身错了，区域监督也不能自动纠正动作真值。

## 2. 从示教生成区域标签：事件、图像变化与 SAM2

论文 §3.2 和附录 A.6 使用交互事件附近的变化作为粗种子，然后离线传播掩码。以下给出工程化实现顺序。

1. 从同一 episode 的夹爪状态、接触传感器或任务阶段中选取交互索引 $k$。对夹爪宽度变化使用滞回和连续帧判决，避免一次噪声产生多个事件；此为工程建议。
2. 将图像划分为与模型视觉 patch 一致的空间区域。对 patch 计算两个窗口的帧间颜色变化。下面采用恰好 w 个差分的半开区间，避免端点越界；论文式 (2)(3) 的区间记号实现时需要明确索引约定。
3. 保留接触前静止、接触后变化明显的区域作为目标种子；另取近邻帧运动明显区域作为操作主体种子。对多次交互分别处理。
4. 将两种粗种子转成像素掩码，调用 SAM2 视频传播，独立维护目标和主体的对象 ID；向交互前后传播时分别检查一致性。
5. 将像素掩码映射回实际送入视觉编码器的网格，记录有效性、来源帧、事件类型、预处理版本和掩码版本。

教学记号：

$$
D_{\rm pre}(p)=\frac1w\sum_{i=k-2w}^{k-w-1}\|I_{i+1}(p)-I_i(p)\|_2,
$$

$$
D_{\rm post}(p)=\frac1w\sum_{i=k}^{k+w-1}\|I_{i+1}(p)-I_i(p)\|_2.
$$

$$
C_{\rm tar}=\{p:D_{\rm pre}(p)<\tau,\ D_{\rm post}(p)>\tau\}.
$$

论文使用变化阈值 10；阈值随 RGB 数值范围、曝光、patch 聚合定义而改变，不能把它直接用于归一化后 0～1 的像素。论文掩码覆盖率阈值为 0.25：

$$
m_p=\mathbf 1\left[\frac{|p\cap M|}{|p|}>0.25\right].
$$

例如 16×16 的 patch 中有 65 个前景像素则保留，64 个则不保留。若训练采用随机 crop，必须先对原始像素掩码施加相同变换，再计算覆盖率。不可独立随机增强掩码与图像。

SAM2 的 [video predictor 源码](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py) 提供视频状态与传播接口。以下是适配示意，需要固定 SAM2 commit 和 checkpoint 后核对具体签名；不是已运行脚本：

~~~python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(cfg, checkpoint)
state = predictor.init_state(video_path=video_dir)
predictor.add_new_mask(state, frame_idx=k, obj_id=1, mask=target_seed)
predictor.add_new_mask(state, frame_idx=k, obj_id=2, mask=body_seed)
for idx, ids, logits in predictor.propagate_in_video(state):
    # 保存 logits > 0 的掩码；检查 id、时间戳、分辨率
    save_masks(idx, ids, logits > 0)
# 另行反向传播；检查传播起点和状态是否需要重置
~~~

**EGO 迁移不能直接照搬帧差。** 头戴相机转动会让整个背景移动。建议优先选有可靠相机位姿及深度的片段：用 $T_{WC}$ 表示相机到世界的变换，则同一静态点从旧帧投到新帧的变换是

$$
X_{C_2}=T_{WC_2}^{-1}T_{WC_1}X_{C_1}.
$$

比较重投影后的图像差异，而不是原始坐标帧差。遮挡、深度无效和反光区必须排除。没有深度时可用光流残差或局部单应性作为低置信候选，但不能把视差引起的残差全部当作对象运动。对于 UMI，腕部相机随操作一起移动也面临同一问题。

## 3. 从注意力矩阵到可微区域监督

![区域质量与梯度路径](/Embodied-AI-Roadmap/diagrams/a3-vla-detail.svg)

图2：本站原创。注意力监督只约束目标概率质量，不指定每个 patch 的精确概率。对比项采用池化后的特征。

设最后一层的注意力张量为 [B, heads, Q, K]，先区分 query 与 key 的 token 范围。有效语言 query 集合为 $Q_l$；有效图像 key 为 $K_i$；padding 不属于任何集合。对每个 query，在实际可见 key 上进行 softmax。定义图像条件归一化（独立实现选择，作者未公开具体归一化代码）：

$$
p_{hqp}=\frac{\exp(s_{hqp})}{\sum_{j\in K_i}\exp(s_{hqj})},\qquad
s_{hqp}=\frac{q_{hq}^\top k_{hp}}{\sqrt{d_h}}.
$$

$$
P_p=\frac{1}{H|Q_l|}\sum_{h=1}^H\sum_{q\in Q_l}p_{hqp}.
$$

$$
r=\sum_{p\in C_{\rm tar}}P_p,\qquad L_l=\max(0,\eta-r)^2.
$$

与 cross entropy 强制掩码内均匀注意力不同，hinge 只要求区域总质量达到下界。区域已获得足够概率后，不再对这项施加梯度。状态到图像的项同理，但状态 query 的定义要按 backbone 适配。

教学推导：当 $r<\eta$，对单个 query 的有效图像 logits 有

$$
\frac{\partial r}{\partial s_j}=p_j(m_j-r),\qquad
\frac{\partial L_l}{\partial s_j}=-2(\eta-r)p_j(m_j-r).
$$

区域内的梯度通常为负，梯度下降提高其 logit；区域外相反。达到门槛后两者均归零。平均多头多 query 时再乘相应平均权重。

**分母陷阱。** 全序列 softmax 的图像概率之和可能只有 0.3。若直接要求目标质量 0.5，就可能迫使语言注意力整体迁移至图像，而不仅是目标区域内重新分配。对视觉 key 二次归一化可隔离这个效应，却也可能让绝对图像质量为 0.001 的 query 获得很高的条件质量。建议同时记录条件质量、绝对质量、有效 query 数及均匀基线 $|C|/|K_i|$；两种定义的阈值不能混用。

## 4. 语义对比：不仅看向目标，还要认识目标

给定语言 token 特征以及正负区域视觉特征，分别平均池化得到 $e_l,e_+,e_-$。作者用余弦相似度形成两类对比目标，仅在交互前使用。下面用等价的 softplus 形式表示（不额外引入论文未指定的温度）：

$$
L_c=\log(1+\exp(\cos(e_l,e_-)-\cos(e_l,e_+))).
$$

如果正区域相似度 0.8、负区域 0.1，则 loss 约 0.403；若二者交换，则约 1.103。此项对最终 embedding 的语义排列施加约束，而注意力项主要约束 Q/K 产生的概率。

实现时需要防止正区域为空、负区域为空、池化向量为零。目标容器、工具、定位治具也可能是必要上下文，不能无条件视为负样本。工程版可将不确定 patch 标为 ignore，或在有明确语义标注时仅选可靠 hard negative。此为独立改造，不是原论文的实验证明。

总损失为

$$
L=L_{\rm FM}+\lambda_1(L_l+L_s)+\lambda_2L_c.
$$

论文的两类区域损失在训练期加入；部署仍走原策略。建议为每个有效任务单独归一化：

$$
\bar L=\frac{\sum_b g_b L_b}{\max(1,\sum_b g_b)}.
$$

这里 $g_b$ 表示标签有效而非任务成功。没有区域标签的样本仍可参与动作损失，不能当作负样本。相反，只有人类视频和指令、没有可靠动作的样本只能参与视觉辅助项，不能虚构动作真值。

## 5. openpi 源码接口与 π₀.₅ 适配

核对基础实现 [pi0_pytorch.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)：训练 forward 调用 embed_prefix、embed_suffix，拼接 padding 和 attention mask，联合执行 PaliGemma 与 action expert，再取后缀最后 action_horizon 个输出投影为速度；返回逐动作维度的 MSE。原 forward 没有返回可用于区域损失的注意力图。

尤其注意：embed_suffix 中独立 state_proj token 分支只在非 pi05 时启用。π₀.₅ 的 state 通过输入处理进入离散语言前缀，不应直接套用 π₀ 的后缀状态索引。需要核对 tokenizer 生成的状态 token span，并将任务文本与状态字符串分开。若无法可靠区分，先关闭状态监督，只实现语言区域项。

| 张量 | 形状（工程合同） | 约束 |
| --- | --- | --- |
| image_tokens | [B, V, P, E] | 相机顺序、patch 网格与掩码一致 |
| target_mask / body_mask | [B, V, P] | bool；不要把缺失视角当背景 |
| language_query | [B, Q] | 不包含 padding、动作或状态 |
| state_query | [B, Q] | π₀ 与 π₀.₅ 分开定义 |
| attention logits | [B, H, Q, K] | 应用实际 causal / padding mask |
| action_target | [B, T, D] | 沿用基础模型的动作归一化 |
| label_valid | [B, V] | 记录有效性及离线标签版本 |

完整矩阵的内存开销为 $O(BHN^2)$。教学建议仅为被监督的 query 重算对应 logits，并沿用原注意力的 RoPE、缩放、mask 和 key 复制方式；不要直接拿 RoPE 之前的投影计算一个不一致的辅助 attention。若 fused kernel 不暴露注意力权重，可在最后一层建立精确的辅助计算分支。避免 hook 收集 detached attention，否则 loss 无法更新 Q/K；在 gradient checkpointing 场景避免前向两次导致缓存串样本。先用极小模型确认梯度，再优化内存。

示意代码（输入已完成有效 token 筛选；真实 batch 应逐样本处理空标签）：

~~~python
import torch
import torch.nn.functional as F

def region_loss(logits, region, eta):
    # [heads, valid_queries, valid_image_keys]
    assert region.any() and (~region).any()
    p = logits.float().softmax(dim=-1)
    mass = (p * region).sum(dim=-1).mean()
    return F.relu(eta - mass).square()

def semantic_loss(language, visual, pos, neg):
    assert pos.any() and neg.any()
    a = language.float().mean(0)
    p = visual[pos].float().mean(0)
    n = visual[neg].float().mean(0)
    return F.softplus(F.cosine_similarity(a, n, dim=0)
                     - F.cosine_similarity(a, p, dim=0))
~~~

初次实验不要改推理接口，不改动作编码，不训练 SAM2。先确认辅助损失对视觉投影与目标 LoRA 参数的梯度非零，动作 loss 数值与原 baseline 对齐。冻结视觉编码器后，语义监督是否仍能作用于想要更新的层取决于取特征位置。

## 6. 实验结果如何解读

论文表1中 π₀.₅ 在 VLABench 五类任务的平均成功率显示为 0.41 到 0.67；按五个表项重新计算是 41.6% 到 66.8%，即 25.2 个百分点。表2 RoboTwin 2.0 的提升较小。两者不是同一个数据集，也不是相对百分比。作者的测试规模分别为每任务 50、100 episode；不要用单个任务 100% 成功来声称通用能力。

作者在附录 A.5 给出 π₀ 和 π₀.₅ 的 30k 步训练，batch 分别为32和256。复现实验先缩小规模，再验证 batch 与训练步数效应；梯度累积不必然等价于原训练配方。具体显存依赖序列长度、LoRA、精度和 checkpointing，不在未实测时承诺硬件需求。

应重点区分两种可能机制：一是视觉语义泛化确实改善；二是模型仅学到掩码对应的固定位置偏置。注意力可视化不是因果证据。做位置交换、指令交换、遮挡目标和保持背景不变的干预实验，才能更有说服力地检验 grounding。

## 7. 可执行复现路线：先独立模块，再闭环

### A. CPU 数学与接口

在仓库根目录运行：

~~~bash
python3 examples/a3_vla_attention_contract_cpu.py
~~~

脚本只依赖标准库，检查半开区间边界、覆盖率、条件归一化、hinge 梯度有限差分、对比排序及有效样本平均。通过不代表已跑通作者模型。

### B. 生成 50 条示教的区域缓存

先选固定外部相机，两个外观相似目标、两个可交换位置。保存 episode_id、frame_timestamp、view_id、interaction_idx、target_mask、body_mask、mask_valid、source_hash、preprocess_hash。训练和验证按原始录制 episode / 操作员分组切分，禁止相邻裁剪跨集合。

对每类任务随机抽帧人工复核种子召回、传播 ID switch、空掩码率、前景占比、目标覆盖和时序一致性。选择不确定标签舍弃而非强制监督。

### C. 训练与消融

| 组别 | 动作 loss | 语言区域 | 状态区域 | 对比项 |
| --- | --- | --- | --- | --- |
| A 基线 | 是 | 否 | 否 | 否 |
| B | 是 | 是 | 否 | 否 |
| C | 是 | 是 | 是 | 否 |
| D | 是 | 是 | 是 | 是 |
| E 负对照 | 是 | 随机等面积掩码 | 否 | 否 |

先在同一初始化与同一数据顺序上做小规模训练；工程起点可设辅助权重 0.1 并观察梯度范数，再进行网格搜索。该数字不是作者超参。π₀.₅ 未可靠解析状态 span 时先比较 A、B、D 的无状态版本。日志至少包含动作误差、各辅助 loss、梯度范数、有效区域比例、绝对/条件注意力质量、正负余弦差。

### D. 工业任务闭环评测

将抓取物料 A/B、目标工位位置、光照、背景和操作速度独立随机化。报告任务成功、错抓率、首次接触成功、人工接管率与耗时，至少按熟悉/新场景分组；保留失败录像和策略版本。不要把辅助 loss 下降当作真实成功率提升。

若错抓率下降但插入卡滞不变，说明可能改善的是目标识别，不是接触控制；可再结合 [TacInno](../tacinno-tactile-innovation/) 的触觉监督思路进行独立消融。若错误主要来自时间同步、TCP 标定、相机坐标偏差，则应优先修数据和控制，而不是加注意力 loss。

## 8. 复现边界与下一步

本次核查覆盖论文与两个基础项目接口，未取得作者专用源码和权重，未运行 SAM2 GPU 推理、VLA 训练或仿真评测。本文提出的 EGO 去运动、标签有效性、分组切分、对照干预等是针对工业数据治理的独立建议。

推荐阅读：论文 §3.2–3.5、附录 A.5–A.7；[SAM2 源码](https://github.com/facebookresearch/sam2)；[openpi](https://github.com/Physical-Intelligence/openpi)。先回答三个问题：为什么未来帧可以生成离线标签但不能输入推理？为什么 hinge 不是精确注意力标注？为什么 π₀ 与 π₀.₅ 的 state token 不能共用硬编码索引？
