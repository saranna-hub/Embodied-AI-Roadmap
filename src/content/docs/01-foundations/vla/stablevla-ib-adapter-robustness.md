---
title: "StableVLA 源码级精读：信息瓶颈视觉 Adapter、通道 Gram 注意力与工业扰动评测"
description: "以 ICML 2026 原论文、项目页和开源代码交叉核验 IB-Adapter：DINOv2/SigLIP→通道协方差→Sigmoid→L1 动作头，附 CPU 契约与真实复现路径"
sidebar:
  order: 8
---

> **证据基线（2026-11-07）**：本篇对照 [ICML 2026 正式论文页](https://proceedings.mlr.press/v306/fu26x.html)、[作者项目页](https://dagroup-pku.github.io/StableVLA/) 与 [作者开源 HumanNet/StableVLA](https://github.com/DAGroup-PKU/HumanNet/tree/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA)，固定审阅树 SHA <code>1ef11ca91422ad1f60365b54ee54ec74d745bbf4</code>。**官方代码和模型权重已经可访问**，但本站只进行了静态源码核查及 CPU 合成数学验证，没有执行 StableVLA GPU 训练、LIBERO rollout 或真机复现。论文中的量化指标均为**作者报告**，不是本站结果。

## 0. 问题先说清：视觉 backbone 很强，为什么机械臂仍会看错？

工业真实部署可能出现：同一相机的曝光漂移、反光金属表面、腕部镜头油污、运动模糊、工装遮挡、灯光更换；模型在 clean 图像上的高成功率，无法保证扰动时的动作决策正确。

传统 VLA 数据流：视觉编码器将图像变成 patch token，**projector/adapter 将视觉维度映射到 LLM 隐空间**，文本与图像共同决定动作。即使视觉教师本身有强语义，投影器也可能把与操作无关的高能噪声一起送给策略。StableVLA 的研究假设：**薄弱点之一在模态对齐投影层，而非必须增大整个 VLA**。

![StableVLA 主干与 Fused IB-Adapter 的插入位置](/Embodied-AI-Roadmap/diagrams/stablevla-ib-system.svg)

*图 1｜本站根据作者源码重绘。**DINOv2 + SigLIP、Qwen2.5-0.5B、连续 L1 动作头**属于该代码分支；这**不是** π₀.₅ 的 SigLIP + Gemma + Flow Matching 动作专家。*

### 0.1 本文和现有手册有什么不同？

[π₀.₅ 的数据→Flow Loss→LoRA](./pi05-libero-data-loss-lora-source-guide/)讲训练数据与 Flow Matching；[MINERVA 评测](../../06-simulation/minerva-libero-capacity-and-generalization/)讲基准设置。本篇只解决**视觉 token 如何在对齐层排除干扰、模型实测如何与零样本鲁棒性区分、如何设计可证伪对照**。不要把 StableVLA 的 L1 动作训练写成 π₀.₅ 原生实现。

## 1. 一手来源与可复现性：名称相近不代表代码相同

| 验证项 | 证据与结论 |
| --- | --- |
| 论文 | [StableVLA, ICML 2026（PMLR 306）](https://proceedings.mlr.press/v306/fu26x.html)，2026-05-18 首次上传 arXiv：[2605.18287](https://arxiv.org/abs/2605.18287)。 |
| 算法图/实验表 | [作者项目页](https://dagroup-pku.github.io/StableVLA/)给出架构、LIBERO、CALVIN 与 Astribot S1 真机统计。 |
| 官方实现 | [HumanNet/src/model/StableVLA](https://github.com/DAGroup-PKU/HumanNet/tree/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA)：源码、训练、仿真评估脚本。 |
| 权重 | [DAGroup-PKU/StableVLA](https://huggingface.co/DAGroup-PKU/StableVLA/tree/main)，存有不同任务 checkpoint；作者 README 也给出下载用的 <code>beikui12345/stablevla</code> 路径，执行前必须核实仓库和路径是否对应。 |
| 文档警告 | 作者 <code>docs/stablevla.md</code> 的论文链接仍留有 <code>TODO</code>，但 <code>src/model/StableVLA/README.md</code> 已提供训练/推理命令。不能因为旧文档 TODO 就断言“未开源”。 |

**术语**：本文的 *Information Bottleneck* 是论文的设计动机；代码中可直接查验的是通道相关性与门控结构。**未在阅读的训练代码中发现一个可直接等同于互信息 $I(Z;X)$ 的显式变分上界优化项**，不要把模型结构的效果误说成对互信息完成了精确求解。

## 2. 从输入到输出：两路视觉先融合，再跨模态对齐

官方 [DinoSigLIPViTBackbone](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/prismatic/models/backbones/vision/dinosiglip_vit.py) 同时运行 DINOv2 与 SigLIP。在相同 patch 网格上按最后一维拼接：

$$
Z_{\mathrm{vis}}=\operatorname{Concat}_C \bigl(F_{\mathrm{DINOv2}}(I),F_{\mathrm{SigLIP}}(I)\bigr)
\in\mathbb R^{B\times N\times C_v}.
$$

此处 **DINOv2，不是 DINOv3**；直接按记忆把教师升级成 DINOv3 会改变特征统计、patch 对齐和 checkpoint 兼容性。代码注册的 224×224、patch14 输入在标准裁剪下对应 $16\times16=256$ 空间 patch；多相机输入 token 需要根据真实的 <code>num_images_in_input</code> 与 wrapper 维度确认，不能把 256 当作所有实验恒定值。

随后接入 <code>FusedFANProjector</code>，输出

$$
E_{\mathrm{vis}}=P_{\mathrm{IB}}(Z_{\mathrm{vis}})\in\mathbb R^{B\times N\times C_\ell}.
$$

<code>C_v</code> 为视觉融合维，<code>C_\ell</code> 为语言模型隐藏维。视觉 token 与语言 token 在 VLM 层组装，多模态嵌入参与下游动作学习。该实现提供 proprio token 以及 L1 连续动作头，作者 LIBERO 配置用 8 步 action chunk 与 7 维 action，**不能与 openpi 的 10 步、32 维 padding 容器混用**。

**工程输入输出契约**：

| 环节 | 张量/信息 | 常见错误 |
| --- | --- | --- |
| 摄像头 | 当前 RGB，输入前需符合 DINOv2/SigLIP 两套 normalize/resize | 单独调整其中一套图像 resize，破坏 patch 配对 |
| 双视觉特征 | <code>[B,N,Cv]</code> | 把双路特征沿 token 维拼接，错用作者按通道融合方式 |
| Adapter | <code>[B,N,Cl]</code> | 将通道 Gram 误当空间 $N\times N$ 自注意力 |
| Prompt + state | 文本 token、视觉 token、proprio | 将未来 action 标签当历史输入，制造训练泄漏 |
| Action head | <code>[B,8,7]</code>（LIBERO 典型配置） | 复制 π₀.₅ Flow Loss，以为官方 StableVLA 用 FM |
| 环境 | 7D 动作反归一化、夹爪控制语义与仿真步进 | 把学到的动作张量当机器人 TCP 世界位姿 |

## 3. 真正核心：为什么是 $D\times D$ 的通道相关性？

普通 Transformer token 注意力比较两个空间位置，权重通常形如 $N\times N$；这里作者的 [EfficientChannelAttention](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/prismatic/util/nn_utils.py) 沿 **通道**计算相关性。

先把已经映射到语言维度的特征设为 $Y\in\mathbb R^{B\times N\times C_\ell}$，分成 $H=16$ 个头，$D=C_\ell/H$。该实现明确写着：

$$
Q=\operatorname{reshape}(W_qY)\in\mathbb R^{B\times H\times D\times N},\quad
K=\operatorname{reshape}(Y)\in\mathbb R^{B\times H\times D\times N}.
$$

**注意**：$K$ 是原样重排的输入，代码没有独立的 $W_k$。然后：

$$
G_{bhij}=\frac{1}{\sqrt D}\sum_{n=1}^{N}Q_{bhin}K_{bhjn},\qquad
G\in\mathbb R^{B\times H\times D\times D}.
$$

这一步中的求和维是 patch/token 索引 $n$，输出是两个**通道**的相关性，不是位置之间的相关性。$\sqrt D$ 是源码指定的缩放；不要随意替换成 $\sqrt N$，否则数值和 checkpoint 均发生变化。

接下来：

$$
S_{bhij}=\sigma(G_{bhij})\,T_{h},
\quad
V=\operatorname{LN}\!\left(W_{v2}\,\operatorname{GELU}(W_{v1}Y)\right),
$$

$$
O_{bh} = S_{bh}V_{bh}^{\top},\quad
O\in\mathbb R^{B\times N\times C_\ell}
$$

其中 $T_h$ 是可学习的逐头温度参数，而 $\sigma$ 是 Sigmoid。代码的 $V_{bh}^\top$ 在矩阵乘法阶段形状为 $[D,N]$；输出再 transpose 回 $[N,D]$。

![StableVLA 通道 Gram、Sigmoid 与双流融合张量过程](/Embodied-AI-Roadmap/diagrams/stablevla-ib-channel-detail.svg)

*图 2｜注意力沿通道维做 $D\times D$ 的 Gram，相比普通 patch 注意力的 $N\times N$ 维度不同。形状和运算顺序均由源码核验。*

### 3.1 Sigmoid 不同于 Softmax：不是“注意力概率”

常规 Softmax 对同一行强制归一：

$$
\alpha_{ij}=\frac{\exp G_{ij}}{\sum_k\exp G_{ik}},\qquad \sum_j\alpha_{ij}=1.
$$

Sigmoid 是各元素独立门控：

$$
S_{ij}=\frac{T_h}{1+\exp(-G_{ij})},\quad
\sum_jS_{ij}\ne1\ \text{（通常）}.
$$

所以某个通道增强**不必**挤占另一个通道的权重。代价是输出幅值与通道数、温度、特征尺度有关：Sigmoid 的 $G=0$ 给出 0.5，**不是“零相关就自动变成零”**。在含大幅度结构性噪声时，门控本身仍可能被误导。代码在投影前加了 LayerNorm，但这不等价于严格的抗噪保证。

### 3.2 两路融合公式，源码里的门系数是常量

源代码先计算：

$$
F_{\mathrm{clean}}=\operatorname{GELU}\bigl(W_2\,\operatorname{GELU}(W_1Z)\bigr),
$$

$$
F_{\mathrm{robust}}=\operatorname{ECA}\bigl(\operatorname{GELU}(W_fZ)\bigr),
$$

$$
E_{\mathrm{vis}}=W_{\mathrm{out}}\,
\operatorname{LN}\bigl(F_{\mathrm{clean}}+\tanh(0.3) F_{\mathrm{robust}}\bigr).
$$

最后一式的 <code>0.3</code> 是由 <code>register_buffer('gate', tensor([0.3]))</code> 固定的 buffer，**不是 trainable Parameter**；实际融合系数约 $0.291313$。可学习的是注意力的 <code>temperature</code>、Q/MLP/输出投影等权重。若后续有人讲“训练学得最优 fusion gate 0.3”，这和当前代码不符。

### 3.3 两条源码实现的 Dropout 存在默认差异

- [训练模型里的 <code>FusedFANProjector</code>](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/prismatic/util/nn_utils.py) 构造默认 <code>drop_prob=0.0</code>。
- [HF 加载用 <code>PrismaticProjector</code>](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/prismatic/extern/hf/modeling_prismatic.py) 的 clean 流里是 <code>nn.Dropout(p=0.3)</code>，训练状态才生效。

这**不代表推理预测一定不一致**（<code>eval()</code> 会关闭 Dropout），但复现预训练与微调时必须核实实际实例化的是哪条路径、是否载入匹配的 projector 权重。**本文只确认该代码差异，不擅自断言作者实验使用了错误实现。**

## 4. 从论文优势数字到正确的评测读法

作者 [实验表](https://dagroup-pku.github.io/StableVLA/)（LIBERO-Spatial，severity 5）：

| 模型 | Clean | S5 | 从 Clean 下降 | 比 VLA-Adapter 的 S5 |
| --- | ---: | ---: | ---: | ---: |
| VLA-Adapter 0.5B | 96.0% | 58.5% | -37.5 pp | 基线 |
| StableVLA 0.5B | 96.2% | 82.0% | -14.2 pp | **+23.5 pp** |
| OpenPi-0.5（作者表中 3B） | 98.4% | 62.4% | -36.0 pp | +3.9 pp |

**注意三个边界**：一是以上数字只对应一个任务套件、一个 severity，不等于作者项目页的“仿真平均改进 +35.2%”。二是 +23.5 **百分点** 与相对提升 $(82-58.5)/58.5\approx40.17\%$ 不是同一指标。三是作者表中 OpenPi 在 Object-S5 达到 **76.4%**，优于 StableVLA 的 **70.2%**；不能概括为“StableVLA 所有任务均超过 π₀.₅”。

在真实 Astribot S1 任务中，论文项目页用相对 clean 的性能变化 $\Delta$pp 来呈现 noise、blur、oil、shelter。**不要把负的 $\Delta$pp 当作绝对成功率**。例如 Pick & Place：StableVLA clean 80%，Noise $\Delta=-30$pp，换算绝对 50%；VLA-Adapter clean 80%、Noise $\Delta=-66.7$pp，约为 13.3%。

**鲁棒性对照至少同时报告** clean success、corrupted success、绝对下降 pp、相对保留率、执行时延、失败类型。单独宣传 corrupted improvement，可能掩盖 clean 任务成功率下降或控制频率受损。

## 5. 训练脚本核查：它不是“无需图像增强就完全不训练”

官方 [StableVLA README](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/README.md) 列出了三条路线：

1. 用作者 VLM checkpoint 直接进入 LIBERO 微调（推荐）；
2. 可选地预训练带 Fused IB-Adapter 的 VLM；
3. LIBERO/CALVIN 闭环干净与扰动两组评测。

LIBERO 微调示例关键参数：<code>--use_minivlm True --use_l1_regression True --use_proprio True --num_images_in_input 2 --use_lora True --lora_rank 64 --batch_size 16 --max_steps 150005</code>（其中 <code>use_l1_regression</code> 在配置中默认 True）。源码 [<code>run_forward_pass</code>](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/vla-scripts/finetune_fused.py#L388-L445) 调用 <code>action_head.predict_action</code> 后执行：

$$
\mathcal L_{\mathrm{action}}=\frac{1}{BHA}\sum_{b,h,d}\left|\hat a_{bhd}-a_{bhd}\right|.
$$

这里是 **L1 连续动作回归**。StableVLA 的视觉改进与动作头属于不同设计层，**无法未经改造就把同一 checkpoint 当成 π₀.₅ 的 Flow Matching 权重**。

同时，作者示例命令明确包含 <code>--image_aug True</code>。论文所说 **“无需额外数据或额外增强策略”** 应理解为其结构改进不以专门新增抗干扰样本为必要条件；不能把宣传语理解成官方训练命令“关闭所有图像增强”。若做严谨消融，应把作者默认图像增强设置与 IB-Adapter 的影响正交比较。

### 5.1 推荐复现路径（不把下载等于复现）

官方资源在 [模型文件](https://huggingface.co/DAGroup-PKU/StableVLA/tree/main) 可见，仓库 README 列出 PyTorch 2.2.0、FlashAttention 2.5.5、LIBERO RLDS（约 10GB）和评估脚本。**这些依赖是作者写的环境范例，不是本站运行验证**；某个 CUDA/GPU 组合能否直接安装仍需单独核实。

~~~bash
# 只列作者代码的关键路径，运行前锁定 commit 和虚拟环境
git clone https://github.com/DAGroup-PKU/HumanNet.git
cd HumanNet/src/model/StableVLA
# 查看源码与自述文档中的依赖、数据路径、权重下载和命令
grep -n "use_l1_regression\|use_lora\|image_aug" vla-scripts/finetune_fused.py
grep -n "corruption_severity\|corruption_type" experiments/robot/libero/run_libero_eval_noise.py
~~~

**真正跑复现建议按 5 阶段**：

1. 先检查 checkpoint 的 <code>arch_specifier</code> 必须为 <code>no-align+fusedfan-projector</code>，两套图像 backbone 的预处理版本一致，并确认输入两路相机的帧顺序/颜色通道。
2. 在一个 LIBERO 套件上跑**预训练权重 clean evaluation**，确认随机种子、任务数、语言指令、成功判定、8 步开环动作队列与 action unnorm key。
3. 跑同 checkpoint 的 <code>run_libero_eval_noise.py</code>，固定 corruption type/severity 为 3/4/5；源码会把污染施加于第三人称与腕部画面。先做单因素实验，不要同时调曝光和 action。
4. 严谨比较基线 MLP 与 Fused IB-Adapter：固定 backbone、action head、数据划分、增强、seed、训练步数、动作频率与监督损失。至少 3 seeds，置信区间按 episode/task 分层统计。
5. 再迁移到工业夹爪：安全限速、可复位治具、急停、接管日志；可增设钢件反光、油污、曝光、焦距、单相机失帧的真实扰动。**先评估再训练，不直接在不安全设备上做压力实验。**

## 6. 能否移植到 π₀.₅？哪些是可迁移方法，哪些不是？

对于已有 openpi π₀.₅：视觉编码器后、VLM/动作专家读取视觉前缀之前，**理论上**可以设计一个通道门控 adapter，比较：

$$
\hat A= \pi_{\theta}\bigl(I,\ell,s\bigr)
\quad\text{与}\quad
\hat A'=\pi_{\theta,\phi}\bigl(P_\phi(F_{\mathrm{vis}}(I)),\ell,s\bigr).
$$

但这属于**本手册提出的研究改造方案**，不是 StableVLA 官方发布的 π₀.₅ 插件。关键风险：openpi 的视觉 token 与 prefix KV cache、mask、checkpoint 维度、JAX/PyTorch 路径皆不同；改 Adapter 后必须检查是否需要重算缓存、冻结范围和梯度路径。

围绕工业 EGO/UMI，推荐最小研究矩阵（不是官方实验）：

| 组别 | 视觉表征 | Adapter | 动作训练 | 研究目的 |
| --- | --- | --- | --- | --- |
| A | 原生 π₀.₅ 视觉路径 | 原始 | 固定同样数据 LoRA | 闭环基线 |
| B | 原生 π₀.₅ 视觉路径 | 新增轻量 IB 通道模块 | 与 A 相同 | 单独隔离 Adapter 效果 |
| C | 与 A 相同 | 原始 | 训练端视觉增强 | 判断增强是否已足够 |
| D | 与 A 相同 | IB 模块 | 与 C 相同增强 | 判断两者是否互补 |

如果引入 DINOv3 或 Depth Anything 多教师，**必须另外设 E/F 组**，先对齐空间 patch、冻结教师并统计对原生 action loss 的梯度夹角；特征蒸馏和 IB 门控不是同一个研究变量。余弦近零只表明一阶梯度近乎正交，不代表“两个任务完全互不影响”。

**EGO 人类视频只有图像与人手轨迹时，缺少对应机器人 action 真值**。不能直接拿未重定向的人手末端当作 LIBERO 7D action；可以先作为视觉鲁棒性自监督、语义/目标监督或经物理可达性检验的 H2R 重定向数据，详见 [EGO RGB-only→π₀.₅](../../04-human-to-robot/ego-rgb-only-to-pi05/)。

## 7. 一个可执行的 CPU 数学契约，不需要把 GPU 结果编出来

仓库内 [stablevla_ib_contract_cpu.py](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/stablevla_ib_contract_cpu.py) 只用 Python 标准库：

- 验证输入 $[N,C]$ 在 2 heads/2 channels 的合成形状下，通道 Gram 的输出是 $[H,D,D]$ 而不是 patch attention $[N,N]$。
- 对 token 做相同置换：Gram 不变，输出随 token 同序置换；这是 $Q^TK$ 的代数性质，**不是官方模型训练成功率**。
- 验证 $\sigma(0)=0.5$、独立 Sigmoid 行和通常不等于 1，反驳“零相关就彻底滤掉”的过度描述。
- 验证代码 buffer 的 $\tanh(0.3)\simeq0.291313$；验证作者 LIBERO-Spatial-S5 数值的百分点算术。

~~~bash
python3 examples/stablevla_ib_contract_cpu.py
~~~

**检查边界**：为明确数学结构，CPU 演示把 Q、K、V 的权重都简化为恒等映射，不计算作者 checkpoint 输出；它验证的是**矩阵维度、等变性和数值边界**，不是对作者网络的数值等价复现。实际效果必须由真实模型和环境验证。

## 8. 反例、局限和面试追问

**反例一：相关噪声**。若镜头油污覆盖固定区域并长期存在，噪声可能与物体轮廓共同移动，通道相关性很高；Gram 不会神奇地识别“哪个物理通道是噪声”。因此，门控的抗噪能力仍来自训练得到的统计规律。

**反例二：关键信息被抑制**。对于衬套压装/电泳前悬挂，毫米级定位或接触征兆可能是弱信号，过强的通道抑制会减少细粒度几何信息。评估应补充定位误差、首次接触时刻误差、成功率、掉件和急停率，而不只看模型输出的语义可视化。

**反例三：单一污染基准不能覆盖真机退化**。人工 Gaussian noise/blur 不等同 rolling-shutter、宽动态反射、遮挡、工具震动与相机时延；同一基准的纯视觉成功率不能等价为工业部署安全性。

**4 个面试问题**：为什么 $QK^\top$ 在这里是通道 Gram 而不是 patch attention？为什么 Sigmoid 不保证真正在统计意义上“滤噪”？为什么固定 0.3 的 buffer 不能写成可学习 gate？为什么在 StableVLA 上证明的视觉 adapter，不能说已经在 π₀.₅ Flow Matching 上成功？

## 9. 参考来源与可复现边界

- [正式论文：PMLR ICML 2026](https://proceedings.mlr.press/v306/fu26x.html)；[arXiv](https://arxiv.org/abs/2605.18287)；[作者图表与真机演示](https://dagroup-pku.github.io/StableVLA/)。
- [官方仓库固定树](https://github.com/DAGroup-PKU/HumanNet/tree/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA)；[EfficientChannelAttention/FusedFANProjector](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/prismatic/util/nn_utils.py)；[HF Projector](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/prismatic/extern/hf/modeling_prismatic.py)。
- [源码训练入口](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/vla-scripts/finetune_fused.py)；[扰动评估](https://github.com/DAGroup-PKU/HumanNet/blob/1ef11ca91422ad1f60365b54ee54ec74d745bbf4/src/model/StableVLA/experiments/robot/libero/run_libero_eval_noise.py)；[官方模型权重](https://huggingface.co/DAGroup-PKU/StableVLA/tree/main)。
- **区分事实**：实验成功率来自作者；通道矩阵与 fixed gate 来自锁定源码；IB 信息论解释与 π₀.₅ 移植为教学分析；CPU 合成测试只检验契约。**未完成作者模型本地复现、GPU 训练或真机验证**。
