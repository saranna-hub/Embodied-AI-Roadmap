---
title: "EgoWAM 源码级教程：为何 DINO 与去相机运动的 3D Flow 比像素预测更适合人机共训"
description: "CoRL 2026 / arXiv v2 的论文、官方代码和配置核查：HPT 双头世界动作模型、两条 Flow Matching 梯度、相机运动补偿、Zarr 数据合同、DINO/3D Flow 对照、复现与工业 π₀.₅ 迁移边界"
sidebar:
  order: 2
---

> **更新：2026-11-11｜论文：2026-07-08 首发、2026-10-13 v2｜Georgia Tech / CoRL 2026。** 下文标识 **【论文】** 表示作者实际报告，**【源码】** 表示当前官方仓库公开实现或配置，**【教学推导】** 表示便于理解的独立数学分析，**【提议实验】** 不代表作者或本站已做过。本站核查了源代码与训练文档，并提供**无 GPU 的合成合同脚本**；没有跑过 EgoWAM 真正的模型训练、GPU 推理或真机回放。
>
> 资料： [论文 v2 / 公式与附录](https://arxiv.org/html/2607.08436v2) · [作者实验展示](https://gatech-rl2.github.io/egowam.github.io/) · [官方仓库](https://github.com/GaTech-RL2/EgoWAM) · [官方训练文档](https://github.com/GaTech-RL2/EgoWAM/blob/main/docs/training.md) · [官方数据文档](https://github.com/GaTech-RL2/EgoWAM/blob/main/docs/data.md)。前置：[Flow Matching](../../00-prerequisites/flow-matching/) · [WING 的相机运动解耦](../../04-human-to-robot/wing-ego-spectral-transfer/) · [π₀.₅ 视觉蒸馏](../vla/pi05-vision-distill-gradient-geometry/)。

# EgoWAM：人类动作学不像，也可以让人类数据教会模型“世界怎么变”

把人的双臂示教直接混进机器人行为克隆有一个常见问题：人可以伸手绕过瓶颈抓杯子，机器人夹爪只能从正面进入。哪怕把两者都转为末端 6DoF + 夹爪的 14 维动作，**数值空间相同也不等于动作条件分布相同**。人类抓法、视角、速度、可达空间仍可能冲突。

EgoWAM 的关键不是再提高一套动作重定向精度，而是将**任务相关的未来世界变化**作为第二种监督送入**共享 Transformer 主干**，在人类动作不可靠时仍利用其视频中的物体变化、交互进度和环境几何。

![EgoWAM HPT 双路径架构：人机数据共训，动作与未来世界的梯度汇入共享 trunk](/Embodied-AI-Roadmap/diagrams/egowam-system.svg)

*图 1｜原创教学架构图。世界头只在训练时参与，推理时被移除。主干采用 HPT，并不是把一套完整视频扩散模型串接到 π₀.₅ 后面。*

## 0. 先看四条结论，防止把论文解读跑偏

1. **【论文】相同主干、动作头和数据配比**下比较 Pixel、Pixel-PT、DINO 和 3D Flow。DINO 对未见物体/场景更强，3D Flow 对位姿/空间扰动及 ID 操作更强；作者报告 OOD 最高相对提升 4 倍、3D Flow ID 提升约 20–30%，不是“任何工业场景成功率必然 +30 点”。
2. **【论文】训练时用两种监督；部署时仅保留动作头**。没有在线 rollout 时的视频生成计算开销；不能据此证明世界头可用于推理时规划、失败反思或恢复。
3. **【源码】旗舰视觉输入 stem 默认是 ResNet-18，不是 DINO**；名为 DINO 的世界头，是预测**未来帧 DINOv2 patch features**。将 DINOv2 换成视觉编码器又是**独立消融**，切勿把这两者混为一谈。
4. **【论文】收益主要是上下文、物体/场景泛化**；作者明确承认尚未证明从人类数据学到全新运动原语，更没有在一个统一多任务策略上验证所有结论。

## 1. 为什么让模型看“结果”比强迫复制人手动作更有用？

设人类与机器人数据分别为：

$$
\mathcal D_H = \{(o_t^H,a_t^H,s_{t+T_H}^H)\},\qquad
\mathcal D_R = \{(o_t^R,a_t^R,s_{t+T_R}^R)\}.
$$

标准跨本体行为克隆对同一个编码器和动作解码器优化：

$$
\mathcal L_{\rm BC}=\mathcal L_{\rm action}^{H}+\mathcal L_{\rm action}^{R}.
$$

即使对齐到相同的 14 维末端空间，同一张“杯子靠近碟子”的图像在人类与机器人侧的最优动作也可能不同。假如网络同时学习两种相互冲突的抓取轨迹，就可能产生平均化、夹爪碰撞或不可执行的动作。

【论文】EgoWAM 保留这条强动作基线，再增加一条世界监督通道：

$$
z_t=f_\phi(o_t),\quad
\mathcal L_{\rm WAM}
=\underbrace{\mathcal L_{\rm action}^{R}+\mathcal L_{\rm action}^{H}}_{\text{末端动作预测}}
+\lambda\underbrace{(\mathcal L_{\rm world}^{R}+\mathcal L_{\rm world}^{H})}_{\text{未来世界预测}},\quad\lambda=1.
$$

重要的是**世界分支的梯度会流回共享主干**，并不是额外训练一个与机器人策略完全隔离的视频模型。用链式法则可写为：

$$
\nabla_\phi\mathcal L_{\rm WAM}
=J_{f_\phi}^{\top}\!\left[
\nabla_{z}\mathcal L_{\rm action}
+\lambda\nabla_{z}\mathcal L_{\rm world}\right].
$$

因此训练期世界头改变了主干“要保留什么视觉/几何信息”的偏好。**不等于两条梯度总是协同**：如果世界标签是头部晃动、光照变化或视觉幻觉，它同样会污染表示。这也解释为什么未来世界的目标空间比“多加一个 loss”更重要。

### 1.1 与 Ego-Pi、WING、π₀.₅ 的边界

| 方法 | 人类数据发挥作用的入口 | 机器人动作 GT | 推理是否额外生成世界 |
| --- | --- | --- | --- |
| 直接 BC 共训 | 人手/机器人统一动作监督 | 需要 | 否 |
| Ego-Pi | 人机共享的视觉与动作训练 | 人类动作需有效映射，机器人示教仍关键 | 否 |
| WING | 去相机运动的 latent action + 低频谱引导 | 机器人动作用于最终监督 | 依设计，不等同 EgoWAM |
| **EgoWAM** | **未来帧 DINO 特征/3D 运动的辅助监督** | **机器人动作仍必要** | **否，世界头推理丢弃** |
| 直接 π₀.₅ LoRA | 机器人动作块 Flow Matching | 需要 | 否 |

这使 EgoWAM 特别适合研究**大量 EGO RGB 如何辅助少量机器人动作学习**；但并没有证明纯 RGB 视频在没有机器人示教时可以直接训练出能执行的机器人动作策略。

## 2. 模型结构：HPT token 从哪里来，到哪里去？

【论文+官方配置】主干是 Heterogeneous Pretrained Transformer（HPT）。不同本体的视觉、状态经自己的 stem 投射到同一 token 宽度，再经过共享 trunk。训练时有**可学习的 action query 和 future query**。

| 变量 | 论文旗舰设置 | 含义 |
| --- | --- | --- |
| 机器人 EGO RGB | 当前一帧，另有左右腕部图像 | 人机共享第一视角；机器人额外有双腕相机 |
| 人类 EGO RGB | 当前一帧，Project Aria | 人类无机器人腕相机 |
| 状态向量 | 机器人一侧 14D，另一侧按 stem 配置 | 需 per-domain keymap，不可照搬字段 |
| shared trunk | 16 blocks，256 hidden，8 heads | 域嵌入 + 共同注意力 |
| action queries | 64 个 trunk token | **不是**输出动作块长度 |
| future queries | 16 个 trunk token | 训练期世界查询 |
| action head | 6-block CrossTransformer，128 hidden，4 heads | 预测未来 **100×14** 动作 chunk |
| world head | Pixel / DINO / 3D Flow 可替换 | 训练期预测未来状态/未来运动 |
| action sampler | 50 步 velocity/Flow Matching | 论文值，依赖具体实现 |
| 控制 | 作者平台 30 Hz | 不应照搬到另一种控制驱动 |

特别注意：**64 个 query token ≠ 100 个动作时间点，16 个 future token ≠ 16 个视频未来帧**。Query 数是 trunk 的内部学习参数；动作块长度和未来目标语义由 head 与数据转换决定。

【源码】可在下列位置追踪完整计算链：

- <code>egowam/hydra_configs/model/hpt_wm_cotrain_3dflow.yaml</code>：<code>trunk.embed_dim=256</code>、<code>future_horizon=16</code>、<code>wm_prediction_mode=joint</code>、<code>training_stage=3</code>、<code>action_domains=null</code>。
- <code>egowam/algo/hpt_wm.py</code>：HPT 模型、主干与辅助头分发；检查未来标签在哪里进入 loss。
- <code>egowam/models/fm_policy.py</code> 与 <code>point_flow_decoder.py</code>：动作流匹配/点条件流解码。
- <code>egowam/rldb/zarr/point_flow_loader.py</code>：1120 点网格、轨迹读取、有效性掩码与时间采样。

这些是官方路径，不是本站另写的“伪源码”。建议在实验时 pin 具体 Git commit，并导出 Hydra 完整配置。若官方更新了默认值，应以**本次实验 pin 的 commit**为准。

## 3. 两种 Flow：动作生成和未来 3D Flow 不能混淆

### 3.1 动作 Flow Matching

【论文】对动作真值 $a$、标准高斯噪声 $\epsilon$、采样时刻 $\tau$：

$$
\epsilon\sim\mathcal N(0,I),\quad \tau\sim\operatorname{Beta}(1.5,1),\quad
a_\tau=(1-\tau)\epsilon+\tau a,\quad v^*=a-\epsilon.
$$

预测速度场的教学等价损失：

$$
\mathcal L_{\rm action}=
\mathbb E_{\tau,\epsilon,o,a}
\left\|v_\theta(a_\tau,\tau;f_\phi(o))-(a-\epsilon)\right\|_2^2.
$$

这里 $a$ 是**100×14 的末端动作轨迹**，不是每个像素的光流。网络通过积分速度得到动作块；动作 $a$ 的单位、SE(3) 参数化、量纲归一化仍要与控制接口一一对应。对于固定 $\epsilon,a$ 的线性路径，真实速度与 $\tau$ 无关；实际网络预测会随状态和 $\tau$ 改变。

注意：**论文不同世界头可能采用不同的训练输出语义**；不能将 VAE/RAE 分支的噪声预测误写成所有分支都输出速度场。下一节单独列出。

### 3.2 3D Flow 是物体点相对于同一参考相机的三维位移

![EgoWAM 3D flow 运动补偿与掩码/损失细节](/Embodied-AI-Roadmap/diagrams/egowam-3dflow-detail.svg)

*图 2｜左：只旋转头部时，未补偿点轨迹产生伪运动；右：统一到 t 时刻坐标系后的未来三维位移和有效点监督。所有箭头均为教学重绘。*

【论文】假设 $T_{WC}(t)$ 是当前相机到世界的变换，当前点和未来点分别为 $X_t^{C_t}$、$X_{t+T}^{C_{t+T}}$。将**未来点搬回当前相机坐标系**：

$$
\widetilde X_{t+T}^{C_t}
=T_{WC}(t)^{-1}T_{WC}(t+T)X_{t+T}^{C_{t+T}}.
$$

三维世界变化标签为：

$$
F_{t\to t+T}^{C_t}=\widetilde X_{t+T}^{C_t}-X_t^{C_t}.
$$

若被跟踪的点在世界中静止，不管采集者转头还是平移，理想情况下 $F\approx0$；若物体被移动 5 cm，正确对齐后该点应保留约 5 cm 的运动。**这不是“2D 光流乘深度”的随意近似**：前提是 3D 跟踪点一致、两个时刻相机位姿方向正确、标定与同步合格。

【教学推导】令世界点 $P$ 静止，则 $X_t^{C_t}=T_{WC}(t)^{-1}P$，且 $X_{t+T}^{C_{t+T}}=T_{WC}(t+T)^{-1}P$。代入得到

$$
\widetilde X_{t+T}^{C_t}=T_{WC}(t)^{-1}P=X_t^{C_t}
\quad\Rightarrow\quad F=0.
$$

如果错误地将未来点直接减当前点，就会把相机运动算成场景运动。角度、平移和单位尺度出错会造成系统性伪流；尤其 RGB-only 单目轨迹缺少可靠米制尺度时，不能直接声明补偿获得真实米制 3D 位移。

### 3.3 3D Flow world head 的目标、形状与 Mask

【论文+源码】Track4World 输出的点轨迹在同一参照系下构造目标。官方 <code>docs/data.md</code> 与 <code>point_flow_loader.py</code> 给出固定的 **28×40=1120 个锚点**，每点有三维坐标：

| 字段 | 典型 shape | 使用位置与风险 |
| --- | --- | --- |
| <code>sampled_uv.npy</code> | [1120,2] | 当前帧像素 anchor，只用于定位，不是 3D 真值 |
| <code>sampled_points_t.npy</code> | [T,1120,3] | 当前点 3D，存在 NaN/失效点 |
| <code>sampled_points_tk.npy</code> | [T,1120,3] | 相隔 k 的未来点 |
| <code>future_flow_3d_points</code> | [B,H,1120,3]（变换后示意） | 模型预测目标；实际 H 依 transform |
| <code>__current_points</code> | [B,1120,3] | 点条件 query |
| <code>__validity_mask</code> | 有效点/时间掩码 | 必须覆盖丢跟踪、边界、NaN |
| <code>metadata.json</code> | 帧索引、frame_skip 等 | 防止跨 episode 拼接 |

对于有效标注位置的**教学版**流匹配损失：

$$
s_\tau=(1-\tau)\epsilon+\tau F,\quad
\mathcal L_{\rm 3D}=
\frac{\sum_{b,h,n,d}M_{bhn}\left(u_\psi(s_\tau,\tau,z,q)-(F-\epsilon)\right)_{bhnd}^{2}}
{3\sum_{b,h,n}M_{bhn}+\delta}.
$$

$M$ 表示同时满足可见、有限值、合法时间及重采样约束的 mask；$\delta$ 防止空批次分母为零。**这是本站的可检验教学合同**，不能冒充作者源码的逐字损失实现。官方使用 PointFMPolicy + point-conditioned decoder，独立检查其 normalization/mask 细节才可精确复刻。

## 4. 三种未来世界目标究竟在教 trunk 什么？

【论文】保持主干、动作头、人机混合比例一致，主要替换未来目标与对应 head。目标不同意味着**相互比较的 loss 绝对值没有直接意义**。

| 分支 | 训练标签 $s_{t+T}$ | 优点 | 失效原因 | 官方资产 |
| --- | --- | --- | --- | --- |
| Pixel | 冻结 Wan/Cosmos VAE 的未来帧隐变量 | 保留纹理、直观可视化 | 学到背景、光照、相机晃动，跨域负担大 | Cosmos tokenizer 权重为 gated |
| Pixel-PT | 同一 VAE 目标，预训练 Wan2.1-1.3B 的头 | 更强视频生成先验 | 大模型也可能幻觉“物体已完成动作” | 公开 Wan2.1 权重 + gated tokenizer |
| DINO | 冻结 DINOv2-B 的未来 patch features | 物体语义/场景泛化好 | 特征仍绑定图像 patch，没完全消除头动 | RAE stage1 与 upstream RAE 权重 |
| 3D Flow | 去相机运动的每点 3D 位移 | 更接近物理变化、空间鲁棒 | 对 SLAM/VIO、点追踪、遮挡/尺度敏感 | Track4World sidecar；训练时不需 gated VAE |

**Pixel 的目标不是裸 RGB L2，DINO 的目标也不是 DINO 分类 logits。** DINO 是未来帧的每 patch 特征；论文附录明确采用 DINOv2-B、丢弃 CLS/register token 并做 per-token LayerNorm，输出栅格为 $16\times16\times768$。RAE 辅助的 DiT/denoiser 仍参与训练，但运行时的 action 路径不依赖这个头。

【论文】VAE 与 DINO 分支采用噪声预测型损失：

$$
\mathcal L_{\rm VAE/RAE}=\mathbb E\|\epsilon-\epsilon_\psi(s_\tau,\tau,z)\|^2,
$$

而 3D Flow 分支使用**速度场**监督：

$$
\mathcal L_{\rm Flow}=\mathbb E\|u_\psi(s_\tau,\tau,z,q)-(F_q-\epsilon_q)\|^2.
$$

这两种参数化**即便共用线性插值过程，也不是直接一样的网络输出**。移植时必须检查预测类型、时间方向和损失系数，否则会把监督信号的符号与尺度弄反。

### 4.1 结果应如何阅读？

【论文】作者在杯子放碟、叠衣、装购物袋三项双臂任务上比较 robot-only、1:1 in-domain 人类数据及约 10:1 自然 EGO 数据。每任务约 300–360 条机器人示教；每种配置覆盖 ID、未见物体与未见场景。作者报告：在跨人机动作未对齐数据中，BC 共训可比 robot-only 更差；DINO / 3D Flow 保留收益，DINO 在 OOD 上最高达 4 倍相对改善，3D Flow 的 ID 改善更突出。

**4× 是相对倍率，不是多出 4 倍成功率的“百分点”；也不意味着 4× 对所有任务和数据配比普遍成立。** 因果归因还有世界头结构不同的残余混杂因素；作者附录对比参数量说明容量并不能单独解释排名，但不等于已做严格同参数量的所有消融。

更值得警惕的是**漂亮的像素预测可能是错误世界状态**。论文袋装商品任务中，预训练 Pixel-PT 可能“幻觉”袋子在抓取前已经打开，导致策略过早进入抓取下一阶段。工业过程的“下一帧好看”与“状态转移物理正确”不是同一指标。

## 5. 从 EGO/UMI 采集到训练张量：完整数据合同

【官方】EgoWAM 训练前将机器人数据和 Aria 人类数据整理为 **每 episode 一个 Zarr**；真实机器人有双腕相机、末端状态与命令，人类 Aria 有手 21 点 keypoints、头位姿和 RGB 时间戳。官方 <code>egowam/rldb/embodiment/</code> 通过 keymap 把这些物理来源转换为模型输入。

时间、坐标、动作和标签是四个独立的合同：

1. **因果观测**：$o_t$ 只能取 $t$ 及之前有时间戳证明的数据。未来 $I_{t+T}$ 可用于世界头**训练标签**，禁止通过编码器输入或缓存泄漏到推理条件。
2. **相机-世界变换**：记录 <code>T_world_camera</code> 方向、单位、采样时间和标定版本。EGO 相机运动补偿不能直接套用手持 UMI 的 gripper pose 变换。
3. **动作空间**：每臂 6DoF pose + 1 gripper 命令，共 14 维；不能把无动作 RGB 视频伪装成有效 14D。论文中人机用不同持续时长窗口：人 1.0s（30 帧）、机器人 1.5s（45 帧），都重采样为长度 $k=100$ 的 chunk。
4. **有效标签**：对失跟踪、遮挡、NaN/Inf、边界窗口分别记录 mask。3D 点位置侧车和图像 episode 必须按源片段 ID + 时间戳绑定，不能只靠数组行号。

【论文】动作各维的 1%/99% 分位数映射到 $[-1,1]$，可写为

$$
\widetilde a=2\frac{a-q_{0.01}}{q_{0.99}-q_{0.01}}-1,\qquad
a=q_{0.01}+\frac{\widetilde a+1}{2}(q_{0.99}-q_{0.01}).
$$

分位数必须只从训练 split 计算，记录 domain 和版本。若 $q_{0.99}=q_{0.01}$ 必须定义零跨度退化策略；测试时不要偷偷用验证集重估。反归一化之后，还要经过关节/末端工作空间与速度限制检查。

## 6. 源码复现：先无 GPU 验证，再做真正训练

### 6.1 CPU 数学与标签测试（本站提供，可直接运行）

仓库内：<code>examples/egowam_world_target_contract_cpu.py</code>。它只用 Python 标准库，以**构造的已知旋转和平移**测试静态点补偿为零、物体位移保留、错误帧对齐负例、CFM 线性路径、mask 不吸入无效值、归一化往返和因果性：

~~~bash
python3 examples/egowam_world_target_contract_cpu.py
~~~

这不是 EgoWAM 模型训练，也不读取作者真实数据；它的意义是让容易被忽视的坐标、掩码与信息泄漏错误在训练前立即报错。

### 6.2 官方代码可执行路径（需要作者数据和 GPU）

【源码】官方仓库的 <code>README.md</code>、<code>docs/install.md</code>、<code>docs/data.md</code>、<code>docs/training.md</code> 提供下述命令。请先确认数据授权、依赖版本与存储资源，不能仅复制命令就认定已复现：

~~~bash
git clone https://github.com/GaTech-RL2/EgoWAM.git
cd EgoWAM
uv venv emimic --python 3.11
source emimic/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
export EGOWAM_DATA_ROOT=/path/to/zarr_datasets

# 同时训练人+机器人动作，约 1% debug 数据，官方 BC smoke test
python egowam/trainHydra.py --config-name=train_zarr.yaml \
  data=debug model=hpt_bc_cotrain trainer=debug \
  name=debug description=smoke

# 3D Flow，共训；必须显式加 flow3d=point3d
python egowam/trainHydra.py --config-name=train_zarr_wm.yaml \
  data=debug_wm3dflow model=hpt_wm_cotrain_3dflow \
  flow3d=point3d trainer=debug \
  name=debug description=smoke
~~~

**重要障碍**：官方 debug 数据仍需数据根目录及 resolver/授权配置；3D Flow 需要 Track4World sidecar，首次可能花较久时间计算或读取统计值，不能把“脚本能 import”当模型端到端成功。作者描述训练环境是 Python 3.11、PyTorch 2.5、CUDA 12.1、A40/L40S；四张 RTX 6000 Ada 48GB 是可能的实验资源，但未在本站验证吞吐量与显存。

### 6.3 DINO 目标的两个完全不同的“DINO”

若做论文的 DINO world target，官方示例：

~~~bash
python egowam/trainHydra.py --config-name=train_zarr_wm.yaml \
  data=debug_wm model=hpt_wm_cotrain_rae trainer=debug \
  model.robomimic_model.wm_prediction_mode=joint \
  model.robomimic_model.training_stage=3 \
  'model.robomimic_model.auxiliary_ac_keys.eva_bimanual=[future_image_latent]' \
  'model.robomimic_model.auxiliary_ac_keys.aria_bimanual=[future_image_latent]' \
  model.robomimic_model.auxiliary_loss_weights.future_ee_pose=0.0 \
  model.robomimic_model.auxiliary_loss_weights.future_flow=0.0 \
  name=debug description=smoke
~~~

它依赖 RAE stage-1 decoder、RAE upstream checkpoint 及冻结 DINOv2 特征。**DINO world target ≠ 视觉输入 stem 改为 DINO**。官方旗舰实验使用 ResNet-18 输入 stem；独立的 DINOv2-stem 实验还需要将 256 宽扩大到 768，并修改消费这些特征的 cross-attention bridge，官方提供专门的 SLURM launcher。**不要只改一个 encoder 类名后继续引用原维度**。

【源码】<code>hpt_wm_cotrain_3dflow.yaml</code> 还将 <code>cosmos_vae: null</code>，明确 3D Flow 分支不需那个 gated VAE；而 Pixel 分支必须预检 Cosmos tokenizer 权限。官方仓库某些公开数据文档可能包含旧式访问配置示例；**不得把任何密钥复制进笔记、提交、日志或脚本**，应通过项目提供的正式凭证渠道申请。

### 6.4 真正复现前的 Gate

| Gate | 检查依据 | 未过时怎么处理 |
| --- | --- | --- |
| 数据可达 | Zarr episode + source ID + 时间戳 + action units | 中止模型训练 |
| 3D Flow 可用 | 每条 sidecar 的点数/长度/有效率/尺度 | 不可用则先做 DINO |
| 数据集隔离 | 同一源视频/任务物体不能跨 train/test | 按原片段 ID 重切分 |
| Head/时间一致 | 100×14 动作、H=100 流与 query 数不混淆 | 明确报错 |
| 负例测试 | 故意颠倒相机变换时误差必须显著增加 | 先修外参 |
| 复现统计 | ID/OOD 指标、随机种子、成功标准和 95% CI | 不声明复制论文效果 |
| 部署安全 | 校验动作反归一化、限位、速度/碰撞 | 未通过则只离线评估 |

## 7. 如何迁移到工业 EGO/UMI 与 π₀.₅（研究方案）

**这是建议路线，不是已移植代码。** EgoWAM 是 HPT，不是官方 π₀.₅ 的某个即插即用选项。合理迁移应从**视觉与动作监督解耦**开始，而不是把两套整个网络拼起来。

**阶段 A：建立基线**。固定两代表仿真任务、同一批机器人示教、同一 π₀.₅ LoRA 初始化和训练步数；先测 robot-only 的成功率和外观/场景 OOD，记录任务成功、接管、时间、失败归因。

**阶段 B：低成本表征目标**。从 EGO RGB 帧构造未来 DINO patch features。训练时用只见当前帧的视觉 Adapter 预测未来特征，世界损失回传到共享/adapter 参数；机器人 Action Flow Loss 原样保留，人类视频没有有效动作时该批次必须设置 <code>action_mask=0</code>。这是“人类只监督世界”的最小可验证方案。

**阶段 C：几何目标**。仅对有时间对齐的 VIO/SLAM、相机外参和有效 3D 点轨迹的 EGO/UMI 追加未来 3D Flow 标签。小规模比对静态背景的流中位数（应接近 0）、物体运动的可重复性和跨设备标定差异；不要用未校准的单目深度冒充真实米制变化。

**阶段 D：控制变量消融**。至少四组：R0 robot-only、R1 robot + human BC（仅有效动作）、R2 robot action + human DINO world-only、R3 robot action + human 3D Flow world-only；可增 R4 action+world joint human。固定人机数据抽样比例、总 optimizer steps、共享参数数量（或显式报告差异）。在每组上同时测 OOD 和 ID，不能只报训练 loss。

**阶段 E：长期记忆/纠正闭环**。执行失败记录用于新增任务状态和后训练数据；EgoWAM 的世界头**没有**失败自动归因器，也**没有**强化学习奖励机制，不应把未来状态预测与成功/失败判定直接画等号。可将状态识别/纠错独立成模块，并通过外部环境真实回馈评测。

### 7.1 研究假设的可证伪标准

若 DINO-only 人类数据组的 OOD 成功率、场景扰动表现不优于 robot-only；或者加入 3D Flow 后静态背景伪运动变大/动作任务受损，那么“世界监督改善迁移”的假设在本项目任务上就**未被验证**。此时优先核对时间对齐、相机补偿、监督权重、点有效率和人机数据采样比，而不是把 loss 降低当成功。

对于 20 万小时数据规模的规划，应首先分层统计 **Video-only / Action-valid / Geometry-valid** 的实际可用小时数，单独记录相机轨迹质量、动作监督有效率、场景覆盖和重定向质量；并从小样本 controlled ablation 估算收益，不能从论文外推工业成功率。

## 8. 局限、复现边界与延伸阅读

**作者承认的边界**：一任务一策略，尚未证明多任务统一模型的扩展规律；世界预测主要改善上下文和泛化，未证明人类数据创造新的动作原语；3D Flow 依赖几何跟踪与相机姿态。作者附录在 486 条杯子任务人类视频上比较 RGB-only 相机轨迹与 Aria VIO：经 Sim(3) 对齐后绝对轨迹误差约 1.4 cm，1 秒平移误差约 1.8 cm；**不能据此说 RGB-only 完全替代 VIO 的米制位姿**。

**数据/代码/权重四分法（2026-11-11 核查）**：论文 v2 可公开阅读；[EgoWAM 官方代码仓库](https://github.com/GaTech-RL2/EgoWAM) 开放了训练配置、模型和处理流程；[EgoVerse 数据入口](https://github.com/GaTech-RL2/EgoVerse) 及项目文档有下载条件；[训练 checkpoints](https://huggingface.co/boeyyyy/EgoWAM-checkpoints) 提供模型资产。完整训练数据可能涉及受控资源、存储和第三方 gated 模型；**代码公开 ≠ 所有数据和权重无需授权，更不等于本站已完整训练复现**。

若要进一步学习，可串读 [Ego4WAM 的数据规模/对齐问题](./ego4wam/) → 本文的世界目标选择 → [WING 的交互运动补偿](../../04-human-to-robot/wing-ego-spectral-transfer/) → [π₀.₅ 单教师视觉蒸馏](../vla/pi05-vision-distill-gradient-geometry/)，再做真实任务的对照实验。

### 一手出处索引

- **论文方法与公式**：[arXiv:2607.08436v2，第 3–4 节](https://arxiv.org/html/2607.08436v2)；**作者实验和限制**：第 5–7 节及附录 A/B。
- **公开实现**：[GaTech-RL2/EgoWAM 主仓库](https://github.com/GaTech-RL2/EgoWAM)；[HPT 世界训练模块](https://github.com/GaTech-RL2/EgoWAM/blob/main/egowam/algo/hpt_wm.py)；[3D point-flow decoder](https://github.com/GaTech-RL2/EgoWAM/blob/main/egowam/models/point_flow_decoder.py)；[3D 流数据读取](https://github.com/GaTech-RL2/EgoWAM/blob/main/egowam/rldb/zarr/point_flow_loader.py)。
- **复现命令与依赖**：[训练文档](https://github.com/GaTech-RL2/EgoWAM/blob/main/docs/training.md) · [数据字段](https://github.com/GaTech-RL2/EgoWAM/blob/main/docs/data.md) · [安装/授权边界](https://github.com/GaTech-RL2/EgoWAM/blob/main/docs/install.md)。

**学习检验：** 请口头回答三个问题：① 为什么“未来帧 DINO”不是“DINO 视觉 backbone”？② 为何静态背景在未补偿 3D flow 中会有大位移，正确补偿后应为零？③ 为什么世界头训练中有效、推理时却可以移除？如果三题都能用张量与梯度而非概念作答，才算真正掌握这篇文章的核心。
