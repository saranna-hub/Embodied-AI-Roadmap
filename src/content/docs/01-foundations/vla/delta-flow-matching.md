---
title: "Δ-Flow Matching｜VLA 与 WAM 一步动作生成深读"
description: "arXiv 2610.15794：历史先验、FM/EC 损失、Stop-Gradient、张量维度、实验结果、工业机器人复现"
sidebar:
  order: 5
---

# Δ-Flow Matching：为什么从历史动作出发，未来动作可以一步生成？

> 论文：[Zuo et al., Δ Flow Matching for Robot Foundation Models](https://arxiv.org/html/2610.15794)，arXiv:2610.15794v1，2026-10-11；[作者项目页](https://zuo-kuangji.github.io/delta-flow/)。本文逐项核对论文第 3–7 节及附录 A–E。**本文两张图是教学重绘而不是作者原始架构图；示例代码为最小原理复现，非作者官方训练代码。**

前置学习：[Transformer / QKV](../../../00-prerequisites/transformer/) → [Flow Matching](../../../00-prerequisites/flow-matching/) → [π₀ 深度教程](../pi0/)。

![Δ-Flow VLA/WAM 总体架构图](/Embodied-AI-Roadmap/diagrams/delta-flow-architecture.svg)

*图 1｜从左到右读：历史数据构造 Flow 起点，网络输出速度，右侧使用一次 Euler 更新得到未来动作；下方 WAM 多出视觉预测分支，视觉向动作提供中间特征，动作梯度在连接处被切断。该图是教学重绘，符号及前向路径与论文 Fig. 2 对应。*

## 1. 首先区分“学什么”和“从哪里开始学”

想象机器人前 10 个控制时刻一直向插孔移动，那么未来短时间的动作通常与过去高度相关。传统 Flow Matching（FM）每次生成动作都从独立的高斯噪声出发，学习“噪声到动作”的整段映射；论文提出，既然机器人已执行的最近动作可用，就用它作为起点，并添加适量噪声，学习“历史到未来”的局部变化。

**两个 Δ 不可混为一谈：** Δ-Flow 中的 Δ 是生成起点到未来目标的残差；7D local-delta action 中的 delta 是机器人末端相对于局部参考位姿的动作表征。前者是生成方法，后者是标签坐标定义，完全可以同时使用。

| 组件 | 普通高斯 Flow Matching | Δ-Flow |
| --- | --- | --- |
| 起点 | 高斯随机向量 | 最近已执行动作/历史视觉 latent + 轻微噪声 |
| 网络 | 原来的 VLA 或 WAM | 网络主体可保持不变 |
| 监督 | 采样生成时间的 FM | FM + 一步终点监督 EC |
| 推理 | 多次速度场求值 | 每个模态一次速度场求值 |
| 主要依赖 | 条件观测和指令 | 条件观测 + **可靠的真实历史反馈** |
| 典型风险 | 一步更新误差大 | 执行历史错误时容易继承错误先验 |

### 总图每个方框的输入输出

| 图中组件 | 输入 | 输出 | 能不能出现在部署时 |
| --- | --- | --- | --- |
| 历史动作 | 过去 H 步已执行控制 | 历史 $\mathbf h_a$，形状 $[B,H,D_a]$ | 可以，取机器人实际执行记录 |
| 构造源 | $\mathbf h_a$、Gaussian noise、mask | 起点 $\mathbf s_a$ | 可以 |
| 条件编码器 | 当前 RGB、语言、proprioception | $c_t$ | 可以 |
| 动作 Expert | 当前 Flow 状态、生成时间、条件 | 速度 $\mathbf f_{\theta_a}$ | 可以 |
| 未来动作真值 | 数据集中的下一段控制 | $\mathbf y_a$ | **仅训练/评测可用** |
| WAM 视觉分支 | 观测视觉源与条件 | 视觉预测 latent 与中间特征 | 可以；未来真值视觉不能偷看 |
| EC 终点损失 | 单步生成结果 + 未来动作真值 | L1 训练监督 | 仅训练 |

**极易犯错：** 新一轮动作只执行了前 5 步，就把上次网络规划的另外 5 步也写进 history。论文这里的 action history 是**过去已执行**动作，不是过去网络“曾经打算执行”的完整输出。

## 2. 数学推导：把每个负号讲清楚

论文中 $t$ 表示真实控制时刻，$\tau$ 表示生成流的虚拟时间。采用 **$\tau=1$ 为源、$\tau=0$ 为目标** 的方向，因此推理 Euler 要从 1 往 0 走。

### 2.1 高斯源 FM 为什么预测噪声减动作？

假设未来动作真值 $\mathbf y_a$、标准高斯样本 $\boldsymbol\epsilon_a$：

$$
\mathbf y_a^\tau=(1-\tau)\mathbf y_a+\tau\boldsymbol\epsilon_a.
$$

对 $\tau$ 求导：

$$
\frac{\mathrm d\mathbf y_a^\tau}{\mathrm d\tau}=\boldsymbol\epsilon_a-\mathbf y_a.
$$

网络的训练目标是这个速度场。但推理方向是从 $\tau=1$ 到 0，所以：

$$
\mathbf y_a^{\tau-\delta}=\mathbf y_a^\tau-\delta\,\mathbf f_{\theta_a}(\mathbf y_a^\tau,\tau,c_t).
$$

**注意减号：** 网络输出“源减目标”的速度，推理恰恰要反方向更新，才能逼近目标。如果将速度直接加到源上，方向就错了。

### 2.2 用历史构造源：为什么要有噪声和 mask？

作者对动作 $m=a$ 和视觉 $m=v$ 定义统一起点：

$$
\boxed{\mathbf s_m=\alpha_m\mathbf h_m+\sigma_m\mathbf b_m\odot\boldsymbol\epsilon_m,\qquad
\boldsymbol\epsilon_m\sim\mathcal N(\mathbf0,I).}
$$

其中 $\mathbf h_m$ 为过去的动作或编码视觉，$\mathbf b_m$ 是逐维噪声 mask。动作源在三种 backbone 中均采用 $\alpha_a=1$ 和 $\sigma_a=0.5$，gripper 与 padding 的维度不注入此噪声。**加入轻噪声不是让模型随机乱动**：如果相同历史可能走向不同后续动作，随机起点可帮助表达多种可能；但能否表达多峰分布还取决于数据与模型容量。

用起点重写插值：

$$
\mathbf y_m^\tau=(1-\tau)\mathbf y_m+\tau\mathbf s_m.
$$

训练速度真值：

$$
\mathbf u_m=\mathbf s_m-\mathbf y_m.
$$

FM 损失：

$$
\mathcal L_{\mathrm{FM}}^m=
\mathbb E\left[\left\|
\mathbf f_{\theta_m}(\mathbf y_m^\tau,\tau,c_t^m)
-(\mathbf s_m-\mathbf y_m)
\right\|_2^2\right].
$$

直观地说，如果短期未来与历史接近，那么所需学习的残差通常会较小。**但这不是普遍数学定理**：碰撞后反弹、重规划、抓取状态突然改变时，未来与历史也可能相差很大。

### 2.3 为什么 FM 训练还要 EC？

普通 FM 随机采样 $\tau$，学习流上各点的局部速度；但部署时只执行一步大跨越。为直接监督“一步就走对”，论文定义：

$$
\widehat{\mathbf y}_m=\mathbf s_m-\mathbf f_{\theta_m}(\mathbf s_m,1,c_t^m).
$$

$$
\boxed{\mathcal L_{\mathrm{EC}}^m=
\mathbb E\left[\|\widehat{\mathbf y}_m-\mathbf y_m\|_1\right].}
$$

假设网络准确输出 $\mathbf f=\mathbf s-\mathbf y$，代入可得 $\widehat{\mathbf y}=\mathbf s-(\mathbf s-\mathbf y)=\mathbf y$。因此 EC 为 0。**EC 的输入始终固定在 $\tau=1$；它不是随机时间点上的第二个 FM。**

VLA 只训练动作：

$$
\mathcal L_{\mathrm{VLA}}=
\mathcal L_{\mathrm{FM}}^a+
\lambda_{\mathrm{EC}}^a\mathcal L_{\mathrm{EC}}^a.
$$

WAM 对视觉和动作都训练：

$$
\mathcal L_{\mathrm{WAM}}=
\lambda_v(\mathcal L_{\mathrm{FM}}^v+
\lambda_{\mathrm{EC}}^v\mathcal L_{\mathrm{EC}}^v)+
\lambda_a(\mathcal L_{\mathrm{FM}}^a+
\lambda_{\mathrm{EC}}^a\mathcal L_{\mathrm{EC}}^a).
$$

注意视觉损失通常在 latent 空间、动作损失在归一化 action 空间，两者的数值量纲未必可直接比较。

## 3. Stop-Gradient：视觉特征可以给动作，但动作梯度为什么不能反向改视觉？

论文 WAM 会将视觉网络的中间特征 $\mathbf R_v$ 提供给动作 Expert。定义：

$$
c_t^a=(c_t,\operatorname{sg}(\mathbf R_v)).
$$

其中 sg 是 Stop-Gradient。如果视觉参数为 $\theta_v$，没有 sg 时动作损失沿此连接可能产生：

$$
\frac{\partial\mathcal L_a}{\partial\theta_v}=
\frac{\partial\mathcal L_a}{\partial\mathbf R_v}
\frac{\partial\mathbf R_v}{\partial\theta_v}.
$$

使用 sg 后，这条链上的动作梯度贡献被置零，**但视觉分支自身的视觉损失梯度仍存在**。因此不能把 Stop-Gradient 理解为冻结所有视觉权重，而是：动作读取了视觉特征，却不能通过这一支路反过来修改它。

这一做法特别值得和多任务/多教师梯度冲突研究对照：**loss 权重缩放仅缩放梯度，sg 直接移除某一回传路径**。论文在 ImageWAM 相关消融中报告 future PSNR 从 15.04 dB 提升到 20.86 dB，LIBERO-Plus 从 83.24% 到 84.96%。这不是证明任何系统都必须切梯度，而是说明“共享视觉特征”和“共享反传梯度”是两个独立的设计选择。

## 4. 用三种 backbone 走通张量形状与时间

### 4.1 π₀.₅：仅动作 Δ-Flow

论文保持 PaliGemma 视觉语言主干、Gemma 动作 Expert，没有加入未来视频生成。作者附录中的设置为：

| 张量/参数 | 形状或数值 | 含义 |
| --- | --- | --- |
| 当前视觉 + 指令 + state | 依 pi05_base 原模型 | 作为条件 $c_t$ |
| 最近已执行动作 $\mathbf h_a$ | $[B,10,7]$ | 10 步历史，7 维动作 |
| 未来动作真值 $\mathbf y_a$ | $[B,10,7]$ | 训练目标 |
| 起点 $\mathbf s_a$ | $[B,10,7]$ | 历史 + Gaussian 扰动 |
| Flow 时间 $\tau$ | 每条样本一个标量 | FM 随机抽，EC 固定 1 |
| 网络输出 | $[B,10,7]$ | 每步每维速度 |
| Replan | 生成 10 步、执行 5 步 | 下一次重规划要取真实已执行动作 |

作者的 π₀.₅ loss 权重为 $\mathcal L_{\mathrm{FM}}^a+0.5\mathcal L_{\mathrm{EC}}^a$；action 使用 checkpoint 的 quantile-normalized 空间。

### 4.2 ImageWAM：当前图像特征 → 未来图像 latent + 16 步动作

视觉与动作结构分别为 FLUX.2-klein-base-4B 图像编辑主干、action DiT。论文把外部相机与腕相机各缩放到 $224\times224$，水平拼接为 $224\times448$。当前图像是模型条件，未来第 16 步图像仅作为训练真值；动作目标是连续 16 步，动作历史也是 16 步，部署每执行 12 步再规划。

视觉源有两个被实验比较的比例：

$$
\mathbf s_v=\frac{\mathbf h_v+\boldsymbol\epsilon_v}{\sqrt2}
\quad\text{（50% 历史平方系数占比）}.
$$

$$
\mathbf s_v=\mathbf h_v+0.5\boldsymbol\epsilon_v
\quad\text{（80% 历史平方系数占比）}.
$$

这两个 50%/80% **不是图片像素保留比例或 token Mask 比例**。不能凭输入 RGB 的尺寸推断 latent token 数量；真正的 latent 布局要以 FLUX 的 VAE 和视觉处理实现为准。

ImageWAM 使用的有效损失为：

$$
\mathcal L=
\mathcal L_{\mathrm{FM}}^a+0.5\mathcal L_{\mathrm{EC}}^a+
0.5\mathcal L_{\mathrm{FM}}^v+0.25\mathcal L_{\mathrm{EC}}^v.
$$

### 4.3 Fast-WAM：视频 token 的未来采样有多稀疏？

![Fast-WAM dense 与 sparse 视频时间布局](/Embodied-AI-Roadmap/diagrams/delta-flow-timeline.svg)

*图 2｜教学重绘：Dense 和 Sparse 都预测 32 步动作，仅未来视频时刻采样密度不同。此处的 Sparse8 意为视觉时间 stride=8，不是“只预测 8 步动作”。*

| 配置 | 未来帧相对时刻 | 未来视频帧 | 未来视觉 token | 动作 horizon |
| --- | --- | ---: | ---: | ---: |
| Joint-Dense，stride 4 | 4,8,12,16,20,24,28,32 | 8 | 196 | 32 |
| Joint-Sparse，stride 8 | 8,16,24,32 | 4 | 98 | 32 |

作者使用 Wan2.2-TI2V-5B 视频主干 + action DiT。Dense 包含当前帧共 9 帧、Sparse 包含当前帧共 5 帧，VAE 时间压缩后分别有 2 / 1 个未来 latent temporal slice。上述 196 / 98 只算未来视觉 token，不包括当前条件帧 token 与动作 token。对照论文 Table 7。

Fast-WAM 源动作使用历史 32 步；历史视觉取 $t-32,t-24,t-16,t-8,t$，并投影到未来 visual latent 的布局。其联合 loss 为：

$$
\mathcal L=
\mathcal L_{\mathrm{FM}}^a+0.5\mathcal L_{\mathrm{EC}}^a+
\mathcal L_{\mathrm{FM}}^v+0.5\mathcal L_{\mathrm{EC}}^v.
$$

**注意：三种 backbone 并没有一套通用的未来帧数、动作 horizon、参数比例或相机编码方式。**

## 5. 训练 vs 推理的精确流程

**训练动作分支：** 从同一条 episode 抽取当前 RGB/任务/状态、过去 H 步 *已执行* 动作和紧随其后的未来 H 步动作，全部映射到同一个 action 标准化空间。由历史加噪声得到源、采样 $\tau$ 得插值张量，然后用 FM 和 EC 两个目标监督速度预测。WAM 再加视觉源与视觉目标；动作的视觉条件从视觉网络中间层读取并 detach。

**推理动作分支：** 历史源构造完成后，把它作为 $\tau=1$ 的起点，一次前向得到 $\mathbf f(\mathbf s_a,1,c_t)$，用 $\mathbf s_a-\mathbf f$ 输出完整未来 chunk。执行其中预定的部分，再用实际环境反馈重新做下一轮。

**WAM 推理**：先用视觉源计算未来视觉网络中间特征，动作 Expert 以其为条件；两个分支各走一次 Euler。论文不要求把未来 latent 真正解码为 RGB 给动作 Expert。视觉预测可以发生在 latent 空间中。

## 6. 定量结果：既看成功率也看公平性

| 原论文结果 | 对照数值 | 注意事项 |
| --- | --- | --- |
| ImageWAM 收敛 | 1 epoch 达到 91.30%，基线此时 24.60%；基线到第 6 epoch 才首次超过 90% | 不要把 6× epoch 等同于严格 6× GPU 总时长 |
| Fast-WAM 到达 95% | 2 epoch vs 8 epoch | 按相同任务配置比较 |
| π₀.₅ LoRA | 1 epoch 92.20% vs Gaussian 5 epoch 91.30% | 来自作者项目页，同 LoRA 对照 |
| 一步采样加速 | π₀.₅ 1.54×、ImageWAM 2.49×、Fast-WAM 4.45× | 相对作者设定的多步采样，**不等于实机整体控制环加速** |
| ImageWAM LIBERO-Plus Overall | 82.49% → 88.55% | 泛化提升 6.06 个百分点 |
| 机器人初始化扰动 | 52.90% → 73.94% | 对这一类扰动提升明显 |
| 摄像头视角扰动 | 82.55% → 82.24% | **没有提高**，不能说所有扰动都有收益 |
| local-delta vs absolute（特定 ImageWAM 消融） | 85.82% vs 61.07% | 动作空间表达也影响泛化，不全归因于 Δ-Flow |
| action jerk | 单步 Δ-Flow 较单步 Gaussian 低约 4.5× | jerk 平滑≠接触力安全或机器人任务成功率 |

三组尤其值得复核的消融：

1. 只换动作源 vs 同时换视觉与动作源：ImageWAM 在 LIBERO-Plus 的报告数值 85.82% → 88.55%。
2. 视觉 endpoint consistency：作者 80% 视觉 prior 条件下，移除视觉 EC 使 87.37% → 84.96%。
3. 梯度隔离：Stop-Gradient 相比不隔离，图像 PSNR 和鲁棒成功率都有改善，但这仍需控制其他变量才能迁移到你的模型。

## 7. 最小可运行 PyTorch 单元实验

以下代码只验证公式、shape、mask 和反向传播，**不是复现真实 $\pi_{0.5}$、ImageWAM 或 Fast-WAM 的训练框架**。这个实验不依赖图像，输入为合成动作序列。

~~~python
import torch
from torch import nn

B, H, D = 8, 10, 7
history = torch.randn(B, H, D)       # 已执行动作历史
target = history + .15*torch.randn(B, H, D)
condition = torch.randn(B, 32)
mask = torch.ones(1, 1, D)
mask[..., -1] = 0                    # 教学假定末维为 gripper
source = history + .5*mask*torch.randn_like(history)
tau = torch.rand(B, 1, 1)            # FM: 随机生成时间
y_tau = (1-tau)*target + tau*source
v_target = source-target             # 源-目标，推理要减速度

class ToyField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D+1+32,128),
                                 nn.SiLU(),nn.Linear(128,D))
    def forward(self,x,t,c):
        time = t.expand(x.shape[0],x.shape[1],1)
        cond = c[:,None,:].expand(-1,x.shape[1],-1)
        return self.net(torch.cat([x,time,cond],dim=-1))

model=ToyField()
fm=(model(y_tau,tau,condition)-v_target).square().mean()
one_step=source-model(source,torch.ones_like(tau),condition)
ec=(one_step-target).abs().mean()
loss=fm+.5*ec
loss.backward()
assert one_step.shape == (B,H,D)
assert torch.isfinite(loss)
print('FM:',fm.item(),'EC:',ec.item())
~~~

**实验中要确认：** FM 是 $\mathbf s-\mathbf y$；EC 必须在 $\tau=1$，并做 $\mathbf s-\mathbf f$，不是加号；noise mask 不对 gripper/PAD 施加运动噪声。真实策略在 episode 开头历史不足时必须有明确定义，不应默默把 padding 当作已执行的动作。

## 8. 在工业机器人数据资产和真实控制闭环中怎样用？

可以选机械臂插接或双臂搬运的一个单一场景，**锁定动作维度、工具坐标系、时间同步、归一化、训练集划分、随机种子**，比较以下组别：Gaussian-10 步、Gaussian-1 步、History-Δ-1 步（无 EC）、History-Δ-1 步（带 EC），WAM 额外比较有无视觉未来预测与有无 Stop-Gradient。

除了成功率，必须统计动作 chunk 衔接处位置/姿态跳变、速度加速度和 jerk、接触力峰值、限速触发率、单步 P50/P95 延迟、重启第一段动作的有效率。否则很容易出现“仿真精度更好，但真机机械抖动更严重”。

**对数据采集质量的要求比普通 FM 更敏感**：历史源来自执行记录，UMI/EGO 后处理的时延偏差、漂移、重定位跳变、夹爪状态误标，会进入生成模型起点。无本体数据还需要 Human-to-Robot 动作映射，不能直接把人手位姿当成真实机械臂已执行 action。

## 9. 复现验收与论文局限

| 复现级别 | 交付物 | 不得夸大的结论 |
| --- | --- | --- |
| L0 概念 | 逐式重推 Eq. 3–9 | 看过论文不等于复现 |
| L1 数学实验 | 上述 shape/mask/FM/EC 单测 | 未接真实策略 |
| L2 真实数据 | Action Schema、统一坐标与时间、历史有效性 mask、同步回放 | 尚无端到端策略 |
| L3 matched ablation | 固定参数/数据/任务，评估 Gaussian/Δ/EC | 不能冒充论文原始指标 |
| L4 benchmark | 原 benchmark 与明确版本号、重复试验及可信区间 | 环境/代码不同应标记工程复现 |
| L5 真机 | 时延、稳定性、动作平滑、安全和成功率实测 | 无真机不得宣称工业可部署 |

论文报告的主要是 LIBERO / LIBERO-Plus 仿真；真实接触、失效后重启、动作历史缺失和历史噪声敏感性还没有由这些数字证明。论文将 $\sigma_a=0.5$ 固定在各模型中，工程上建议额外做历史置信度、噪声强度和突变操作的消融。

### 原文对应位置与参考

- [论文 Fig. 1–2 与第 4 节](https://arxiv.org/html/2610.15794)：历史源公式、FM、EC、VLA/WAM 与 Stop-Gradient。
- [论文附录 A.1–A.3](https://arxiv.org/html/2610.15794)：三种模型的图像布局、动作长度、视觉时刻与 loss 系数；尤其 Table 6–8。
- [作者项目页](https://zuo-kuangji.github.io/delta-flow/)：训练收敛图、LIBERO-Plus 类别指标、动作 jerk、LoRA、模型对比与消融。
- [π₀ 深读](../pi0/)：回顾动作 Expert 与传统多步 Flow Matching。
- **开源边界**：项目页含 GitHub 入口，但本文没有独立验证完整 Δ-Flow 训练代码及对应权重已经发布；因此上面的 PyTorch 是**原理等效练习**，不冒充作者完整复现。

**核心结论**：不是“把网络换成更大的 Transformer”，而是让生成从历史附近出发，FM 学残差速度、EC 约束一步终点、WAM 动作分支读取但不反向干扰视觉分支。真机落地首先要保证历史执行反馈与动作标签可靠。
