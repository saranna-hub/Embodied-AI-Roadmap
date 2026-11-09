---
title: "π₀.₅ 视觉蒸馏与梯度几何：DINOv3 教师、残差 Adapter、LoRA 和 Flow Matching 的可证伪实验"
description: "以 openpi 官方 JAX/PyTorch 源码为基线，设计不泄漏教师信息的视觉蒸馏，对比单教师与梯度投影，推导张量、损失、控制变量和 CPU 契约"
sidebar:
  order: 10
---

> **研究状态｜2026-11-09**：这是基于一手源码和公开论文的**研究性实验设计**，不是 Physical Intelligence 发布的新 π₀.₅ 模块，也**没有**本站实测的 GPU / LIBERO / 真机成功率。核查基线：[openpi 官方 main，commit 215abfb217db](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)（2026-11-09 读取 GitHub main 指向该提交）；视觉教师依据 [Meta DINOv3 原始论文](https://arxiv.org/abs/2508.10104) 与 [官方说明](https://ai.meta.com/research/publications/dinov3/)；梯度投影依据 [PCGrad 原始论文](https://arxiv.org/abs/2001.06782)。此页随上游源码变更需要重新审计，禁止将示意伪代码当作 openpi 原生接口。

## 0. 问题定义：为什么“加个视觉教师”并不保证动作更好？

一个机器人策略并不是分类器。假设 π₀.₅ 对金属插头的视觉表征能够区分插头和插座，却没有保留插孔边缘的毫米级位置信息。即使蒸馏后的视觉特征在 DINOv3 相似度上更好，**末端执行器在接触前的修正动作仍可能变差**。相反，深度教师能突出几何，却可能忽略任务上下文；不同监督源在共享参数上可能彼此竞争。

本章围绕三个**需要分别检验**的假设：

1. 冻结 DINOv3 教师，仅在训练中为 π₀.₅ 的视觉 token 提供辅助监督，能改善少样本操作的视觉稳定性。
2. 使用小型残差 Adapter，而不是直接替换 SigLIP 或改写 action expert，可以控制工程复杂度与归因范围。
3. 只有当教师损失与动作损失在**实际共享参数**上稳定出现负梯度内积，PCGrad 式投影才可能减少一阶冲突；余弦接近 0 **不等价**于明显冲突，更不保证蒸馏有收益。

先读 [π₀.₅ × LIBERO 数据、LoRA 和 Flow Loss](./pi05-libero-data-loss-lora-source-guide/)、[π₀ 双专家注意力与 Prefix KV Cache](./pi0-joint-attention-code-audit/) 和 [StableVLA 的信息瓶颈视觉 Adapter](./stablevla-ib-adapter-robustness/)。本篇解决的是**训练期视觉监督如何作用于 π₀.₅ 参数**，不是重复介绍推理接口或把 StableVLA 误写成 openpi。

![视觉教师只在训练期存在；π₀.₅ 推理仍沿原生观测路径](/Embodied-AI-Roadmap/diagrams/pi05-vision-distill-system.svg)

*图 1｜原创研究性设计图。蓝色是 openpi 官方基础路径，绿色模块是本站建议增量，虚线表示仅训练可用的教师监督。教师视频特征不得作为推理输入。*

## 1. 严格分清“已在官方源码实现”和“待研究的插入”

| 组件 | 官方代码能确认的事实 | 本章拟研究的增量 |
| --- | --- | --- |
| 视觉编码器 | <code>pi0.py</code> 构建 SigLIP So400m/14，<code>embed_prefix</code> 对每路有效 RGB 形成视觉 token | 在视觉输出到联合多模态 Prefix 之间加残差 Adapter |
| 视觉输入 | [<code>model.py</code>](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/model.py) 定义 base/left_wrist/right_wrist 三个键和有效性 mask | 新增单教师蒸馏监督，仅用于匹配**实际存在且有效**的相机 |
| 动作专家 | <code>Pi0Config(pi05=True)</code> 支持连续 Flow Matching；源码默认动作容器 32 维、horizon 50 | 不改动作语义；LIBERO 对照仍用 <code>H=10</code> 和有效 7 维 |
| 训练目标 | [<code>pi0.py compute_loss</code>](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L189-L222) 返回逐动作位置上的速度 MSE | 额外加入对齐后的教师特征损失，不替换 Flow Loss |
| 低秩微调 | [<code>pi0_config.py get_freeze_filter</code>](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py#L90-L118) 按参数路径构建冻结过滤器 | 明确记录 Adapter、vision tower、Gemma 与 action expert 的可训练集合 |
| PyTorch 路径 | [<code>pi0_pytorch.py</code>](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py#L188-L231) 的 <code>embed_prefix</code> 调用 <code>embed_image</code> | 研究性封装需同时覆盖训练与推理，默认不提供蒸馏 hook |

**重要版本差异**：π₀.₅ 默认将本体状态离散化为语言 Prefix 输入，并用 AdaRMSNorm 注入时间；但官方 <code>pi05_libero</code> 特意配置 <code>discrete_state_input=False</code>，不能把所有 π₀.₅ 任务视为相同 state token。LoRA 过滤器也**不自动等价于冻结整个视觉塔**：过滤规则的范围要用实际参数树枚举核实。任何“冻结视觉、只训练 Adapter”的实验必须显式检查优化器参数组，不能仅凭配置名下结论。

### 1.1 从像素到动作：张量和依赖关系

设 batch 为 $B$，当前有效相机数为 $V$，视觉 patch 数为 $N$，PaliGemma 隐宽为 $D_v$，教师特征维为 $D_T$，动作长度为 $H$，模型动作容器维度为 $D_a$。

| 数据 | 训练张量 | 推理必须存在？ | 梯度归属 |
| --- | --- | --- | --- |
| 当前 RGB | $I_{b,v}\in\mathbb R^{3\times h\times w}$ | 是 | 不对像素求梯度 |
| SigLIP/PaliGemma Prefix | $E\in\mathbb R^{B\times VN\times D_v}$ | 是 | 按冻结策略决定 |
| 残差 Adapter | $E'=E+\alpha A(\mathrm{LN}(E))$，shape 不变 | 是 | 可训练 |
| DINOv3 教师 | $T(I)\in\mathbb R^{B\times VN_T\times D_T}$ | **否** | 永远冻结，输出 stop-gradient |
| 教师投影头 | $P(E')\in\mathbb R^{B\times VN_T\times D_T}$ | 否 | 训练期可训练 |
| Flow 输入 | 观测 + $x_\tau\in\mathbb R^{B\times H\times D_a}$ | 采样时使用噪声 | Action expert、LoRA |
| Flow 输出 | $v_\theta(x_\tau,\tau,o)$，同 $x_\tau$ shape | 是 | 动作损失反传 |

**这里的 $N_T$ 不要求等于 $N$**。如果两侧预处理尺度、patch stride、裁剪窗口不同，应先为每个视角按图像空间构建对齐映射 $\mathcal W$，再匹配特征，不能把 flatten 的 token 按序硬凑。对于腕部镜头缺帧，保留 <code>image_mask=False</code>；蒸馏损失对该视角的样本权重必须为零。

## 2. 为什么优先小残差 Adapter，而不立即替换视觉主干？

[官方 JAX <code>embed_prefix</code>](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L106-L130) 调用 <code>self.PaliGemma.img</code> 得到已进入 PaliGemma 多模态表示空间的 token；[官方 PyTorch 对应代码](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py#L188-L218) 通过 <code>paligemma_with_expert.embed_image</code> 形成 Prefix。实验上在**这一个可检验的张量接口**插入残差适配器，相比替换整个 SigLIP，能最大限度保留预训练检查点与 action expert 的接口。

选择一个瓶颈残差：

$$
A(E)=W_{\mathrm{up}}\,\phi(W_{\mathrm{down}}E),\qquad
W_{\mathrm{down}}\in\mathbb R^{d\times D_v},\
W_{\mathrm{up}}\in\mathbb R^{D_v\times d}.
$$

$$
E'=E+\alpha A(\mathrm{LN}(E)),\qquad \alpha\in\mathbb R.
$$

可以把 $W_{\mathrm{up}}$ **零初始化**，使训练第一步 $E'=E$。但需注意：若上下两个投影都零初始化，链式梯度可能同时为零；合理方案是下投影随机初始化、上投影零初始化，并对初始步的梯度数值进行检查。

参数量约为 $2D_vd$，不含 bias 与 LN。若 $D_v=2048,d=128$，仅两层线性矩阵约 **524,288** 个参数；这**只是示例参数预算**，不是官方 openpi 新增模块的实测显存。多视角共享 Adapter 能避免为每个相机学习完全独立的坐标系，但训练损失仍必须按相机有效性 mask 归一。

### 2.1 教师为什么只能在训练期出现？

DINOv3 的高质量密集特征为蒸馏提供监督目标。**不意味着机器人必须在推理时运行两个视觉主干**。本设计是 teacher–student：教师看训练帧 $I_t$，生成冻结目标；学生仍只看 $I_t$，通过可训练 Adapter 提升表征。

$$
F_T=\operatorname{stopgrad}\bigl(T(\operatorname{prep}_T(I_t))\bigr),\quad
F_S=\mathcal W\bigl(P(E'(I_t))\bigr).
$$

$$
\mathcal L_{\mathrm{feat}}=
\frac{\sum_{b,v,n}m_{bvn}\,
\|\operatorname{norm}(F_{S,bvn})-\operatorname{norm}(F_{T,bvn})\|_2^2}
{\max(1,\sum_{b,v,n}m_{bvn})}.
$$

$\operatorname{norm}(z)=z/(\|z\|_2+\epsilon)$ 是按特征维做归一；若像素区域是背景、时钟错位或不可信重投影区域，可增加 mask，而不是将其伪装成精确几何真值。

**不可泄漏条件**：老师不能读取后续机器人动作、整段 episode 的未来帧、任务成功/失败标签再作为当前时刻的额外输入；若教师用未来窗口编码为标签，只能明确标记为训练期 hindsight 监督，推理端绝不能读取该窗口。本章默认采用**当前帧**教师，以简化因果性。

## 3. Flow Matching 动作目标与教师损失怎样结合？

锁定官方 [<code>pi0.py</code> 的符号约定](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L189-L232)：$t=1$ 是**噪声**，$t=0$ 是**真实动作**；这与 π₀ 原始论文某些记法方向相反。对动作 chunk $A$、标准高斯 $\epsilon$，

$$
x_t=t\epsilon+(1-t)A,\qquad u_t=\epsilon-A,\qquad
t\sim 0.999\,\mathrm{Beta}(1.5,1)+0.001.
$$

训练时最小化

$$
\mathcal L_{\mathrm{FM}}=
\mathbb E_{o,A,t,\epsilon}
\left[\frac{1}{H D_{\mathrm{valid}}}
\sum_{h=1}^H\sum_{j=1}^{D_a}
m^{a}_{hj}\bigl(v_\theta(x_t,t,o)_{hj}-u_{t,hj}\bigr)^2
\right].
$$

**上式的有效维度 mask 是工业训练改造的建议，不是官方 <code>compute_loss</code> 原生实现。** 官方 JAX 返回 <code>mean((v-u)^2, axis=-1)</code>，因此 padding 的 32 维是否存在 mask，必须追踪数据变换和训练 reducer，不能直接把上式当官方代码。对 LIBERO 7D 控制输出，建议报告“官方 loss”及“研究性 masked loss”两个口径，并把归一化统计与控制 API 固定。

加辅助损失：

$$
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{FM}}+\lambda\,\mathcal L_{\mathrm{feat}},
\quad \lambda\ge0.
$$

这个式子**只定义前向目标，不保证两个梯度兼容**。教师损失若集中于物体类别、颜色或纹理，可能拉动共享视觉表征远离末端位姿的细粒度需求。改变 $\lambda$ 也不是自动“等权学习”：两个损失的参数梯度量纲和尺度可能差几个数量级。

推理时依旧从 $x_1\sim\mathcal N(0,I)$ 出发，按官方逆时间 Euler：

$$
x_{t-\Delta}=x_t-\Delta\,v_\theta(x_t,t,o),\qquad t:1\to0.
$$

这里**没有教师、没有蒸馏投影头、没有训练动作真值**。实际机器人必须用与训练一致的归一化统计反变换，执行的动作 horizon 和采样步数是两组独立超参数。

## 4. 重点：梯度余弦为 0、为负分别说明什么？

令 $\theta_s$ 是**同时被 Flow 损失和蒸馏损失更新的实际共享参数**，如残差 Adapter。定义：

$$
g_F=\nabla_{\theta_s}\mathcal L_{\mathrm{FM}},\quad
g_T=\nabla_{\theta_s}\mathcal L_{\mathrm{feat}},\quad
c=\frac{g_F^\top g_T}{(\|g_F\|_2+\epsilon)(\|g_T\|_2+\epsilon)}.
$$

先按一阶近似：若只沿教师梯度负方向更新 $\theta'_s=\theta_s-\eta g_T$，

$$
\mathcal L_{\mathrm{FM}}(\theta'_s)-\mathcal L_{\mathrm{FM}}(\theta_s)
\approx-\eta g_F^\top g_T.
$$

- $c>0$：教师方向一阶上倾向于**降低** Flow 损失，但不代表验证成功率必提高。
- $c<0$：教师方向一阶上倾向于**升高** Flow 损失，是方向冲突；小负值在噪声和有限 batch 中可能不稳定。
- $c\approx0$：局部方向近乎正交，表示**直接一阶干扰很小**，既不能称“强冲突”，也不能说两个目标完全独立。共享特征统计、二阶曲率、优化器动量仍会互相影响。

**严禁用错误梯度判断冲突**：例如把蒸馏损失对其独立投影头的梯度与 Flow 对 action expert 的梯度拼接后算 cosine，会引入大量不共享的参数维度；应只选择明确的共享 Adapter 参数。对冻结参数输出零向量时，余弦**未定义**，不应统计为 0。

![梯度夹角的几何意义与以动作目标为主的教师投影](/Embodied-AI-Roadmap/diagrams/pi05-vision-distill-gradient.svg)

*图 2｜研究性一阶优化几何：教师方向进入与动作损失相反的半平面才投影。投影针对特定共享参数，不自动改变冻结的 VLM/专家参数。*

### 4.1 如何实现“动作目标优先”的单向 PCGrad 变体？

[PCGrad 原论文](https://arxiv.org/abs/2001.06782) 的通用方法用于多任务梯度冲突。这里为保证动作优先，采用**研究性单向变体**（不冒称官方 PCGrad 默认全流程）：

$$
\widetilde g_T=
\begin{cases}
g_T-\frac{g_T^\top g_F}{\|g_F\|_2^2+\epsilon}g_F,
&g_T^\top g_F<0,\\
g_T,&g_T^\top g_F\ge0.
\end{cases}
$$

共享参数的更新梯度是

$$
g_{\mathrm{update}}=g_F+\lambda\widetilde g_T.
$$

在 $g_F\ne0$ 且投影有效时，$g_F^\top\widetilde g_T\approx0$，即教师修正不再一阶抵消动作梯度。它**不保证**降低动作验证损失，因为优化器可能有动量、Adam 预条件、步长有限、Hessian 曲率复杂；PCGrad 需要多次反向传播，训练时间和显存也可能增加。

真正要回答的问题不是“PCGrad 听起来更合理”，而是**在具体训练阶段、共享层、相机视角、batch 来源上，负夹角的比例是否足够显著**。如果测得几乎都在 $-0.05$ 到 $0.05$，优先通过放大 batch、拆不同任务、检查损失量纲和噪声置信区间排除统计波动。

### 4.2 谁应该收到什么梯度？

| 参数组 | 纯 π₀.₅ LoRA 基线 | Adapter + 冻结 DINOv3 | 增加梯度投影 |
| --- | --- | --- | --- |
| DINOv3 教师 | 不存在 | 完全冻结、禁止进 optimizer | 同左 |
| DINO 投影头 $P$ | 不存在 | 只接 $\mathcal L_{\mathrm{feat}}$ | 同左（不算共享层余弦） |
| 残差 Adapter $A$ | 不存在 | 接 FM + 教师损失 | 在此组上计算并投影 |
| SigLIP/PaliGemma 主干 | 按显式控制变量设定冻结或 LoRA | 与基线保持同样策略 | 与基线保持一致 |
| Action Expert / LoRA | 接 FM | 接 FM，且必要时通过 Prefix 接收教师间接影响 | 不在此组对教师投影（若无教师梯度） |

额外监督“是否进入动作损失”取决于实际计算图与冻结设置。**要记录每组 nonzero-gradient 参数数、梯度范数、优化器状态，不能只打印 trainable count。** 尤其要注意 LoRA 只作用在所选 Gemma 层时，新增 Adapter 的初始化、AMP dtype、FSDP 参数分片可能需要额外处理。

## 5. 一套可以归因的四组实验，而不是一次堆满技巧

| 组别 | 训练参数及损失 | 要验证的单一问题 | 潜在风险 |
| --- | --- | --- | --- |
| A：基线 | 官方 π₀.₅ + 已锁定的 LoRA；$L_F$ | 原始学习能力 | 需要完整 baseline，不能借用公开 checkpoint 成绩 |
| B：Adapter | A + 残差 Adapter；仍只用 $L_F$ | 增加少量参数本身是否有效 | 参数量差异 |
| C：单教师 | B + DINOv3 冻结特征蒸馏；$L_F+\lambda L_T$ | 教师是否真增加可用信息 | token 语义/几何不一致 |
| D：梯度几何 | C + 共享 Adapter 上的冲突投影 | 对抗性梯度是否是瓶颈 | 多次 backward 与训练不稳定 |

**不建议一开始就两教师、视觉主干替换、强化学习后训练、全量微调一起上**，否则即使成功率提高，也无法判断收益来自教师、参数量、数据顺序、优化器还是动作学习。建议先单教师 DINOv3；Depth Anything 作为后续独立实验，必须单独验证相机位置变化和深度尺度一致性。

公平对照还需要固定：同一个 <code>pi05_base</code> checkpoint、数据集快照、任务/场景 train-test 划分、RGB 裁剪、随机种子、batch size、有效动作维度、horizon、数据样本数、训练 token 步数、正常/扰动评测协议。至少三个独立种子并报告每任务指标和置信区间，不能只给总平均。

### 5.1 数据集与模型分离的验证矩阵

训练：官方 LIBERO 可作为**仿真 API 通路的基准**；工业 EGO/UMI 数据需先通过动作语义、坐标系、同步、episode、norm stats 质检。只有 RGB 没有可执行动作标签的数据**不能**直接计算 π₀.₅ Flow Loss，但可提供当前帧教师蒸馏输入；训练时应显式标记 <code>has_action</code>，不允许无动作样本伪造随机 action 当标签。

评测：分别记录 clean、光照/反光、视角轻微变化、遮挡、未知物体、新场景。主要指标是**同任务同环境协议下的闭环成功率**，辅助指标为接触前 TCP 误差、碰撞、重试次数、动作平滑度、推理时延和峰值显存。视觉特征余弦和训练 Loss 只能作为诊断，不能替代物理执行结果。

任务成功率估计为 $\hat p=k/n$，单一任务评估次数有限时必须同时报告 $n$；若 A/C 各只有 10 次 rollout，“70% vs 80%”未必具有统计意义。真实对比最好采用相同重置状态与对象配置、分任务统计并报告置信区间。

## 6. 复现流程：先跑 CPU 数学契约，再跑官方基线，最后改模型

### 6.1 无 GPU Linux 的第一关：数学与数据规则

仓库里可运行：

~~~bash
python3 examples/pi05_vision_gradient_contract_cpu.py
~~~

脚本只用 Python 标准库，实际测试：余弦为负才投影、近似正交不误判为负、零向量拒绝误称 cosine=0、投影后的方向符合一阶保护、训练/推理两条路径的教师可见性、缺失相机不能进损失、Flow 的反向 Euler 符号。**它不导入 openpi，也不训练任何神经网络。**

### 6.2 GPU 阶段的可核查修改点（教学伪代码，不是官方补丁）

在官方固定 commit 上创建**独立实验分支**，先跑原生 <code>pi05_libero</code> 推理与微调，再做最少改动：

~~~python
# 研究性伪代码：官方 openpi 未提供这些类和接口
obs, actions = next(data_iter)           # 当前图像 + 可执行动作 chunk
E, mask = official_embed_prefix(obs)    # [B, V*N, Dv]，需确认 patch 排序
E2 = E + alpha * adapter(layer_norm(E))
flow_loss = official_flow_loss_with_prefix(E2, obs, actions)

with no_grad():
    target = dino_v3(current_rgb_only)    # teacher eval, no future frames
feature_pred = align_project(E2, current_rgb_geometry)
teacher_loss = normalized_masked_mse(
    feature_pred, target, valid_camera_and_patch_mask
)
# 仅在共享 adapter 参数上测 g_F / g_T；按预先声明的规则做投影
# 若没有可靠共享梯度，不强行执行 PCGrad
total_or_projected_gradients(flow_loss, teacher_loss)
~~~

需维护三套可验证单测：**张量**（Patch 顺序/维度、mask、dtype）、**梯度**（哪些参数参与 FM/教师）、**执行**（与官方 checkpoint、norm stats、动作变换相容）。不要为了偷懒在推理端额外调用 DINOv3，因为那样已经变成双编码器模型，计算代价及研究问题都变了。

### 6.3 逐步升阶与停止条件

第一步原生 π₀.₅ + LIBERO pipeline：数据变换、normalization 和动作客户端结果一致；第二步 B 组 Adapter 在零初始化下的**初始输出应与 A 组相同**（容差由 dtype 决定）；第三步 C 组教师损失能下降，但动作指标不能显著低于基线；第四步仅在统计确认负夹角后加 D 组；第五步做工业机器人数据的闭环迁移。若 B 组已明显恶化，先检查接口与参数冻结，不要用 PCGrad“修复”架构 bug。

### 6.4 关于可用算力和依赖的边界

官方 [openpi README](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) 的粗略单卡需求：推理 >8 GB，LoRA 微调 >22.5 GB，全量微调 >70 GB；这些是**原始实现的指导数字**，Adapter + DINOv3 训练期教师会增大峰值内存。四卡分片能否放下取决于模型分片、教师冻结推理、activation checkpoint、序列长度与训练批大小，必须通过一次 profile 验证，不能按参数量线性估算。教师不需要存梯度，但其前向激活/输入缓存仍占显存。

## 7. 失败模式与安全边界

**错误 1：把视觉蒸馏成绩说成任务成功率。** 特征对齐越好不必然动作越准。对照必须有 robot rollout 和 held-out 场景。

**错误 2：把正交当负冲突。** 余弦 $0\pm\delta$ 只有局部方向信息，先验证梯度采样噪声，再判断是否值得梯度手术。

**错误 3：混淆冻结集合。** LoRA 名称只表示某些权重采用低秩增量；源码 <code>get_freeze_filter</code> 的字符串匹配范围必须枚举验证，特别是视觉塔是否意外参与训练。

**错误 4：伪对齐密集视觉。** 同一个图像裁剪窗口的 patch 对应、教师/学生增广及视角 mask 必须一致；否则优化会迫使策略学习错误的空间映射。

**错误 5：推理依赖训练期信息。** DINOv3 教师目标、未来帧、成功标签、真实动作只能出现在训练和离线评测，不能放进在线条件。

**错误 6：认为双任务总损失下降就是泛化。** 如果 $\lambda$ 太大，模型可能优化教师特征而牺牲动作。本项目以闭环成功率、泛化性、动作接口误差为最终判断。

## 8. 与 StableVLA、DINOv3 和 PCGrad 的技术关系

[StableVLA](./stablevla-ib-adapter-robustness/) 是已有论文与作者源码中的**双视觉主干 + 通道门控 + L1 动作头**，而本章是给**openpi/π₀.₅ Flow Matching**设计的**训练期单教师蒸馏**；架构、前向时延和损失都不相同。DINOv3 的 [Gram anchoring](https://ai.meta.com/research/publications/dinov3/) 是其自身大规模视觉自监督训练中保护密集特征的方法，**不等于本文 PCGrad 梯度手术，也不等于 StableVLA 通道 Gram**。三者虽然都出现“Gram/梯度/特征”字样，优化对象完全不同。

延伸对比：[DINOv3 原始论文（2025-08）](https://arxiv.org/abs/2508.10104)、[Meta 官方发布（2025-08-14）](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)、[Gradient Surgery for Multi-Task Learning（2020）](https://arxiv.org/abs/2001.06782)、[openpi JAX 模型源码](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)、[openpi JAX 冻结过滤](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)、[openpi PyTorch 前缀实现](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py)。

## 9. 自测题（先回答再看解析）

**问 1**：$g_F=(1,0),g_T=(-1,2)$，两者冲突吗？单向投影后教师梯度是什么？

**答**：内积 $-1$，冲突。投影 $\widetilde g_T=(-1,2)-(-1)(1,0)=(0,2)$；再合并 $g_F+\lambda(0,2)$，教师分量不再一阶抵消 $g_F$。这个结论仅在一阶局部、正确共享参数上成立。

**问 2**：如果老师对 $E'$ 的特征目标有梯度，而动作专家被冻结，Adapter 能从 Flow Loss 学习吗？

**答**：能否学习取决于动作专家的前向计算图是否把动作损失的梯度沿 Prefix 回传至 Adapter。即使专家权重冻结，常规自动微分仍能通过其算子向输入回传梯度；但如果前缀在错误位置被 <code>detach</code>、计算放在 <code>no_grad</code> 或训练配置截断，这条路径便消失。必须打印 Adapter 的 $g_F$ 范数验证。

**问 3**：视觉教师增加 1 张深度特征图，是否可以直接将其拼到末端 action 的 32D？

**答**：不可以。视觉教师输出是观测特征，不是机器人控制动作；动作 32D padding 是带物理语义的控制容器。要做跨模态学习，应在视觉表示空间进行对齐，输出仍经过原动作专家与物理反归一化接口。

**练习**：从 [CPU 契约脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/pi05_vision_gradient_contract_cpu.py) 出发，把负梯度比例、余弦置信区间和额外 backward 耗时纳入真实训练日志。先证明**测量正确**，再谈“梯度手术提升成功率”。
