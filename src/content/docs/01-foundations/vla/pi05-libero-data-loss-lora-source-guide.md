---
title: "π₀.₅ × LIBERO 源码实战：数据契约、归一化、Flow Loss、LoRA 与闭环评测"
description: "锁定 openpi 源码逐步追踪 LeRobot 数据→动作 chunk→归一化→状态 token→Flow Matching→LoRA 参数→部署，附合成可执行契约测试"
sidebar:
  order: 7
---

> **源码核验基线｜2026-10-29**：下文固定 Physical Intelligence 官方 **openpi JAX 主仓库 commit [215abfb217db](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)**，不是凭旧版论文或社区二次实现推断。本文是**逐段静态源码核查 + 可运行的无 GPU 合成契约检查方案**，并**未**在此站完成 π₀.₅ GPU 微调、LIBERO 成功率复现或真机实验。训练/推理数据的物理语义比改 optimizer 更优先。

## 导读：为什么“Loss 降了”仍然可能完全抓不起来？

在 LIBERO 或工业 UMI 数据里，一个动作向量是**控制命令**，不是抽象特征。将已经是 delta 的动作再次减去 state、把夹爪开合做成位置增量、把模型训练时的归一化参数与推理端错配，都会造成动作失真；只观察训练 Loss，通常发现不了这些错误。

本篇的交付物是一条可逐行核查、可在无 GPU 机器上首先验证的数据与模型路径：

![π₀.₅ LIBERO 训练数据契约与推理反变换流程](/Embodied-AI-Roadmap/diagrams/pi05-libero-contract-pipeline.svg)

*图 1｜训练和推理两条路径在 policy/transform 与 norm stats 处必须完全对称。蓝色为训练，绿色为推理，橙色为必须做 schema 检查的接口；图为对照源码重绘，不是官方系统图。*

**本章与其他章节的分工**：[π₀/π₀.₅ 联合 Attention、Mask 与 Prefix KV cache](./pi0-joint-attention-code-audit/) 已专门讲网络结构；[π₀.₅ 训练与泛化](./pi05-deep/) 讲论文动机；本文专门解决“把实际采集数据喂进去—算 Loss—恢复物理动作—闭环执行”的工程问题。

## 1. 先把代码版本、实验配置和动作语义锁死

官方 [training/config.py L743–764](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py#L743-L764) 定义了名为 <code>pi05_libero</code> 的训练配置：

| 配置参数 | 该 commit 的实际值 | 不能误解为 |
| --- | --- | --- |
| 模型 | <code>Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)</code> | 所有 π₀.₅ 版本都必须 10 步或必须不离散 state |
| checkpoint | <code>gs://openpi-assets/checkpoints/pi05_base/params</code> | 已经完成 LIBERO 微调的 checkpoint |
| 数据 | <code>physical-intelligence/libero</code>（LeRobot） | 原生任意 RGB-only 视频都可用 |
| prompt | <code>prompt_from_task=True</code> | 让 action 由语言单独决定 |
| <code>extra_delta_transform</code> | **False** | 所有动作都要再减一次 state |
| batch | 256 | 16GB CPU 或小显存单卡能直接跑 |
| optimizer | AdamW，gradient clipping 1.0 | 仅修改 LR 就可适配所有机器人 |
| 学习率 | warmup 10k，峰值 5e-5，长 decay 配置 | 30k 步末尾已衰减到零 |
| EMA | 0.999 | LoRA 与 full-FT 都默认相同 |
| 训练步数 | 30,000 | 完整复现实验仅需执行一个命令 |

这里的 horizon=10 是**机器人动作序列步数**，Flow Matching 采样过程另有 <code>num_steps=10</code> 的默认值，二者恰好都为 10 并不代表它们是同一个超参数。

官方 [examples/libero/README.md](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md) 报告已训练的 π₀.₅ LIBERO checkpoint 对四套任务分别为 98.8、98.2、98.0、92.4，均值 96.85；这是**作者仓库的报告结果，不是本站实测**。使用此 checkpoint 做验证时应明确区别“官方微调模型评估”和“自行从 base 微调”。

## 2. 原始 LeRobot 样本到 Model Observation 究竟经过哪些变换？

### 2.1 LeRobot 的 horizon 是时间偏移索引，不是把整段视频当一个图像

[training/data_loader.py L132–152](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py#L132-L152) 用数据集帧率计算每个未来动作的时间偏移：

$$
\Delta t_k=\frac{k}{f_{\mathrm{dataset}}},\quad k=0,\ldots,H-1 .
$$

例如 <code>fps=10</code>、<code>H=10</code>，动作目标时间戳是当前帧加 0、0.1、…、0.9 秒；**时间跨度是 0.9 秒，而 10 个控制周期名义覆盖 1.0 秒**。这两种表述不矛盾，不能把采样点最大时间差与动作命令占用时长混为一谈。

模型训练样本的数据入口大体是：

| 数据层 | 关键字段/张量 | 变换依据 | 检查点 |
| --- | --- | --- | --- |
| 原始 LeRobot | <code>image</code>、<code>wrist_image</code>、<code>state</code>、<code>actions</code>、<code>task</code> | 数据转换脚本 | fps、episode 边界、RGB 顺序、unit |
| Repack | <code>observation/image</code> 等标准键 | <code>LeRobotLiberoDataConfig</code> | 只改 key，不改变动作含义 |
| LiberoInputs | <code>image.base_0_rgb</code>、<code>left_wrist_0_rgb</code>、<code>state</code> | <code>libero_policy.py</code> | 缺失的右腕图像用全 0，mask=False |
| Normalize | state/action 的分位数归一化 | <code>q01/q99</code> | 使用**本数据集** norm stats |
| ModelTransform | resize 224、tokenize、pad 到 32 维 | <code>ModelTransformFactory</code> | shape、图像 mask、token_len |
| Compute loss | <code>Observation</code> 和 <code>Actions [B,10,32]</code> | <code>Pi0.compute_loss</code> | 动作物理量是否被正确编码 |

官方 [config.py L291–354](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py#L291-L354) 和 [libero_policy.py L23–90](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py#L23-L90) 规定：第 3 路 right wrist 是缺失输入，通常填零并将有效 mask 置 False。**不能因为输入 shape 正确，就把虚构的第三个视角当作有效视觉条件。**

### 2.2 训练与推理变换必须顺序对称

训练数据的处理顺序在 [training/data_loader.py L172–191](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py#L172-L191) 中可核对：

~~~text
dataset
  → repack
  → robot-specific data transforms
  → Normalize(q01/q99)
  → model transforms (resize/tokenize/pad)
  → Pi0.compute_loss
~~~

推理时 [policies/policy_config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy_config.py) 以**相同顺序**处理输入，输出按相反方向处理：model output→Unnormalize→LiberoOutputs 提取前 7 维。它还优先从 checkpoint 附带 assets 加载 norm stats，防止训练/推理归一化不一致。

**重要：** <code>repack</code> 的数据集专用映射不一定会自动应用在实时推理。真实机器人客户端传入字段名称，应与推理侧要求匹配，不能只证明 LeRobot dataset 能加载。

## 3. 动作语义：绝对位置、位姿增量、速度不是一回事

假设训练样本在同一维上当前状态是 $q_t$，目标是 $q^*_{t+k}$：

- **绝对目标**：$a_{t+k}^{abs}=q^*_{t+k}$，有原点/物理单位。
- **相对当前状态的目标**：$a_{t+k}^{rel}=q^*_{t+k}-q_t$，相对同一时间起点。
- **相邻时间步差分**：$a_{t+k}^{step}=q^*_{t+k}-q^*_{t+k-1}$，每一帧相对上一帧。
- **速度命令**：$v_{t+k}$，需要按控制周期积分才成为位移。

不能因为都叫“delta action”就把后两者等同。官方 [transforms.py DeltaActions L229–257](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L229-L257) 实现是对选中的动作维度**统一减当前 state 的对应分量**：

$$
a_{t+k,j}^{rel}=a_{t+k,j}^{abs}-q_{t,j}.
$$

这不是逐帧差分，且只适用于 state/action 在相同单位、相同坐标/物理语义的维度。旋转姿态与 SE(3) 不能机械地逐维相减；四元数也不能直接按四元数分量减法表示旋转。

### 3.1 本次源码审计最容易踩的坑：重复 Delta

对于 LIBERO，官方注释确认数据**已经表示为 delta 动作**，<code>pi05_libero</code> 使用 <code>extra_delta_transform=False</code>。对已是增量的命令再应用上述变换，会得到：

$$
a_{\mathrm{wrong}}=a_{\mathrm{delta}}-q_t .
$$

例如第一个目标增量 +0.02、当前状态 +0.70，错误变换后得到 -0.68，模型会学到完全不同的操作。即使数值归一化后看起来落在正常区间，也不能恢复正确物理意义。

工业 UMI 场景至少要存储：

| 属性 | 示例 | 为什么要保留 |
| --- | --- | --- |
| <code>action_type</code> | target_pose / relative_pose / velocity / joint_target | 决定是否适用 DeltaActions |
| <code>frame_id</code> | world / robot_base / tool | 避免跨坐标系相减 |
| <code>rotation_repr</code> | quat_xyzw / rot6d / axis_angle | 避免旋转数值混算 |
| <code>units</code> | m、rad、m/s | 控制器语义一致 |
| <code>control_dt</code> | 0.05 s | 对速度/加速度至关重要 |
| <code>gripper_semantics</code> | absolute_opening / toggle / force | 不能把开合也无脑 delta |

此表是从工业数据管线角度提出的建议字段，不是 LIBERO 官方元数据的原字段表。

## 4. π₀.₅ 为什么采用分位数归一化？state 到底送没送进模型？

### 4.1 q01/q99，而不是统一做 z-score

官方 [config.py create_base_config](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py#L178-L190) 对 PI0 之外的版本默认启用 <code>use_quantile_norm=True</code>。参考 [transforms.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L134-L184)，对每个维度：

$$
a_{\mathrm{norm}}=2\frac{a-q_{01}}{q_{99}-q_{01}+\epsilon}-1.
$$

推理再通过保存的训练统计量恢复：

$$
a=q_{01}+\frac{a_{\mathrm{norm}}+1}{2}(q_{99}-q_{01}+\epsilon).
$$

例如某平移维度 $q_{01}=-0.2$ 米、$q_{99}=0.2$ 米，0.1 米应映射为约 0.5；若错用单位为毫米的统计量，最终动作误差可以放大三个数量级。分位数缩放并不会自动修正奇异值、错误单位或动作空间定义；分位数之外的数据也不保证被截断在 [-1,1]。

### 4.2 π₀.₅ 默认 state 语言离散化；但 LIBERO 官方配置有例外！

在 [pi0_config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py#L11-L44) 中，<code>pi05=True</code> 的默认 <code>discrete_state_input</code> 是 True。开启时，[tokenizer.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py#L23-L39) 将归一化 state 量化到 256 个 bins，并拼入：

~~~text
Task: <任务文字>, State: <离散状态序列>;
Action:
~~~

**但在固定的 <code>pi05_libero</code> 官方配置里，<code>discrete_state_input=False</code>。** 同时，[pi0.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L139-L177) 对 <code>pi05=True</code> **不会**在动作 Suffix 加连续状态 token。因而在这份具体配置中，state 虽然仍保留在数据契约、batch 形状和可能的输入变换中，却**不作为独立 state token 直接进入 Transformer**。模型可从图像读到部分状态线索，但不等于从观测向量直接感知了 proprioception。

**结论**：面试时说“所有 π₀.₅ 都将 state 放入 Prefix”是错误的；必须同时说出模型默认值和实际训练配置的覆盖设置。这一点对状态不可见、夹爪图像不充分的工业任务尤其重要。

## 5. π₀.₅ 的 Flow Matching 从数据到梯度，逐行对源码

源码：[models/pi0.py L187–218](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L187-L218)。

令 $A\in\mathbb R^{B\times H\times D}$ 是**已经归一化并填充后的**动作。训练随机采样 $\epsilon\sim\mathcal N(0,I)$，$\tau\sim 0.001+0.999\,\mathrm{Beta}(1.5,1)$，构造：

$$
x_\tau=(1-\tau)A+\tau\epsilon,\qquad u_\tau=\epsilon-A.
$$

模型输入包含图像+语言前缀、噪声动作和流时间；动作专家预测 $v_\theta(x_\tau,\tau,o)$。

训练目标：

$$
\mathcal L=\frac{1}{BHD}\sum_{b=1}^{B}\sum_{h=1}^{H}\sum_{d=1}^{D}
\left[v_\theta(x_\tau,\tau,o)_{bhd}-(\epsilon-A)_{bhd}\right]^2.
$$

这对应两层 reduction：<code>pi0.py</code> 的 <code>mean(..., axis=-1)</code> 先对 $D$ 求均值并返回 $[B,H]$；[scripts/train.py L150–158](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train.py#L150-L158) 再对剩余 batch × horizon 求均值。源代码**没有在该损失实现内自动按有效控制维数应用 action-dimension mask**。

### 5.1 padding 的微妙后果

原始 LIBERO 动作输出是 7D，模型默认容器 32D；[PadStatesAndActions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L328-L341) 会把后 25 维填为零。可是训练时对 32 维**全部采噪声**、**全部监督**，填充维预测的目标也是 $\epsilon-0$，并非完全没有梯度。

![有效控制维数与 padding 维的训练监督、梯度稀释示意](/Embodied-AI-Roadmap/diagrams/pi05-libero-loss-mask.svg)

*图 2｜红橙色是真实 7 个执行维，深灰色是额外填充 25 维；官方默认损失在 32 个维度上求均值。只有读取自定义实现明确出现 mask 时，才能声称填充维没有梯度。*

若假设仅真实 7 维有单位大小的预测误差，其余 25 维误差恰好为零，平均 MSE 等于 $7/32\approx0.219$ 而不是 1。但**不能把这个示意数值当作真实训练整体 Loss**：真实填充维也产生误差和梯度。

工程改进候选是显式 masked loss：

$$
\mathcal L_{\mathrm{valid}}=
\frac{\sum_{b,h,d}m_{bhd}(v_\theta-u_\tau)^2}
{\sum_{b,h,d}m_{bhd}+\epsilon}.
$$

这只是**教学对照方案，不是本次被核查源码的默认实现**；改变 loss 归一化会改变梯度尺度，因此需要重新比较学习率和任务成功率。

### 5.2 为什么训练只需要一次前向，而推理需要多轮采样？

训练有真实 $A$，可以直接造出 $(x_\tau,u_\tau)$ 并一次拟合速度场；推理没有 $A$，从纯噪声 $x_1$ 开始：

$$
x_{\tau-\Delta}=x_\tau-\Delta\,v_\theta(x_\tau,\tau,o).
$$

默认 [sample_actions L219–280](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L219-L280) 使用 <code>num_steps=10</code>、步长 -0.1，从 1 积分到 0。首次前向缓存**观测 Prefix 的 KV**，随后每轮只重新计算本轮动作 Suffix。动作 horizon=10 不要求 num_steps=10，也不代表可直接执行全部 10 步而不重新观测。

## 6. LoRA 到底改哪些权重？不要把 PyTorch/JAX 混成一个运行配方

LoRA 对矩阵 $W$ 增加低秩更新：

$$
W'=W+\frac{\alpha}{r}BA,\quad A\in\mathbb R^{r\times d_{in}},\quad B\in\mathbb R^{d_{out}\times r}.
$$

[models/lora.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/lora.py) 的一般默认缩放是 $\alpha/r$，可通过 rslora 选项切换 $\alpha/\sqrt r$。对 JAX 配置，要**同时**设置 <code>paligemma_variant="gemma_2b_lora"</code>、<code>action_expert_variant="gemma_300m_lora"</code>（如果希望两侧都做 LoRA），并使用**与模型配置一致**的 <code>get_freeze_filter()</code>。这与使用“冻结 VLM，action expert 全量训练”的策略不同。

源码中 JAX <code>get_freeze_filter()</code> 会用路径规则冻结基础参数，并将匹配 <code>lora</code> 的新增参数排除冻结；冻结的并不必然等于“整个网络全部不可训练”，应打印 trainable / frozen 参数数量核验。官方 JAX LoRA 示例通常令 <code>ema_decay=None</code>，而 <code>pi05_libero</code> **full-FT** 配置是 <code>ema_decay=0.999</code>。两种策略不能盲目复制设置。

**版本边界**：官方 README 在这个提交的 PyTorch 部分列出 LoRA 训练尚不支持。想复现 JAX LoRA，就用 JAX 代码与对应 checkpoint，不应把 JAX 的配置照搬给 PyTorch 然后认为官方支持。社区 Issue 或 fork 中出现 LoRA 配方，只能作为对照，不代表上游验证结果。

| 对照方法 | 更新参数 | GPU 占用预期 | 适合验证的问题 |
| --- | --- | --- | --- |
| Full FT | 全网络可训练部分 | 相对更高 | 适配新的动作与视觉分布 |
| 双专家 LoRA | 两侧被注入的低秩参数 + 非冻结部分 | 通常较低 | 少量机器人数据能否保留底座泛化 |
| 仅 Action Expert LoRA | 动作专家低秩参数 + 非冻结部分 | 通常较低 | 只调动作是否足以适配新本体 |
| Frozen VLM + new head | 新投影/动作头等 | 不一定低于 LoRA | 能否只学新的物理动作映射 |

LoRA 不保证永远好于 Full FT；工业机器人状态/action schema 改动很大时，必要的输入输出投影仍须正确初始化与训练。

## 7. 跑通前必须分层验收：无 GPU → GPU → LIBERO → 真机

### 7.1 无 GPU 先跑数据契约检查

仓库提供小型可执行[合成契约检查脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/pi05_libero_contract_demo.py)，只用 Python 标准库，不依赖模型权重和 GPU：

~~~bash
python3 examples/pi05_libero_contract_demo.py
~~~

脚本验证：量化归一化往返、误做二次 delta 后的误差、7D→32D padding、Flow 目标方向、有效维 loss 与全维 loss 对比、动作 chunk 时间跨度。**脚本只验证数学和 schema 约定，不等于运行官方训练数据加载器或训练模型。**

### 7.2 Linux + 16GB RAM + 无 GPU 应怎么准备？

在当前本地资源下，优先完成上面的契约测试、官方代码配置核查、少量 LeRobot 样本的文件结构检查与可视化。**不应承诺本地跑 π₀.₅ 全参数训练。** 下一阶段使用可访问的远程 GPU 时，再按官方项目的 [README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)、[LIBERO README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md) 配置实际训练环境。记录 GPU 型号、显存、依赖锁定版本、checkpoint SHA 和每步耗时。

### 7.3 最小 GPU 实验的验收门槛

1. **数据阶段**：抽 1 个 episode，核实 camera mask、fps、动作单位、7D 差分语义、任务指令以及每个 <code>t+k/fps</code> 是否仍在 episode 内。
2. **预处理阶段**：对同一动作做 Normalize→Unnormalize，最大绝对误差达到浮点容忍范围；右腕 mask 与实际缺失情况一致。
3. **前向阶段**：<code>Observation</code>、<code>Actions</code> 形状匹配，一次 loss 不是 NaN；逐项记录 valid-action loss 与 full-32D loss（需要自建诊断，不要误称官方默认已经输出）。
4. **小样本 overfit**：固定 1–5 段动作，在训练预算受控下观察模型能否记住特定序列，且采样后动作物理幅值合理。
5. **微调阶段**：比较 <code>pi05_libero</code> full FT 与受控 LoRA 配置；报 trainable 参数、显存峰值、step time、loss 曲线。
6. **仿真验证**：在 LIBERO 的未见初始状态上闭环执行。对比官方 π₀.₅ checkpoint 不是声称从零训练的结果。分别保存每任务成功率、平均动作耗时与失败录像。
7. **真机阶段**：先低速、非危险治具、独立急停，先验证控制映射再执行任务。收集错误片段供下一轮离线清洗。

### 7.4 Horizon、频率、推理耗时三者怎么一起算？

设数据/控制频率为 $f$ Hz，动作 chunk 长度 $H$，实际每次执行 $K\le H$ 步，单次云端策略推理耗时 $T_{\mathrm{infer}}$，网络通信耗时 $T_{\mathrm{comm}}$。若执行 $K$ 步期间同步请求一次新动作，理想无需停顿的粗略条件：

$$
T_{\mathrm{infer}}+T_{\mathrm{comm}}\le \frac Kf.
$$

例如 $f=20$ Hz、$K=4$，有效动作窗口仅 0.2 秒。端到端还要考虑传感器采样延迟、排队、控制接口缓冲，不能简单把 GPU 单次 forward 延时当作完整控制延迟。

长 chunk 会降低请求频率，却可能增加开环漂移；短 chunk 能频繁重观测，但计算/网络负担更重。**这项权衡与 π₀.₅ 模型的 Flow 积分步数是两个不同维度。**

## 8. 一个与 EGO/UMI 采集平台直接相关的反例

假设你的头戴 EGO 只有 RGB，UMI 有世界系位姿但没有可执行机器人的 <code>joint action</code>。把手腕 $T_{\mathrm{world}\leftarrow hand}$ 直接编码成 LIBERO 的前 6D delta action，会同时违反：

1. **坐标系**：世界系人手姿态与机器人 base/末端增量不是同一物理量。
2. **动作空间**：人手和机器人 IK、关节限制、接触几何不同。
3. **控制语义**：LIBERO 已是相应机器人控制命令；EGO 手轨迹首先只是观察标签。

正确的工程步骤是：标定相机/工具世界系→确定目标机器人 action schema→运动学重定向与可达性验证→仿真回放/执行成功率筛选→使用真实机器人观测与动作进行 VLA 微调。对无法重定向的视频，优先用于阶段标注、语言监督或视觉子目标，而不是伪造连续 action 真值。延伸阅读：[EGO RGB-only→π₀.₅](../../../04-human-to-robot/ego-rgb-only-to-pi05/) 与 [EGO/UMI 时空标定](../../../02-data-engineering/ego-umi-spatiotemporal-calibration/)。

## 9. 本篇需要记住的 6 个可面试追问

1. 为什么 <code>pi05_libero</code> 用 <code>discrete_state_input=False</code>，却仍有 Observation.state？
2. <code>DeltaActions</code> 是动作对当前状态做差，还是相邻两个控制命令做差？
3. π₀.₅ 为什么默认使用分位数归一化？训练端和推理端不一致有什么后果？
4. 当 7D 填到 32D，为什么不能自然推导“padding 不参与 Loss”？
5. Flow 时间变量与机器人时间分别代表什么？为什么 H=10 和 num_steps=10 并无必然关系？
6. LoRA 的冻结规则与实际 trainable 参数是否总一致？什么时候应该改动作输入输出层？

**答题验收标准**：每题要能给出数据张量、公式、源码锚点和一个失败反例，而不只是讲概念。

## 参考与来源边界

- **官方代码（锁定 commit）**：[config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)、[data_loader.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/data_loader.py)、[libero_policy.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py)、[transforms.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py)、[pi0.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)、[pi0_config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)、[tokenizer.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py)、[JAX train.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train.py)。
- **论文/演示**：[π₀.₅: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)、[openpi 官方 LIBERO 示例](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)。
- **分类**：模型配置、函数行为与官方 LIBERO 数值是代码/官方陈述；表格中的工业 schema、损失 mask 改造、容错时延估算和消融实验是本站教学推导。未完成 GPU / 真实机器人训练，不应写成已经复现。
