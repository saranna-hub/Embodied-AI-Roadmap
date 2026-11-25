---
title: "TacInno：触觉创新量、条件 VQ 与 π₀.₅ 的接触学习"
description: "从条件期望与16-bit触觉编码到隔离式注意力、训练数据时序和工业接触任务评测"
sidebar:
  order: 30
---

> 2026-11-25 核查。[论文 v1](https://arxiv.org/html/2611.26405v1)（11月19日提交），[作者项目页](https://tacinno.github.io/)；项目页本次访问失败，未核实 TacInno 官方训练源码、权重及数据下载。已核对 [openpi PyTorch 基础实现](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)。下文区分论文设计、本站教学推导和独立复现建议，不把基础模型源码当作 TacInno 实现。

## 1. 问题：为什么更多传感器不一定产生更好的策略

插入工件、夹持圆柱体、拔出钥匙等任务的失败往往不是目标识别错误，而是视觉无法可靠区分的接触状态，例如轻微滑动和卡滞。直接加入触觉图像并使用行为克隆，模型仍可能仅通过视觉和动作的统计捷径降低损失。

例如夹爪闭合使软胶压缩，即使没有看到触觉图也能粗略预测压缩趋势；真正有价值的可能是压缩趋势之外的切向滑移。将完整未来触觉变化作为监督，会把大量训练预算分配给前者。TacInno 将未来触觉变化压缩成条件离散码，作为辅助监督，不把未来观测送进推理网络。

![训练两阶段与部署数据流](/Embodied-AI-Roadmap/diagrams/tacinno-innovation-system.svg)

图1：本站原创教学图。橙色代表仅离线可用的未来信息，绿色代表部署时必须仍然存在的触觉分支。

## 2. 统计定义与条件信息

用控制索引 $t$ 表示当前时刻，$X_t$ 表示触觉深度和标记位移场，$o_t$ 表示视觉观测，$a$ 表示已执行动作。论文以当前时刻为参考，选取 5、10、15 步后的触觉变化：

$$
Y_k=X_{t+5k}-X_t,\qquad k\in\{1,2,3\}.
$$

注意是相对同一个初始帧的累计变化，不是三个相邻间隔的差分。定义条件集合

$$
C_k=(X_t,o_{t:t+5k},a_{t+1:t+5k}).
$$

其中视觉仅采四个离散时点，动作每个控制步均需要记录。创新量定义为

$$
\nu_k=Y_k-\mathbb E[Y_k\mid C_k].
$$

下面是本站独立推导：平方误差下，任何只依赖 C 的预测器 g 都满足

$$
\mathbb E\|Y-g(C)\|^2=
\mathbb E\|\nu\|^2+\mathbb E\|\mathbb E[Y\mid C]-g(C)\|^2.
$$

因为条件均值残差与任意平方可积的 C 函数正交，交叉项为零。这解释了为什么有条件重建可以优先利用容易预测的部分。然而不相关不意味着统计独立；真实有限样本、有限模型和 L1 重建也不保证精确恢复上述平方误差意义下的创新量。

一个合成例子：观测变化 Y = 2a + z，a 是已知闭合位移，z 是取正负 1 的滑移扰动。若 z 与 a 独立，则条件均值为 2a，创新量就是 z。仅靠变化幅值挑样本会选中大闭合位移而非滑移事件。

未来视觉和执行动作仅允许用于训练期标签生成。条件中包括未来可见现象，是为了不让编码器重复学习随后相机即可看到的信息，不是暗示推理时知道这些未来数据。若不可预测成分是传感器噪声而非有用接触状态，创新量仍可能很大，因此需要质量治理。

## 3. 阶段 A：条件 VQ tokenizer

论文的时间窗口包含 t、t+5、t+10、t+15 四个触觉与视觉观测，以及 15 个动作。冻结 DINOv2 提取图像特征；时空 Transformer 做帧内注意力与跨帧因果注意力。编码器每帧附加四个可学习 query，取最后一帧的四个输出为 z。每个 query 使用独立的 16 项码本：

$$
c_s=\arg\min_{j\in\{0,\dots,15\}}\|z_s-e_{s,j}\|^2,\qquad s=1,\dots,4.
$$

因此每个窗口得到四个类别索引，总容量为 16 bit（最多 65536 种组合，并非一定全部出现）。不要误用一个 65536 类联合分类器；论文的策略监督是四个 16 类分类器。

解码器获得初始触觉、量化码、视觉与动作条件，通过未来帧的 mask token 输出三个累计变化预测。视觉及动作对每个目标只可见到相应时刻；共享的离线码本身总结整个窗口，因此不能把这种条件注意力等同于在线因果预测。

下面给出可复现的 VQ-VAE 教学目标（不是作者未公开实现的逐行复刻）：

$$
L_{\rm VQ}=L_{\rm rec}+
\sum_s\|\operatorname{sg}(z_s)-e_{s,c_s}\|^2+
\beta\sum_s\|z_s-\operatorname{sg}(e_{s,c_s})\|^2.
$$

用直通估计 $z_s+\operatorname{sg}(e_{s,c_s}-z_s)$ 让重建梯度回到 encoder。论文重建使用接触加权 L1；具体权重、码向量维度、优化器及训练时长应等待官方实现或明确作为自定超参，不能凭空填入。

| 张量（B 为 batch） | 形状合同 | 意义 |
| --- | --- | --- |
| tactile_window | [B,4,S,C,H,W] | S 为传感器数，C 为深度/位移通道 |
| action_window | [B,15,D] | 每步动作，不只是四个采样点 |
| latent | [B,4,E] | 四个连续 latent，E 自定 |
| codes | [B,4] int | 四个独立类别，0 到 15 |
| recon | [B,3,S,C,H,W] | 对当前触觉的累计差 |

这些张量是本站设计的 ABI；论文未给出的 S、C、H、W、E 需按实际传感器确定。先训练无条件、视觉条件、动作条件、完整条件四个 tokenizer，再比较 held-out 重建和码本利用率。不应单独用重建误差证明码具有任务价值。

每个 slot 的经验熵和 perplexity 为

$$
H_s=-\sum_j p_{s,j}\log p_{s,j},\quad P_s=\exp(H_s).
$$

若某槽总是同一类别，P 接近 1，存在码本塌缩风险；但高 P 也可能是噪声。应检查码与接触事件关联，并做码打乱反事实对照。

## 4. 阶段 B：接入 π₀.₅ 动作专家

论文保留 PaliGemma 视觉语言主干和 Gemma 动作专家，新增当前 proprioception、60 个触觉 token（使用 t-8、t-4、t 三帧历史）和四个辅助 query，随后是噪声动作 token。部署保留触觉输入，删除 tokenizer 和两个辅助预测头。

![训练期注意力隔离与部署裁剪](/Embodied-AI-Roadmap/diagrams/tacinno-attention-detail.svg)

图2：本站建议的 block mask（行=query，列=key）；组内允许关系是独立实现选项，论文的关键硬约束是辅助 query 不被其余 token 读取。

以 P（视觉语言前缀）、S（状态）、T（触觉）、Q（辅助 query）、A（噪声动作）为分组，最简单的隔离实现让 Q 读取 P/S/T，让 A 读取 P/S/T/A，任何非 Q 的行都禁止访问 Q 列。Q 也不应读取 A，否则训练中带有真值动作信息的噪声动作可能泄漏到辅助任务。

基础 openpi 的 make_att_2d_masks 用累计 block index 和 padding 生成可见关系。如果只把 Q 插在 A 前面并使用原始递增 block mask，A 会看到 Q，违反论文的隔离要求。独立改造需额外显式屏蔽，训练、采样、缓存路径均要一致。

$$
M_{ij}=M^{\rm base}_{ij}m_i m_j\,g(i,j).
$$

这里 $g$ 是上述特殊分组限制。所有层均需使用该限制，不能只改最后一层，否则存在 P/T 中间 token 形成的间接泄漏路径。还需保持训练与部署位置编码一致：删除 Q 后若 A 的位置索引整体减 4，RoPE 与缓存关系会变化。可在部署保留逻辑位置间隔，并用相同噪声与时间进行数值等价验证。论文的确切位置处理未核实。

### 4.1 openpi 源码改造位置（教学路线）

| 基础源码入口 | 已核对的行为 | 建议修改 |
| --- | --- | --- |
| embed_prefix | 图像及语言构成 prefix | 不破坏原 checkpoint 输入 |
| embed_suffix | π₀.₅ 分支使用时间 adaRMS；不像 π₀ 分支显式加入 state_proj token | 新增独立 S/T/Q embedding；不要假定旧 state_proj 存在 |
| make_att_2d_masks | 累计组索引构造可见性 | 加 group-aware mask 与 padding 检查 |
| forward | 输出最后 H 个 token 的逐维 flow MSE | 同时提取 Q 隐状态及 T 接触头，保持动作切片 |
| sample_actions / denoise_step | prefix KV cache + 反向时间积分 | 不需要 Q 或未来码；保持 S/T 条件与训练一致 |

源码：[pi0_pytorch.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)。上述是基础模型接口核对，不是 TacInno 仓库审计。

### 4.2 Flow Matching 与三个训练目标

openpi 使用噪声时间 $s$，区别于物理控制时间 t：

$$
x_s=(1-s)a+s\epsilon,\qquad u_s=\epsilon-a.
$$

$$
L_{\rm flow}=\frac{\sum_{b,h,d}m_{bhd}(v_\theta-u_s)^2}
{\max(1,\sum_{b,h,d}m_{bhd})}.
$$

上式为本站推荐的有效维度归一化扩展；基础源码返回逐元素 MSE，需要在训练外部正确聚合。四个辅助分类器各输出 16 个 logits：

$$
L_{\rm code}=-\frac14\sum_{j=1}^{4}\log p_j(c_j).
$$

总损失与论文一致，由 flow、当前接触属性预测、未来创新码预测三项组成：

$$
L=L_{\rm flow}+\lambda_c L_{\rm contact}+\lambda_z L_{\rm code}.
$$

论文在训练初始设定两项加权辅助损失各为 flow 的 15%。独立实现可在若干固定校准 batch 上估计均值后固定权重，例如

$$
\lambda_z=0.15\frac{\overline L_{\rm flow}}{\max(\overline L_{\rm code},\epsilon)}.
$$

不要每一步按当前损失重新调权，否则改变了优化目标。对没有触觉或标签的样本，分别按有效样本数归一化三个损失；全无标签 batch 的辅助项为零，不能将无效标签当成类别 0。

代码梯度通过 Q 对共享 P/S/T 的注意力改善动作可用表示；但梯度方向是否与 flow 一致仍是实证问题。可以记录两个损失对 tactile encoder 梯度的余弦相似度，不必一开始做梯度手术。冻结/LoRA 方案要确保触觉 encoder 和相关投影确实被优化器覆盖。

## 5. 工业数据工程：时序和动作语义比模型先行

论文定义的关节动作是未来目标相对当前 q_t 的偏移，不能直接套用逐步积分的增量命令。例如 q_t=0.1，未来目标 0.2、0.3，对当前状态的偏移为 0.1、0.2，而逐步增量是 0.1、0.1。机器人平台切换时必须保留 action_kind、reference_step、unit、joint_order。

30 Hz 采集下，5/10/15 个控制步为约 167/333/500 ms；历史 t-8、t-4 对应约 267/133 ms。若触觉只有 20 Hz，不能按相同数组索引取帧。用控制时间轴和已校准的各流时间戳取样，在线历史只允许不晚于目标时点的测量；离线重采样可另设严格误差阈值。动作流记录实际执行的命令和反馈状态，而非模型未执行的候选块。

建议 manifest 最少包含：

| 字段 | 检查 |
| --- | --- |
| episode_id, source_recording_id, t_control_ns | 单调时间、录制来源、分组切分 |
| rgb/tactile timestamps, tactile_sensor_id | 时钟偏差、缺帧、左右传感器 |
| indentation, marker_dx, marker_dy | 单位、零接触校准、饱和、有效像素 |
| executed_action, measured_q, action_kind | 执行确认、动作参考和状态不可混用 |
| tactile_valid, code_valid, action_valid | 三个独立 mask |
| tokenizer_commit, calibration_id, codebook_hash | 标签可复算和版本化 |

每个标签窗口要求 t+15 仍在同一 episode；历史要求 t-8 不跨边界。中途暂停、丢帧、校准变化应触发 invalid 或切段，不通过重复末帧制造未来标签。训练集和验证集应按原始录制、工件批次或操作者隔离。先切分再拟合 tokenizer、归一化统计和损失权重。

EGO/普通 UMI 没有同步触觉，不能直接构造创新码；可以继续用于视觉/动作基础预训练，接触阶段再利用带触觉与执行反馈的少量数据微调。腕部力矩与夹爪电流可以作为另一种条件残差监督的研究对象，但不是论文的视觉触觉信号，也不应将其直接映射成深度图。

## 6. 独立复现路线与评测

第一阶段：运行本站纯 Python 合同测试，先验证条件残差、四码容量、未来标签边界、隔离矩阵和关节动作表示。

    python3 examples/tacinno_contract_cpu.py

第二阶段：在同一批完整录制数据上训练 tokenizer。检查每个 slot 的分布、无条件与完整条件重建差异，以及高残差是否与真实接触事件一致。选择固定 checkpoint 离线生成标签；策略训练不更新该 tokenizer。先用合成信号和少量真实数据验证 shape，再启动 GPU 训练。

第三阶段：在同一 π₀.₅ checkpoint 上构建四组：视觉动作 baseline；视觉+触觉且无码监督；完整条件码监督；无条件码监督。固定示教数、训练预算、执行 horizon、随机种子及成功判定。先做单任务插入/夹持，再做多任务迁移。增设端点两帧 tokenizer、删除辅助 mask 两种消融。注意 tokenizer 若使用额外示教，需要披露并增加等预算对照。

第四阶段：分别进行训练正常/测试正常、训练正常/测试无接触帧、测试时间偏移、测试真实传感器噪声的配对评测。tactile-off 可能制造分布外输入，单凭性能降低不能证明因果机制；需要与训练时完全无触觉 baseline 和同分布扰动联合解读。实际控制仍需独立限力、急停和碰撞防护，不能把辅助损失当作安全保证。

可定义相同种子的配对成功率差

$$
\Delta=\frac1N\sum_{i=1}^N(S_i^{\rm full}-S_i^{\rm off}).
$$

若两个条件下同一 seed 成败不同的数量分别是 n10 和 n01，则差等于 $(n_{10}-n_{01})/N$。对配对 seed 做 bootstrap 置信区间，避免把重复场景当独立工厂证据。每次试验同时记录首次接触时间、插入深度、失败类型、接管率、最大力、执行耗时和传感器延迟。

论文报告六仿真任务平均成功率 87.5%，四真机任务 70.5%，分别超过其最强比较方法 10.7、9.0 个百分点（作者结果，非本站实验）。仿真 tokenizer 使用每任务前90条，策略使用前50条，后10条用于分析；真机每任务采50条，其中45条用于训练。故比较策略示教相同并不意味着 tokenizer 所见数据量相同。论文中插头任务成功率仍仅38%，说明接触监督不能直接解决所有精细装配问题。

## 7. 与现有路线连接及边界

- [InteractVLA](/Embodied-AI-Roadmap/05-control/interactvla-soft-time-geometric-guidance/) 是冻结策略的推理期几何纠正，TacInno 是训练期监督改变表示。两者不能互相替代。
- [π₀.₅ 视觉教师梯度分析](/Embodied-AI-Roadmap/01-foundations/vla/pi05-vision-distill-gradient-geometry/) 可作为辅助触觉梯度分析的基础。
- 条件离散码不是奖励、力控命令、动作 token 或未来接触事件的确定性预言。16 bit 瓶颈不保证互信息最小或跨设备不变。
- 部署仍需 60 个触觉 token 的编码与注意力计算；删除辅助模块不等于零额外推理开销。
- 当前没有核实官方 tokenizer、policy 训练入口与权重；本站脚本仅验证数学及数据合同，不能给出官方 GPU 复现结果。建议以可访问的 openpi 基线加独立模块为起点，完成真实数据验证后再扩大训练。

参考：论文第3节定义和模型，第4节对照与消融；[作者项目页](https://tacinno.github.io/)，[openpi](https://github.com/Physical-Intelligence/openpi)。本站两幅图与测试均为独立教学实现。
