---
title: "EgoActionBench 深读：EGO 双手原子动作标注、时序评价与 SFT→GRPO"
description: "2026-11 EgoActionBench 论文、官方仓库与开放状态核查；从 Left/Right/Both、五阶段标注、Omin/tIoU/F1、Qwen3-32B语义裁判到工业EGO自动标注复现"
sidebar:
  order: 15
---

> **2026-11-17 来源与边界**：原论文 [EgoActionBench, arXiv:2611.03646（2026-11-02 提交）](https://arxiv.org/abs/2611.03646)、[官方项目](https://egoaction-bench.github.io/)、[作者 GitHub 仓库](https://github.com/player0718/EgoActionBench)、[Hugging Face 数据集（需审批）](https://huggingface.co/datasets/play0718/EgoActionBench)。截至核查日，作者仓库已提供 README/示例素材、benchmark 下载说明，**评测工具、权重、自动标注训练集与 SFT/GRPO 训练代码仍为 Coming soon**。下文数学合同脚本是本站独立的**合成数据验证**，不是官方 evaluator，也不代表已下载 1,407 段视频或运行模型。

这篇不是“又一个动作识别论文”。它把真实操作拆成**可验证的动作单元（What）× 精确区间（When）× 执行手（Who）**，并规定如何衡量模型漏检、乱拆、把左右手混写，以及“描述正确却时间不对”。从 [Ego-OSCAR 数据采集](./ego-oscar-deep/) 往下走，最需要补齐的就是**语义数据质量**；这条线与 [MCAP→LeRobot v3 动作和时间契约](./mcap-lerobot-v3-temporal-action-contract/) 互补，但不能替代几何动作标签。

![EgoActionBench 架构与工业数据入口](/Embodied-AI-Roadmap/diagrams/egoactionbench-system.svg)

## 1. 定位：它究竟解决哪一个瓶颈

通用 VLM 能回答“这段录像在切菜”，却不意味着它能回答“左手 2.3–5.1 秒扶住原料；右手 2.8–3.2 秒抬刀；3.2–3.8 秒下压切断；两手是否在共同操控同一物体”。工业任务中的**操作阶段切换、空抓、手-物接触转换、漏抓、人工纠正回流**，往往恰好依赖这些短动作。

定义视频 $V=\{I_t\}_{t=1}^{T}$，输出不是一个 class，也不是一段整片文字，而是可变长集合：

$$
\mathcal{P}(V)=\{p_i=(\hat{s}_i,\hat{e}_i,\hat h_i,w_i)\}_{i=1}^{N_P}.
$$

其中 $[\hat s_i,\hat e_i)$ 采用**左闭右开**时间区间，$\hat h_i\in\{L,R,B\}$ 分别对应 Left、Right、Both，$w_i$ 是该时间段原子动作自然语言描述；模型**不知道真值边界、没有指定动作查询**。如果两个手各自进行独立任务，需要分开两条单手记录；只有**共同操控一个物体**才加一条 Both 协同记录。Both 并不是把 L 和 R 时间段简单求交集。

### 样例：工业放置（示意，不是官方 JSON schema）

~~~json
{
  "source_clip_id": "factory_demo_001",
  "duration_s": 4.0,
  "actions": [
    {"start": 0.40, "end": 2.90, "hand": "L", "caption": "左手稳定夹具边缘"},
    {"start": 0.85, "end": 1.42, "hand": "R", "caption": "右手伸向工件"},
    {"start": 1.42, "end": 1.83, "hand": "R", "caption": "右手夹紧工件"},
    {"start": 1.83, "end": 2.50, "hand": "R", "caption": "右手抬起并移动工件"},
    {"start": 2.30, "end": 2.90, "hand": "B", "caption": "双手协同引导工件对准夹具"}
  ]
}
~~~

这里的 JSON 是**本站推荐的中间交换格式**，不是对 Hugging Face 字段名的断言。保留 source_clip_id、物理时间、hand、版本、审阅状态和证据帧 ID；不要把它变成“机器人末端 6DoF 动作”。

## 2. 数据规模：别把片段数量当成训练小时数

| 属性 | 作者在论文中报告 | 工程解释 |
| --- | ---: | --- |
| 评测集视频 | 1,407 episodes | 来自 Ego4D/EgoLife/EPIC-KITCHENS |
| 评测集有效时长 | 7.982 小时 | 不是几千小时预训练集 |
| 原子动作真值 | 41,132 条 | 人工复核，3 条独立手轨迹 |
| Left / Right / Both | 17,319 / 19,758 / 4,055 | 类别比例明显不均衡 |
| 单 episode 平均长度 | 20.42 秒 | 需要上下文，不能只抽单帧 |
| 动作区间时长中位数 | **0.60 秒** | 2 fps 输入可能天然错过短边界 |
| SFT/GRPO 适配视频 | 5,354 episodes | **额外**自动生成标注的数据 |
| 训练/验证 | 5,146 / 208 | 和人工审校 benchmark 应按源视频 ID 隔离 |

官方 README 明确：benchmark 视频/标注/metadata 已在 HF 发布但需申请访问；自动标注的 5,354 条适配数据未随当前代码仓库公开。**不要把公开的 benchmark 当作训练集使用，否则评估泄漏。** 项目源码也尚未提供 evaluator 和 checkpoint。

## 3. 先建立一个严格的三轨时间线

![Left/Right/Both、短动作以及评估匹配负例](/Embodied-AI-Roadmap/diagrams/egoactionbench-detail.svg)

原子动作的分界条件不是“上一秒和下一秒的文字不同”，而是**操作类型、交互对象、接触关系或主导运动发生有意义的变化**。作者约束 Motion 字段为八类：reach、translate、lift、lower、rotate、press、oscillate、static。语义裁判还检查 Verb、Objects、Tools。工具/物体角色与动作有关：刀在切割中是工具，在“拿起刀”中又是动作对象。

从 EGO 采集原视频建立单调时间轴：保持片源原始 fps、音画偏移、裁剪前后镜头标识，先追踪 Left/Right，再在轨迹上做窗口提案。**不同手允许交叠，一个动作允许横跨多个窗口，真实协作允许与左右手的局部角色动作并存**。因此不能通过统一时间分割，把整段视频强行切成互不重叠的片段。

对一个真实动作持续 $\Delta t$、视频抽帧率 $f$，时间步长约为：

$$
\delta t=\frac{1}{f},\qquad N_{\mathrm{frame}}\approx\Delta t f.
$$

$0.28$ 秒动作在 2 fps 输入下，期望可见样本数仅 $0.56$ 帧；以均匀随机帧相位且没有额外曝光信息为简化假设，命中概率近似 $\min(1,\Delta t f)=0.56$。**这只是采样近似，不是论文给出的检测准确率**；帧不命中不代表绝对没有图像间线索，但能说明短动作标注不能完全依赖稀疏多模态帧。

## 4. 五阶段标注管线：把“人工作业”拆成可追溯工序

作者论文的明确流程如下，工程复刻时须分清**来源事实**与**可替代实现**：

1. **手部预处理**：使用 CaRe-Ego 分离手部 mask，跨帧平滑，并以不同颜色叠到原视频上。mask 是帮助 VLM 判断所属手的提示，不是手指三维姿态或动作真值。
2. **滑窗文本提案**：VLM 逐个约 1 秒窗口对 L、R、B 分别预测描述；参考相邻上下文和上一窗口描述，相同且连续动作可合并。接触/对象/操作改变时不应跨窗口合并。
3. **人工修正**：检查原视频，删去幻觉、补缺漏、分拆复合动作、纠正左右手与边界。每条更改回写片源 ID 和原帧区间。
4. **Agentic QC**：利用接触变化、运动阶段、物体状态变化三类证据审查短边界和语义，提出带证据帧的修订，不应无条件自动覆盖人审真值。
5. **最终人审**：审核意见或退回重新标注，仅通过审核的结果进入基准；未确认的推测动作不应该进入金标准。

这条工序是**语义层的人工闭环**。若应用于工业 UMI/EGO，可在第 1 阶段额外输入已知的手套开合、工具位置、IMU、夹爪接触传感器作为弱证据；**这属于工程扩展，不是作者提供的官方多传感器模型**。传感器证据和“语义模型自信”需要分开保存。

### 数据资产接口：建议增加哪些字段

| 字段 | 来源 | 防止的问题 |
| --- | --- | --- |
| clip_id / source_video_id / original_frame_range | 原始采集 | 同一个母视频跨 train/test 泄漏 |
| start_s / end_s / fps / timebase | 同步后的原生时间轴 | 秒和帧混用、速度改变后的边界失效 |
| hand_track: L/R/B | 模型提案 + 人审 | 左右手关联错、Both 误记为相交 |
| caption / verb / objects / tools / motion | VLM 与人工结构化校核 | “拿着/拧动/释放”混写 |
| evidence_frames / contact_evidence | QC 抽帧 | 审核结论无法追溯 |
| confidence_source / annotation_state / reviewer_id / version | 生产管线 | 把未审查伪标签误认为 GT |
| geometry_link_id / action_abi | UMI / SLAM / LeRobot 独立管线 | 文本语义和几何轨迹错配 |

对“失败后纠正”场景，建议把“动作执行结果证据”“操作者介入”“恢复操作”作为**任务状态额外属性**。EgoActionBench 本身没有提供工业任务成功/失败的闭环状态评测，不能把此项当成论文已有能力。

## 5. 论文最值得学的公式：为什么 Omin 不是 tIoU？

评测真值 $g=(I_g,h_g,c_g,F_g)$；预测 $p=(I_p,\hat h_p,\hat c_p)$。真值结构化语义 $F_g$ 只供评估，不要求模型输出。

首先取两个区间交集长度：

$$
\ell_{\cap}=\max(0,\min(e_g,e_p)-\max(s_g,s_p)).
$$

**候选检索**使用交集与较短区间的比：

$$
O_{\min}(g,p)=\frac{\ell_{\cap}}{\min(e_g-s_g,e_p-s_p)},\qquad \tau_{\mathrm{sem}}=0.5.
$$

候选手类型兼容性不是 3 类任意互通。真值 L/R 可以检索预测 L 或 R，允许“文本对、手错”的候选；真值 B **只可检索 B**：

$$
\mathcal{H}(h_g)=
\begin{cases}
\{L,R\}, & h_g\in\{L,R\},\\
\{B\}, & h_g=B.
\end{cases}
$$

$$
\mathcal{C}(g)=\{p:\hat h_p\in\mathcal H(h_g),\ O_{\min}(g,p)\ge0.5\}.
$$

**边界反例**：真值 $[2,3)$ 秒，预测 $[0,10)$ 秒；$O_{\min}=1$，但

$$
\operatorname{tIoU}(g,p)=\frac{\ell_{\cap}}{|I_g|+|I_p|-\ell_{\cap}}=\frac1{10}=0.1.
$$

这不是实现 Bug，而是**允许含义大致正确但边界很粗的预测参与语义判分，再由 tIoU 单独惩罚时间泛化**。如果误用 tIoU≥0.5 做检索，长预测甚至进不了语义评分，语义能力与时间能力被混在一起。

## 6. 四字段内容分数、语义分与一对一 F1 必须分开

候选语义先经文本裁判判断 Verb / Objects / Tools / Motion。Verb、Motion 二值，Objects、Tools 按条目评分；跳过不适用字段。设有效字段集合 $K_g$：

$$
Con(g,p)=\frac{1}{|K_g|}\sum_{k\in K_g}r_k(g,p),\quad r_k\in[0,1].
$$

只从 $Con>0$ 的候选选择最高者 $p^\star(g)$；同分先匹配手、再最大化 tIoU，最后按照起止时间确定性打破平局。**同一个预测可能被多个真值动作选中**，它可获得多次语义覆盖但不能凭此在 F1 中冒充多次检出。

$$
H_g=\mathbb{1}[h_g=\hat h_{p^\star(g)}],\qquad S_g=\frac{H_g+\sum_{k\in K_g}r_k(g,p^\star(g))}{1+|K_g|}.
$$

未检到正内容候选的 GT，$H_g,S_g,tIoU$ 均为 0，而且**仍保留在真值分母中**。随后对“时间合格且正语义”的二分图做**最大基数一对一匹配**，得到 $TP$；而非直接把每个 GT 独立最佳匹配数当 TP：

$$
FP=N_P-TP,\quad FN=N_G-TP,\quad F1=100\frac{2TP}{N_P+N_G}.
$$

这里 $N_P$ 包含可识别的**无效时间段预测**。报告 0–100 分时，定义：

$$
\operatorname{Semantic}=\frac{100}{N_G}\sum_gS_g,\quad
\operatorname{m\mbox{-}tIoU}=\frac{100}{N_G}\sum_g\operatorname{tIoU}(g,p^\star(g)).
$$

$$
\operatorname{Temporal}=\frac{\operatorname{m\mbox{-}tIoU}+F1}{2},\qquad
\operatorname{EgoActionScore}=\frac{\operatorname{Semantic}+\operatorname{Temporal}}2.
$$

**数值反例**：两个 GT、仅一条跨越二者的预测，即使语义被两个 GT 选中，F1 一对一最大匹配仍只有 $TP=1$，因此 $F1=2/(2+1)=66.67\%$；并不能拿到满分。反过来，把一个真实动作拆成三条近义预测，额外两个 FP 同样压低 F1。官网目前明确评价工具尚未开放，本站 [CPU 合成合同脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egoactionbench_semantic_temporal_contract_cpu.py)只验证上述**公式和边界约束**，不会模拟 Qwen3-32B 自由文本判断的等价性。

## 7. SFT → GRPO：模型究竟训练了什么？

作者有两条多模态底座：**Qwen3-VL-4B-Instruct** 与 **LLaVA-OneVision-2-8B**。输入是视频与任务说明，输出为 L/R/B 三轨的**带时间戳文本序列**，不是 7D/20D/40D 机械动作。

SFT 阶段冻结视觉编码器，训练语言模型与 multimodal connector；目标响应 token 才参与交叉熵（不要把 system/user prompt 算入训练目标）：

$$
\mathcal L_{\mathrm{SFT}}=-\frac1{\sum_jm_j}\sum_jm_j\log p_\theta(y_j^\star\mid V,q,y^\star_{<j}),\quad m_j\in\{0,1\}.
$$

学习率 $10^{-5}$，cosine，effective batch 32。**论文没有发布可直接执行的训练脚本**；下述步骤只能按论文重新实现，不能声称能一键恢复原训练。

GRPO 阶段从 SFT policy 初始化，**视觉编码器与 connector 保持冻结，只更新语言模型**。每条视频采样 $G=4$ 份输出，奖励由语义、tIoU、事件 F1 组成：

$$
R(v,y_i)=0.5S(v,y_i)+0.25I(v,y_i)+0.25F(v,y_i),\quad S,I,F\in[0,1].
$$

作者报告使用 Dr. GRPO 风格损失，组内相对奖励、KL 到固定 SFT reference。以下是**教学概念式**，不等同官方未公开训练实现的全部归一化与截断细节：

$$
A_i=\frac{R_i-\overline R_{\mathrm{video}}}{\sigma_{\mathrm{batch}}+\epsilon},\qquad
\mathcal J\approx\mathbb E[\mathrm{clipped\ ratio}\cdot A_i]-\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{SFT}}).
$$

关键超参：GRPO 学习率 $5\cdot10^{-7}$；每批 6 个视频 × 每视频 4 份回复 = 24 个候选；采样温度 0.8、top-p 0.95、KL 系数 $\beta=0.04$；每组 rollout 只做一次策略更新。**GRPO 优化的是文本标注质量，不是 Flow Matching 动作策略。** 若想把这条线扩展到 VLA，需要把可靠语义标签与正确的连续动作监督对齐，再独立训练策略。

作者报告 GRPO 相对纯 SFT：Qwen3-VL-4B 的 EgoActionScore $40.6\to41.8$、LLaVA-OV2-8B 为 $46.9\to47.9$。提升是**1.2 与 1.0 分**，不要把“相对未适配基线增加 15.9/23.2 分”说成是 GRPO 单独带来的收益。论文还观察到小于 0.5 秒动作的原生模型 m-tIoU 低于 4 分，时间边界是突出瓶颈。

## 8. 按什么顺序复现？先做不依赖 GPU 的合同

**Step A｜开放状态核对**：阅读作者仓库 [README](https://github.com/player0718/EgoActionBench) TODO；按 HF 数据卡申请访问，记录许可证及源视频权利。没有获批就**不要**把“官方数据集已复现”写进报告。

**Step B｜本地公式与输入验收（CPU，可独立执行）**：

~~~bash
git clone https://github.com/saranna-hub/Embodied-AI-Roadmap.git
cd Embodied-AI-Roadmap
python3 examples/egoactionbench_semantic_temporal_contract_cpu.py
~~~

测试涵盖：半开区间、$O_{\min}$ vs tIoU、L/R 与 B 候选兼容、包含长短区间、可重复选择但 F1 一对一、无效区间进入 FP、评分单位以及短动作抽帧负例。只用 Python 标准库，不借助大型语义评估模型。

**Step C｜审核后生成最小 benchmark 子集**：先做 10–20 个**已授权且独立人工复核**的工业片段；按母视频 / 工位 / 日期分组拆分训练与评测；固定标注手册，双人复审高争议边界；保留原视频、手部叠图、候选文本和人工修改差分；计算标注员间区间和字段一致率。

**Step D｜VLM 基线**：用模型原生视频接口执行整段视频生成，限制结构 schema；初测可采用论文公开的 Qwen3-VL-4B **2 fps** 设置，但必须另做 2/5/10 fps 的帧预算消融，并对小于 1 秒动作单独报告。对工业超短接触动作，较高 fps 或局部帧回放更有意义，但计算量会上升。

**Step E｜先 SFT 再酌情 GRPO**：训练仅 target-response token；固定视觉主干、记录 connector / LLM 解冻策略；先看 SFT 是否在独立测试集提高时序、手关联与内容字段，然后才讨论引入固定文本 Judge + KL 的 GRPO。**不存在已核对的官方一键命令，也不能把本站脚本包装为 SFT/GRPO 完整复刻**。

**Step F｜通过后接入管线**：让文本标签组织 [HiFi-UMI 的图像/位姿/夹爪数据](./hifi-umi2k-action-video-contract/) 的语义索引，关联 [EgoHand 相机与手部重建](../03-perception/egohand-dinov3-soma-camera-qc/)，形成“时序语义片段→人审纠正→版本化样本→后续训练筛选”。动作坐标、控制频率、Robot Action ABI 是另一套接口，不能从语义 caption 自动推断。

## 9. 工业 EGO 自动标注的可落地实验设计

| 消融组 | 输入/监督 | 检查指标 | 想回答的问题 |
| --- | --- | --- | --- |
| A0 | 通用 VLM + 原视频 | 手别、漏检、短动作 tIoU | 没有任务适配时错在哪里 |
| A1 | A0 + 手部高亮/分割 | Left/Right/Both、漏检 | 手部提示是否真正有效 |
| A2 | A1 + 滑窗/边界原帧复核 | 小于0.5s / 0.5–1s分桶 | 1 秒窗口能否捕捉亚秒接触 |
| A3 | A2 + 已核对语义 SFT | Verb/Object/Tool/Motion | 领域文本训练是否比大模型更划算 |
| A4（可选） | A3 + GRPO + 固定 Judge | 语义、时间、F1、计算成本 | 小幅增益能否覆盖复杂性 |

另要把人工复核工时、每小时可用动作片段数、伪标签撤销率、跨工位泛化作为生产指标。**不要拿本论文 7.982 小时的人工审校 benchmark 外推百万小时自动标注成本**；论文的性能与数据规模仅供选型，不等于工业验证。

### 三个容易误读的点

- **“评分高”不代表可执行动作正确**：本基准评价文字、区间、手关联，不验证 3D 位姿、力控和机器人任务成功率。
- **Judge 不是绝对真值**：作者固定 Qwen3-32B 作为文本评估器；附录报告特定人工校准样本上的一致性，但跨工业术语、噪声与新类别仍需复核。
- **评测闭环 ≠ 数据生产闭环**：5 阶段数据制作是作者过程描述；目前尚没有全套公开工具链可无缝部署，也没有官方自动标注训练代码可直接微调。

## 10. 工程检查清单与延伸阅读

正式上线前逐一检查：①源视频授权/母视频级隔离；②视频原始时间轴与 fps；③单手与双手互不互相替代的合约；④所有漏检与无效预测计入分母；⑤评估使用语义字段 Judge、不能用单纯词相似度冒充；⑥高风险短动作有原帧审核证据；⑦参数/权重/代码可用状态实时更新；⑧人工修改记录可回流训练集且能撤回。

下一篇阅读建议：[UMI/LeRobot 多模态时序质量](./mcap-lerobot-v3-temporal-action-contract/)、[Ego-OSCAR](./ego-oscar-deep/)、[EgoHand](../03-perception/egohand-dinov3-soma-camera-qc/)、[工业场景状态理解与采集](./ego-umi-spatiotemporal-calibration/)。EgoActionBench 负责“视频中发生了什么”，几何与策略数据线负责“动作如何落地执行”，两者只有通过统一源 ID、时间戳与数据版本才能产生可追踪的训练收益。

**资料版本**：2026-11-17，论文 v1 与作者仓库公开状态核查；本教程示例、扩展方案和 CPU 测试均由本站独立编写。