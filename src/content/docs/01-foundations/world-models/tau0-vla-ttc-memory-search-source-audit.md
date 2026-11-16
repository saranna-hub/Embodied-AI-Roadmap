---
title: "τ₀-VLA 深读：可纠错执行记忆、世界模型预演与分层决策"
description: "核对2026论文及官方源码，贯通多视角Proposal、可纠错Memory、token路由、World Model、Beam Search、40D Flow Matching、LIBERO契约与复现边界"
sidebar:
  order: 12
---

> **2026-11-16 来源/复现边界**：本教程逐项核对 [τ₀-VLA 论文（2026-08-17）](https://arxiv.org/abs/2608.16885)、[官方项目页](https://tau0-vla.github.io/)、[官方仓库](https://github.com/sii-research/tau-0-vla)、[Proposal实现](https://github.com/sii-research/tau-0-vla/blob/main/src/tau0_vla/high_level/proposal/runtime.py)、[World Model官方指南](https://github.com/sii-research/tau-0-vla/blob/main/high_level/world_model/README.md)、[LIBERO官方训练配方](https://github.com/sii-research/tau-0-vla/blob/main/configs/libero/README.md)。文中的论文成绩是**作者报告**，本站的代码仅为**纯 CPU 数学/接口合成验证**。暂未执行官方GPU权重推理、搜索系统联调或真实操作任务。

工业长流程的困难不是单个动作块会不会生成，而是“夹取失败后，下一步该重试、换手，还是继续装配？”这里连接已有 [RTC 异步执行](../../05-control/rtc-inference-training-time-action-conditioning/) 和 [FlowDPG 后训练](../vla/flowdpg-bptt-free-critic-gradient-posttraining/) 两条学习线：**RTC处理动作块时间连续性、FlowDPG处理动作策略改进，τ₀-VLA处理子任务级决策与执行进度记忆；三者并不能相互替代。**

![τ₀-VLA 分层系统、路由与源码开放边界](/Embodied-AI-Roadmap/diagrams/tau0-vla-system.svg)

## 1. 先回答：为什么不能把完整任务直接扔给VLA？

对一个包含“取件→定位→装配→复查”的任务，单体动作策略条件化于完整指令时，模型需要在每次动作推理中自行推断当前阶段。即使当前动作生成完美，选错阶段仍会失败。论文把时间尺度拆为两层：高层使用当前观察、任务目标及记忆输出下一条自然语言子任务；低层在该子任务条件下输出连续动作块。

定义当前观测为多路图像 $o_t$，本体状态为 $s_t$，任务全文为 $\ell$，子任务为 $z_t$，执行记忆为 $\mathcal M_t$，本体和控制模式文字元数据为 $\eta$：

$$
(z_t^{\mathrm{dir}},\mathcal M_t)
=P(\ell,\mathcal M_{t-1},z_{t-1}^{\star},o_t).
$$

$$
a_{t:t+H-1}=\pi_\theta(o_t,s_t,z_t^\star,\eta).
$$

上式的两个 $t$ 分别是“高层决策时刻”和“低层动作块对应的物理时刻”，不能假定两者周期一致。**论文附录给出的部署逻辑是高层后台约1秒刷新一次缓存，低层约30Hz读取缓存**；这不是高层每帧都被调用一次。

### 为什么需要能更正的记忆？

假设上次子任务是“把工件放入夹具”，但实际没有放稳。如果记忆直接写成“已就位”，下一步“压装”会错误。模型需要依据新观察纠正已经过时、超前或错误的进度记录。

输入并非单独一个图像，而是：

| 模块 | 输入 | 输出 | 正确性/风险 |
| --- | --- | --- | --- |
| 高层 Proposal | head、left wrist、right wrist、任务文字、上一子任务、记忆 | think、memory、subtask | 三相机顺序固定，输出文本格式可以正确但语义仍错 |
| Memory | 先前文本进度、当前真实观察 | 更新后的可持续记忆 | 不允许只相信执行命令而不看实际结果 |
| Router | Proposal生成 token 的概率 | 直接执行或额外TTC | 置信度≠可执行概率；需按任务校准 |
| World Model | **仅头部相机**图像 + 候选子任务 | 预测的子任务终态图像 | 无法保证物理可实现，尤其力/接触未显式可见 |
| Value + Search + Reflection | 任务、候选子任务、预测图像及路径 | 最终子任务 | 论文定义完整流程；**当前官方仓库未核实其端到端可运行实现** |
| Low-level VLA | 多视角、状态、子任务、控制元数据 | $H\times40$动作块 | 动作维度有效性与控制语义必须和本体对齐 |

## 2. P：Proposal不是一个神秘的“推理Agent”

[源码 format.py](https://github.com/sii-research/tau-0-vla/blob/main/src/tau0_vla/high_level/proposal/format.py) 中固定相机序列是 head → left → right。输入JSONL的一条样例如下（示意任务，不是论文原始数据）：

~~~json
{"id":"job_001","task_type":"full_qa","instruction":"拾起套筒并放入治具","memory":"上一轮夹取尝试未确认成功","images":{"head":"head.jpg","left":"left.jpg","right":"right.jpg"}}
~~~

对应输出要求包含自定义标记 think、memory、subtask；[runtime.py](https://github.com/sii-research/tau-0-vla/blob/main/src/tau0_vla/high_level/proposal/runtime.py) 解析后会返回 prediction、subtask、think、memory_out、format_valid、next_memory。**format_valid 只衡量语法，不保证动作可行或任务真正完成**。解析失败时，源码会保留旧记忆，而不会把不可信预测写进状态。

官方公开 Proposal 是 Qwen3.5-9B 完整权重；默认单次推理采用 do_sample=False、max_new_tokens=512 与关闭 Qwen 原生 thinking 的模板设置。注意：关闭原生 thinking 与模型自定义输出的 think 字段是两件事。

### 记忆纠错监督如何构造？

设高层标注进度为 $M_t^\star$，训练时从历史示教构建污染版本 $\tilde M_t$（删掉已完成步骤、插入未完成步骤、交换先后关系），再令模型预测正确记录与下一子任务：

$$
\mathcal L_\mathrm{proposal}
=-\sum_{i\in\mathrm{assistant\ tokens}}
\log p_\phi(y_i^\star\mid o_t,\ell,\tilde M_t,z_{t-1}^\star,y_{<i}^\star).
$$

这是对官方 assistant-only SFT 的**教学符号化表达**，并不等于已经复现论文所有记忆扰动采样分布。对于工业任务，错报“压装成功”比漏报一次“治具检查”危险得多：建议构造过期、超前、矛盾、物体掉落、接管后恢复五类负例，并分别统计修复率、误修复率。记忆修复不能把视觉未验证的“已完成”当成真值。

## 3. 为什么让模型多想一会儿？TTC 路由

作者通过首次生成中的token置信度来决定是否花额外计算。对输出第 $i$ 个 token，令 $p_i$ 为已生成token概率，两个最大logit为 $\lambda_i^{(1)},\lambda_i^{(2)}$：

$$
m_i=\lambda_i^{(1)}-\lambda_i^{(2)},\qquad
u_t^\mathrm{all}=\frac{1}{|\mathcal I_\mathrm{all}|}\sum_{i\in\mathcal I_\mathrm{all}}p_i.
$$

对 memory 字段的token子集 $\mathcal I_\mathrm{mem}$ 定义 $u_t^\mathrm{mem}$，原文 Appendix C-J 的门控公式为：

$$
g_t=\mathbf 1[
u_t^\mathrm{all}\le\delta_\mathrm{all}
\lor u_t^\mathrm{mem}\le\delta_\mathrm{mem}].
$$

$g_t=0$ 直接使用子任务；$g_t=1$ 执行搜索。论文文字提及 logit margin，但公布的门控式使用**全体和 memory 字段平均概率**，不是直接拿margin阈值判决。报告或复刻时必须分别记录这些信号，不能把“平均margin阈值”误说成论文公式。

**源码审计提醒**：当前公开的 Proposal CLI 直接调用 do_sample=False 的 generate；返回结构未包含完整 token logits 路由、Value/Reflection的搜索联动。因此“公开了Proposal和World Model”不等于“完整开源TTC推理系统”。独立研究性实现的路由、Beam和Value必须单独标注为等效实现或自研。

## 4. 搜索：为什么不是随便采样 N 次选最大概率？

高层的动作单位是**自然语言子任务**而非逐帧动作；其优势是每条分支可以对齐一个可观察的阶段性成果。

设世界模型 $\mathcal W$ 预测执行子任务 $z$ 后的头部视角图像，评价函数 $V$ 对此结果打分：

$$
\hat o^{b\oplus z}=\mathcal W(\tilde o^b,z),\qquad
v(b\oplus z)=V(\ell,z,\hat o^{b\oplus z}).
$$

Beam保存候选路径 $\rho(b)$、预测图片、**分支局部记忆**、累计价值 $S(b)$：

$$
S(b\oplus z)=S(b)+v(b\oplus z),\qquad
\mathcal Q_d=\operatorname{Top}_B(\mathcal A_d,S).
$$

根节点产生 $N$ 个候选；后续深度每次最多对 $B$ 个保留路径各扩展 $N$ 个候选。深度为 $D$ 时，模型预测调用量上界是：

$$
C(N,B,D)=N+(D-1)BN,\qquad D\ge1.
$$

这只是候选生成和世界模型评估的理论调用规模，不是实际耗时公式；GPU并行、图像生成步数、缓存命中和反思推理决定实际延迟。最终反思模型：

$$
z_t^\star=F(\ell,\mathcal M_t,z_{t-1}^\star,o_t,\mathcal C_t)
$$

**不要求复制最终Beam中的一句话**，可以综合多个候选重新提出更好的子任务。关键安全规则是搜索中的假设记忆 $\mathcal M^b$ 绝不覆盖真实执行记忆 $\mathcal M_t$：预测“已抓起”只应留在模拟分支里，不能写入持久状态。

![TTC搜索树、记忆隔离及 Flow action mask 细节](/Embodied-AI-Roadmap/diagrams/tau0-vla-search-detail.svg)

### 一个完全可检查的合成例子

根节点有“直接放件”与“先检查夹持”。单步价值可能分别是 0.8 和 0.6；若“直接放件”后面只能得 -0.8，而“检查夹持”后面得到 +0.7，则两步累加分别是 0.0 和 1.3。**仅选首步最大分数会选错**。这不是作者的任务分数，只是解释全路径评分与贪心选择差异。

### 图像预演不是物理保障

只有head终态预测时，接触力、手爪打滑、遮挡侧或控制约束可能不可见。世界模型能给出“看起来抓起来了”的照片，但不一定存在可行轨迹。工业实现建议接入物体/工具相对位姿、限位、碰撞距离、接触/夹爪状态作为额外验证；这是**本文提出的扩展方向**，不是官方已经实现。

## 5. 低层：40维 Action Expert 到底学什么？

作者低层使用Qwen3.5 VLM backbone + Mixture-of-Transformers动作专家；两条流通过跨流交互的联合注意力建模视觉/语言/状态/动作。语言输入不只是任务，还包含本体、控制模式、whole-body开关：

~~~text
You are controlling a robot.
Robot type: <embodiment>
Control mode: <eef/eef_wbc/joint>
Whole-body control: <enabled/disabled>
Task: <subtask>
~~~

**40维标准槽位（论文以1为起点，下面用Python切片零起点）**：

| 切片 | 意义 | 说明 |
| --- | --- | --- |
| 0:3 / 3:9 | 左EEF位置/Rot6D | 3 + 6 |
| 9:12 / 12:18 | 右EEF位置/Rot6D | 3 + 6 |
| 18 / 19 | 左/右夹爪 | 原生开度语义需标定 |
| 20:22 | 腰部 | 2 |
| 22:24 | 底盘速度 | 2 |
| 24:32 | 左臂关节 | 8槽，少于8个关节则mask |
| 32:40 | 右臂关节 | 8槽，少于8个关节则mask |

注意：40D模型并不意味着每个训练样本都有40个真实自由度。**不活动的slot必须在状态/动作中置零且从动作loss排除**。对于作者的原生低层训练，动作EEF和关节以当前状态为锚的相对形式编码；和昨日 [HiFi-UMI-2K的20D绝对next-state](../../02-data-engineering/hifi-umi2k-action-video-contract/) 不可直接拼接。HiFi-UMI Rot6D数据卡描述**前两行**，τ₀论文 Appendix C-B 使用旋转矩阵**前两列**；转换时必须先恢复旋转矩阵再按目标约定重新编码，不能直接拷贝六个值。

### Flow Matching的时间方向

论文采用从噪声 $\tau=1$ 积分到干净动作 $\tau=0$ 的反向参数化。动作有效维mask为对角阵 $M$：

$$
a^\tau=\tau M\epsilon+(1-\tau)Ma,\qquad
u=M(\epsilon-a).
$$

以 $H$ 个动作、激活维数 $\operatorname{tr}(M)$ 为例，训练损失：

$$
\mathcal L_\mathrm{FM}=
\mathbb E\left[
\frac{1}{H\operatorname{tr}(M)}
\sum_{j=0}^{H-1}
\left\|Mv_\theta(a_j^\tau,o,s,z,\eta,\tau)-u_j\right\|_2^2
\right].
$$

论文还给出 $\tau=0.001+0.999x,\ x\sim\operatorname{Beta}(1.5,1)$；各样本归一化loss裁剪到100后做batch均值。论文通用低层报告 $H=30$、10步Euler数值积分；**LIBERO专用配方将 horizon改为10**。不能因两个设置均称τ₀而假定推理 chunk完全一致。

简单的一维例子：$a=2,\epsilon=5$，故 $u=3$；从 $\tau=1$ 的 $x=5$ 出发，按 $\Delta\tau=-0.1$ 做十次Euler，应得 $x=2$。如果把符号反过来则得到 $8$。此合成例子在 [CPU契约脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/tau0_ttc_contract_cpu.py) 内被测试。

### LIBERO中为什么40D只激活10个槽？

[官方 LIBERO 说明](https://github.com/sii-research/tau-0-vla/blob/main/configs/libero/README.md)给出原生状态8D、动作7D。状态中的两个相向手指关节压缩为一个夹爪量；EEF轴角转成Rot6D后，模型侧只激活索引 0:9 和 18，共10维。作者在该配方中显式启用 use_action_mask_loss 和 vla_inactive_input_zero；不应把其余30维的随机高斯噪声带进loss。

与 HiFi 2K 20D 相连时要独立明确：左/右手顺序、末端位置参考系、夹爪单位、Rot6D行/列、原始绝对next-state与相对动作、不同视频相机协议、动作执行频率。这里任何一项错位都可能出现“模型loss下降但真正执行抓高/抓早”。

## 6. 开源完整度逐文件核实

下表是 2026-11-16 针对 [sii-research/tau-0-vla](https://github.com/sii-research/tau-0-vla) 主分支的入口检查，而不是对作者未公开私有系统的猜测。

| 组件 | 官方发布物 / 入口 | 当前核实 |
| --- | --- | --- |
| 高层Proposal | high_level/proposal/README.md + runtime.py、infer.py、finetune.py | **有**推理、SFT、服务与权重 |
| 视觉World Model | high_level/world_model/README.md + infer.py、finetune.py | **有**Step1X-Edit基础模型+机器人LoRA推理/微调 |
| TTC Value/Beam/Reflection完整编排 | 论文算法、项目页展示 | **未在检查的仓库文件树中定位到可运行的端到端编排入口**，需自建或继续核查后续发布 |
| 低层模型 | src/tau0_vla/models、trainer、adapters | 有训练、部署骨架与基础权重 |
| LIBERO | configs/libero/README.md、data.py、train.yaml、deploy/libero | 有专用检查点和执行流程 |
| 格式样本 | high_level/examples/ | 有Proposal与World的**不同观测**样例，不能假定输入自动配对 |

重要的文档接口细节：high_level/README.md建议使用 scripts/proposal_to_world_model.py 将两者连接，但仓库递归文件树中**没有发现该脚本**。因此复现实验必须先自己实现严格的JSONL连接与字段校验；不要复制一个不存在的命令后就宣称完整闭环跑通。

另外，论文描述高层TTC依据token概率路由，而当前公开的独立 Proposal推理入口并没有返回门控所需的逐token概率；扩展时须使用生成scores的明确API，并保持与官方权重、标记token及确定性解码兼容。不能从仅有的文本子任务推测置信门控已经复现。

## 7. 可真正着手的复现步骤

**A. 无GPU契约**：在仓库根目录执行下列合成测试，确认路由门槛/Beam累计/虚拟记忆隔离/40D槽位/反向Flow/异常协议均正确。这**不运行任何官方模型**：

~~~bash
python3 examples/tau0_ttc_contract_cpu.py
~~~

**B. 单独运行Proposal**：准备CUDA GPU、Python3.12；用官方README安装 high_level/proposal/requirements.txt、下载全量9B及processor，执行：

~~~bash
python scripts/make_high_level_example.py
python -m tau0_vla.high_level.proposal.infer \
  --model weights/proposal \
  --input outputs/high_level_example/proposal.jsonl \
  --output outputs/proposals.jsonl
~~~

核实 JSONL 每行的 image 路径、memory_out、next_memory、format_valid；同一个真实episode连续测试5–10次，并注入“先前错误记忆”，与无记忆基线比较。

**C. 单独运行World**：另建官方指定的Python3.12环境，取得Step1X-Edit-v1p2模型及官方机器人LoRA；使用 high_level/world_model/README.md 的转换及推理命令。核查图像是否跟随候选子任务改变、是否凭空添删物体，并用**同一测试样本**配对起始图和候选指令。官方示例中 Proposal和World使用的是**不同微波炉示例**，不应把这两个输出直接串成闭环证据。

**D. 低层适配**：按 configs/_template + adapters/_template 填写机器人的camera、state/action、relative与mask协议；先运行 openloop，对齐动作范围和反归一化。LIBERO则用专用libero_eef_40dim和独立模拟环境。60k步作者checkpoint并不等于本文完成了微调。

**E. 自研TTC最小闭环**：加入门控统计、N候选生成、World终态图、Value打分、Top-B保留、Reflection；先在**离线录制的决定性关键帧**测试，冻结低层策略和GPU模型权重；再接入异步缓存与执行记录。要记录每条决策的真实输入 hash、记忆版本、候选、每条预测图、score、深度、调用耗时、最终子任务、真实执行结果。

| 对照组 | 固定条件 | 想回答的问题 |
| --- | --- | --- |
| S0 全任务直接执行 | 同低层模型/数据/种子 | 单体VLA长流程基线 |
| S1 Plan Once | 同一低层，仅引入高层记忆子任务 | 分层与记忆是否改善 |
| S2 Best-of-N | 同候选数量/评分函数，深度1 | 纯采样能改善多少 |
| S3 TTC Beam | 增加未来状态递推和路径评分 | 长程推理是否有效 |
| S4 TTC+记忆纠错 | 加入污染记忆训练、真实结果反馈 | 错误恢复是否改善 |

评测需同时报告**任务成功率、阶段正确率、重复无效尝试率、人工接管率、单次决策p50/p95时延与算力开销**；只有平均成功率而无异常恢复、超时和延迟可能掩盖真实问题。

## 8. 论文结果：哪部分提升真由TTC产生？

| 对比 | 结果 | 证据边界 |
| --- | --- | --- |
| 低层直接执行 → 分层Plan Once | 四长任务平均 27.5% → 45.0% | 同作者环境、同低层；**这组均没有Beam TTC**，每个任务10次 |
| OOD Book Organization 下一子任务 | Plan Once 50.0%、Best-of-N 57.5%、TTC 74.0% | 高层预测准确率，不是物理动作成功率 |
| 加入TTC：Milk Tea | 5/10 → 7/10 | 作者物理实验；小样本 |
| 加入TTC：Book Organization | 6/10 → 9/10 | 作者物理实验；小样本 |
| 加入TTC：Clean Room | 5/10 → 7/10 | 作者物理实验；小样本 |
| 通用数据训练 | 40,115小时 | 论文总训练量，不是公开可下载的数据小时数 |

不要把 **27.5% → 45%** 误写为“搜索提升17.5个百分点”：论文这里主要验证分层Plan Once；TTC另行比较。不要把大模型子任务准确率当成双臂最终任务成功率。十次试验的波动很大，应标注样本数并在自己实验中做置信区间。

## 9. 用于工业流程的研究方案：从失败中修正下一步

目标不是一看到抓取失败就“自动重试三次”，而是由可验证的状态变化产生适当的新子任务。建议定义结构化状态：

~~~text
stage_id, prerequisites, observed_object_pose, gripper_status,
last_subtask, outcome, retry_count, evidence_timestamp,
memory_revision, observation_quality
~~~

在“套筒夹取失败”时，观察到物体仍在桌面、夹爪空载，则应将“已夹取”修订为“未完成”；再由候选生成器提出“调整接近角再抓”“重新定位工件”“人工接管”三条路径。搜索产生的“预期抓取成功”只能在虚拟路径中使用，直到真实观察有证据才写回。**在丢失关键传感器、超过安全重试阈值或图像/位姿对齐异常时，必须进入受控停止或人工审核**。

真正可发表且可验证的方向是：加入物理约束评估与不确定性校准，在等算力预算、等底层策略、等任务分布下比较 Value评分是否能发现“生成图像合理但动作不可达”的假阳性。先做离线候选排序，再做仿真闭环，最后决定是否具备执行条件；不要先把生成图像当作故障恢复证明。

## 10. 源码审计与验收清单

- **论文与公开模块分级**：论文中搜索与价值/反思模块存在，不代表当前仓库发布了完整推理编排。
- **数据合同先行**：相机 head/left/right 顺序、文本记忆版本、40D有效mask、Rot6D行列、相对动作协议都纳入assert。
- **推理变量分离**：路由概率、世界模型生成步骤、Beam N/B/D、动作chunk H是不同变量。
- **防幻觉边界**：预测图仅评估候选，真实记忆只由真实反馈纠正；format_valid不是semantic_valid。
- **实证要求**：同一批关键帧重复测试、分离OOD场景、固定低层权重、给出决策耗时与成功率置信区间。

相关延伸：[π₀.₅ 训练闭环](../vla/pi05-libero-data-loss-lora-source-guide/) · [HiFi-UMI-2K训练数据ABI](../../02-data-engineering/hifi-umi2k-action-video-contract/) · [RTC时序控制](../../05-control/rtc-inference-training-time-action-conditioning/) · [EgoWAM未来视觉监督](./egowam-world-targets-hpt-cotraining/)。
