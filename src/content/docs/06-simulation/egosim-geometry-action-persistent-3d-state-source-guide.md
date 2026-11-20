---
title: "EgoSim 深度教程：EGO 视频如何变成可持续更新的 3D 交互世界"
description: "ECCV 2026 EgoSim 论文与 2026-11-20 官方源码核查：52 通道 DiT、投影手骨架、3D 场景记忆、Sim3/TSDF、数据接口、GPU 推理、CPU 合同与工业失败恢复的实验边界"
sidebar:
  order: 3
---

> **核查日期：2026-11-20。** 依据 [EgoSim 论文 arXiv:2604.01001v2（2026-07-01）](https://arxiv.org/html/2604.01001v2)、[项目页](https://egosimulator.github.io/) 和 [作者官方 GitHub](https://github.com/jinkun-hao/EgoSim)（核查时主干提交 **8a82e88a**）。文中【论文】表示作者论文结果，【源码】表示公开实现，【教学推导】表示为理解机制构造的简化公式，【建议实验】尚未执行。**本站只做过无 GPU 的合成数学与接口合同测试；未运行 EgoSim-14B 的模型推理、训练或工业真实任务。** 数据及预训练权重授权需要按各上游项目分别确认。

# EgoSim：一段 EGO 视频不只是画面，还可以作为可更新世界状态的训练素材

为什么很多视频世界模型能把抓杯子“画得很像”，但下一个动作又让杯子回到原位？它们没有一个能够跨片段保存和修改的 **3D 场景状态**。EgoSim 的关键思路是把“看见什么”与“世界目前是什么样”分开：相机轨迹决定场景投影，手/夹爪关键点决定交互条件；视频生成后重新建图并把交互物体的新几何写回世界状态。

![EgoSim 从视频、三维场景与动作条件到长期状态更新的系统架构](/Embodied-AI-Roadmap/diagrams/egosim-system.svg)

*图 1｜本站原创重绘：上方是数据生产，下方是单段生成与跨片段状态更新。注意世界状态是估计与生成的结果，不是物理引擎严格的真值。*

## 0. 先判断它适不适合你正在做的工业 EGO/UMI

- **适合研究的问题**：没有完整机器人动作标签的大量 EGO 视频，怎样转为“未来交互结果”的监督；世界模型怎样维护阶段状态，避免多步操作时物体复位。
- **不是直接的策略模型**：EgoSim 输入给定相机/手部动作并**生成视频**，它不会自动输出可执行的 π₀.₅ 动作 chunk，也不会证明生成的视频中接触、力和碰撞符合物理规律。
- **与 EgoWAM 不同**：EgoWAM 训练时用未来 DINO/3D Flow 辅助 VLA，推理丢弃世界头；EgoSim 推理时仍运行大型视频 DiT + 重建流程，定位为世界模拟/可视化/候选动作筛选。
- **重要代码状态**：官方公开 **14B 推理权重、单段推理、EGO 数据标注流水线、连续场景更新代码**；截至本次核查官方 README 仍写 **Training — coming soon**。可以推理，但不能声称已经有完整可训练的官方全量预训练 recipe。

## 1. 世界状态、动作、观测：明确谁是条件、谁是预测

【论文】在片段索引 $k$，用 $S_{k-1}$ 表示上一片段结束后的 3D 世界，输入动作分成相机轨迹 $C_k$ 与交互关键点运动 $H_k$：

$$
A_k=(C_k,H_k).
$$

$$
O_k=\Pi(S_{k-1};C_k)+\Delta O(H_k).
$$

$$
S_k=\mathcal U(S_{k-1},O_k).
$$

- $\Pi$ 是把场景点云沿相机轨迹投影成静态场景先验视频，**不是**让网络从头猜静态背景。
- $\Delta O$ 是关于动作引发的动态变化的**概念性残差表达**；实际 DiT 是在 VAE latent 上合成完整视频，而不是像素级简单加法。
- $\mathcal U$ 从生成的观察估计最新几何、区分静态与交互物体，再融合进世界状态；它并非可微的“真实环境状态更新器”。

核心难点是**先生成、再相信多少**。若模型把一个未抓起的工件幻觉成已抓起，直接把错误写进 $S_k$，后续规划会持续偏离。因此工程应用中要把每次 state update 视为**候选更新**，有独立的质量门和可回滚版本。

### 1.1 相机坐标不能和机器人基座坐标混用

【教学推导】取 $T_{WC}(t)$ 为相机到世界的齐次变换、$K$ 为相机内参，则世界点 $P_W$ 的相机坐标与像素坐标为：

$$
P_C=T_{WC}(t)^{-1}P_W,\quad \tilde p\sim KP_C.
$$

其中仅使用 $P_C$ 的前三个分量，且必须满足深度为正。固定世界点、相机移动时，其像素位置变化形成**视差**，这是静态点云重投影应当解释的部分，不能作为被操作物体移动的证据。

如果你的数据已有 UMI 世界坐标手腕轨迹：

$$
P_C^{hand}=T_{WC}(t)^{-1}T_{WH}(t)P_H.
$$

手部原点、相机外参、动作时间戳必须来自**同一个观测时刻**。若把 $T_{CW}$ 反作 $T_{WC}$，或者把毫米轨迹直接与米制点云混合，渲染“看起来能动”也不能说明几何正确。

## 2. 可复核的数据输入输出合同：论文图和代码不是同一层

【源码】官方仓库根目录 README 的**单段推理**需要 4 个实物文件 + 文本提示，以及 CSV 元数据。模型不是直接输入未处理的原视频。

| 文件 / 条件 | 产生方式 | 给模型的含义 | 常见错误 |
| --- | --- | --- | --- |
| **hand_inpaint.png** | 首帧手臂 Mask + 图像补全 | 无遮挡的初始静态场景 | 补全把杯子、夹具抹掉 |
| **rendered_scene.mp4** | 首帧深度反投影点云，按相机轨迹重渲染 | 具有视差的静态场景先验 | 相机位姿取反、尺度漂移 |
| **pc_mask_video.mp4** | 点云覆盖区域的可见性 Mask | 标明需要生成的空洞 | 黑白语义反转 |
| **skeleton_3d.mp4** | MANO 21 点或 URDF 末端几何投影 | 手/夹爪运动控制条件 | 2D 图像对齐而 3D 尺度错 |
| **prompt / caption.txt** | 文字描述、CSV 字段 | 语义上下文 | prompt 把未来结果泄漏进输入 |

**官方默认**单段 $61$ 帧、$16$ FPS、$832\times480$；不是 61 个**新**动作时刻拼接任意次数。若相邻片段共享端点，两个 61 帧片段构成：

$$
N_{\rm unique}=61+(61-1)=121.
$$

不要把连续片段重叠的同一帧当作两次独立采样。用时钟 $t_i=t_0+i/16$ 定义真实时间，轨迹切分、标注与预测都按同一时间轴处理。

### 2.1 EGO 数据标注怎么生产这些文件

【源码】官方 **data_process/README.md** 已给出单片段流程：

1. **规范视频**：目标 16 FPS、61 帧；首帧用 SAM3 找手臂、Qwen-Image-Edit 去手得到干净背景。
2. **重建场景先验**：Depth Anything 3 估计相机内外参和深度，平滑相机轨迹，反投影首帧点云，渲染 3D 先验视频与 Mask。
3. **提取动作条件**：HaMeR 产生 MANO，执行去重与异常处理，重建 21 个三维手部关节，再做图像投影；不是把 MANO 直接当机器人动作标签。
4. **文字描述与元数据**：Qwen2.5-VL caption、metadata CSV；下游按文件路径逐个加载。

若来源是 UMI，不必重复做手腕三维估计，但仍需**T_WH 到相机 T_WC 的变换、末端定义、夹爪开度编码、可见性标记**。可考虑从 UMI 的双手轨迹绘制 skeleton 条件，绝不能认定 UMI 动作等于 21 点 MANO。

【数据合同建议】至少记录：source_video_id、episode_id、camera_frame、world_frame、timestamp_ns、K、T_WC、hand_keypoints_W、valid_mask、pose_source、calibration_version、reconstruction_version、asset_hash。对训练/评测按 **原始视频 ID 和采集 session** 做隔离，不能用片段级随机切分造成数据泄漏。

## 3. 真正的 Transformer/DiT 输入：为什么恰好 52 通道

【源码】核查 [egowm/inference/pipeline.py](https://github.com/jinkun-hao/EgoSim/blob/main/egowm/inference/pipeline.py) 和 [egowm/inference/runner.py](https://github.com/jinkun-hao/EgoSim/blob/main/egowm/inference/runner.py)：

- DiT 以 Wan2.1-Fun-14B-InP 为初始模型，公开加载配置 **in_dim=52**、**out_dim=16**、**dim=5120**、**40 层**、**40 heads**，patch size 为 $(1,2,2)$。
- 输入次序严格是 **[16 noisy latent，4 mask patch，16 masked scene latent，16 hand skeleton latent]**，和“直接把 3D 点云塞入 Transformer token”不同。
- 源码在每步采样时显式执行 **scene × (1 - mask)**，掩掉待生成区域；读取 Mask 时把**首帧设为 0**，保留起始约束。Mask 由像素采样为每个空间 $2\times2$ patch 对应的 4 个通道。
- 首帧、文本 embedding 还经视觉/文本条件分支提供语义与外观上下文。若复现只拼 52 通道、没加载这些上下文，不能宣称同架构。

![EgoSim 52 通道 DiT、遮挡掩码与连续世界状态的细节图](/Embodied-AI-Roadmap/diagrams/egosim-detail.svg)

*图 2｜本站原创重绘：52ch 输入布局、首帧/可见性与两个 61 帧片段共享一帧；右下角展示错误的无条件状态写回如何污染下一段。*

$$
z_{\rm in}^{(t)}
=\operatorname{Concat}
[z_t^{16},M^{4},z_{\rm scene}^{16}\odot(1-M),z_{\rm hand}^{16}]
\in\mathbb R^{B\times52\times F'\times H'\times W'}.
$$

这里的 $1-M$ 在源码里从 mask 的第一个通道扩展到 16 个 scene 通道，**不是直接将四个通道与 16 个通道相减**。VAE 的时空压缩后维数依具体实现确定，不应把 61 原始视频帧直接假定为 61 latent 帧。

### 3.1 噪声预测、Flow Matching 与论文/源码差异

【论文】方法第 3.2 节将训练公式写成对噪声 $\epsilon$ 的均方误差：

$$
\mathcal L_{\rm gen}
=\mathbb E_{z_0,t,\epsilon}\left[\left\|\epsilon-\epsilon_\theta(z_{\rm in}^{(t)},t)\right\|_2^2\right].
$$

但论文实验第 5.1 节写**推理使用 50 步 Flow Matching**；【源码】推理经 WanVideoPipeline 的 scheduler 进行 50 步更新。这意味着**不能仅凭“Flow Matching”这几个字，将上面的 $\epsilon$ 目标直接套成 π₀.₅ 的 $v^*=a-\epsilon$ 动作流匹配**。两者预测变量、时间参数化、插值路径和 scheduler 配合都必须逐项审计，尤其官方**训练入口尚未公开**。

【教学推导】若某实现确实采用直线位移插值，才可以写：

$$
x_\tau=(1-\tau)\epsilon+\tau x_1,\qquad v^*=x_1-\epsilon.
$$

这只是用于检查模型输出与采样器的一致性，**不声称它等于 EgoSim 已公开训练源码**。重要实验结论是：这个 DiT 是**生成世界观测**的网络，不能拿 16 通道视频 latent 输出去替换 π₀.₅ 的 7D 或 14D 动作头。

## 4. 长流程的技术关键：交互状态更新、Sim3 对齐与 TSDF

【论文+源码】持续生成不是单纯把前一段视频附加到 prompt。官方将 **continuous_simulation/** 单独组织，场景重建依赖额外的 SAM3、Depth Anything 3、修改的 egosim_state 后端和相应模型权重，提供 **recon_visualize** 与 **full** 模式。

### 4.1 哪些几何被更新、哪些应该保留

- 语义：VLM/检测器找交互物体，SAM3 跟踪，排除人手等活动主体；交互物体最后可见帧是一个候选状态。
- 几何：深度+相机轨迹得到每帧点云；将动态物体从背景融合剥离，避免同一个杯子同时出现在旧位置与新位置。
- 融合：利用重叠相机位姿估计坐标对应，使用 Sim3 Umeyama 对齐两段重建结果，再用 TSDF 更新静态环境。
- 可信度：深度遮挡、物体遮挡、交互检测错误会污染 3D 状态。**论文内方法并没有保证状态一定物理正确。**

对于两段相机/场景点集 $\{p_i\}$ 和 $\{q_i\}$，Sim3 允许一个统一尺度 $s>0$、旋转 $R\in SO(3)$ 和平移 $t$：

$$
q_i\approx sRp_i+t.
$$

其加权最小二乘形式：

$$
(s^*,R^*,t^*)=\arg\min_{s>0,R^\top R=I,\det R=1,t}
\sum_iw_i\|q_i-(sRp_i+t)\|_2^2.
$$

如果只有单个匹配点或相机几乎不运动，尺度/旋转不可辨识；重叠帧多但全为同一条直线也可能退化。**先拒绝退化，再做融合**，不要因为能算出 SVD 就认为状态可信。

对于一个 TSDF 体素的多帧更新，教学式加权融合可以写：

$$
D_{\rm new}=\frac{w_{\rm old}D_{\rm old}+w_{\rm obs}D_{\rm obs}}{w_{\rm old}+w_{\rm obs}},\qquad
w_{\rm new}=w_{\rm old}+w_{\rm obs}.
$$

这是一种几何一致性处理，不是碰撞接触仿真。论文附录给出 **0.003 m** 体素与深度差阈值等设置；真实工业重建时不得不经分辨率/噪声标定就复制这些值。

### 4.2 把“可回滚状态版本”作为工程扩展，而非作者原创新点

【建议实验】给每次状态更新设置两个版本：

$$
S_k^{\rm candidate}=\mathcal U(S_{k-1}^{\rm accepted},O_k),\quad
S_k^{\rm accepted}=
\begin{cases}
S_k^{\rm candidate}, & g_k=1,\\
S_{k-1}^{\rm accepted},&g_k=0.
\end{cases}
$$

$g_k$ 由可见点比例、遮挡、重投影误差、物体追踪连续性、相机尺度和接触条件组合确定。这样做的目标不是“模型万无一失”，而是**阻断明显错误被写成下一步的世界真值**。现场人工纠正日志可以反过来标记哪些预测状态与实际执行不符，构成后训练数据治理线索。

## 5. 如何不靠臆测地复现公开代码

### 5.1 固定代码版本、准备 GPU 环境

【源码】依照作者 README，首先 pin 官方仓库；下面命令**供读者执行，本站本次没有 GPU 跑过**。

~~~bash
git clone https://github.com/jinkun-hao/EgoSim.git
cd EgoSim
git checkout 8a82e88abb8b79e333a6dc4edc192c8f7f348990

conda create -n egosim python=3.10 -y
conda activate egosim
# 按 CUDA/PyTorch 兼容矩阵先安装 torch/torchvision
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
huggingface-cli download wuzhi-hao/EgoSim --local-dir ./EgoSim-14B
~~~

作者 README 标注 Python 3.10+ / CUDA 12.1+，数据标注模块有 **sam3 / da3 / hamer / caption 四套环境**和较重的额外依赖，不能假定一个环境全部装好。建议分开验证样本推理、几何重建和文本 caption。

### 5.2 先用作者 demo 验证单段推理

~~~bash
# 下载官方 README 链接里的 demo 数据，并确认实际解压目录
PYTHONPATH=. python egowm/inference/runner.py \
  --dataset egodex \
  --model_root ./EgoSim-14B \
  --dataset_root tests/samples/demo_data/egodex \
  --metadata_path tests/samples/demo_data/egodex_metadata.csv \
  --output_dir output_egodex --num_inference_steps 50 --gpu_id 0
~~~

**核查提示**：作者 README 的 demo 下载文字与示例解压文件名有一处不一致；以**实际下载文件名、metadata.csv 的路径和解压后的目录**为准，不要盲目照抄所有路径。成功的最小证据是按样本生成视频 **{id}.mp4** 与对比视频 **{id}_cmp.mp4**，同时检查 61 帧、16 FPS、空洞区域与手轨迹是否对齐；生成文件存在并不等于模型质量复现成功。

### 5.3 再做连续状态更新

~~~bash
cd continuous_simulation
cp configs/project.yaml.example configs/project.yaml
# 按官方说明安装第三方 SAM3、Depth Anything 3
bash scripts/setup_envs.sh
# 项目根目录准备 EgoSim-14B，下载场景权重
bash scripts/run_recon_visualize.sh
# 通过后再运行官方文档的 full 模式
~~~

**顺序不能颠倒**：先独立跑通数据路径和世界重建，再做大型 DiT 生成，最后验证跨 clip 的状态保留。不同机器上的 CUDA 编译、权重许可、显存、SAM3 和 DA3 版本都可能是阻塞。论文报告在 **8×H200** 上对 14B DiT 全参数微调 4000 步；这不是“四张 48GB 显卡就一定能直接按同配置复刻”的证据。

## 6. 不需要 GPU 的可执行合同：验证哪些隐蔽接口错误

本站新增 [EgoSim CPU 合成合同脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egosim_world_state_contract_cpu.py)，仅 Python 标准库：

~~~bash
python3 examples/egosim_world_state_contract_cpu.py
~~~

覆盖：**52ch 布局**、**2×2 Mask 四通道与首帧**、**世界坐标重投影/相机移动负例**、**Sim3 尺度与退化拒绝（简化无旋转 oracle）**、**失败状态拒绝写回**、**61 帧拼接重合端点去重**。这些是**接口单元测试**，不是 DiT 推理精度、3D 重建精度或策略闭环成功率的证明。

建议将测试进一步分成三级：

| 验证级别 | 输入 | 输出证据 | 暂不能证明 |
| --- | --- | --- | --- |
| L0 合成合同 | 人造坐标/可见性/状态 | 每个 PASS 与失败负控 | 真实模型表现 |
| L1 标注小样本 | 10 段真实 EGO、UMI、标定 | 点云重投影、滑动窗口 QC | 可执行动作策略 |
| L2 GPU/连续生成 | 官方权重 + 场景后端 | 多段画面、世界状态版本与重投影误差 | 碰撞接触正确、VLA 改进 |
| L3 下游任务 A/B | 相同机器人动作数据、受控真实/仿真任务 | 成功率、恢复率、人工接管、成本 | 超出指定任务的泛化 |

## 7. 论文结果要按评价对象解释

【论文】作者在 EgoDex/EgoVid 上使用 PSNR、SSIM、LPIPS 和相机/深度误差。列出部分作者数据，**不是本站实验**：

| 设置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 解读 |
| --- | ---: | ---: | ---: | --- |
| EgoDex 单段 EgoSim | 25.056 | 0.896 | 0.170 | 有首帧与已知场景几何先验 |
| EgoDex 连续 EgoSim | 19.165 | 0.835 | 0.220 | 中间状态更新后出现累积误差 |
| AgiBot 无人类预训 | 16.36 | 0.69 | 0.31 | 作者机器人生成基线 |
| AgiBot 有人类预训 | 18.67 | 0.72 | 0.28 | 对比同样总训练步数 |

**注意：** 14B 生成视频 PSNR/LPIPS 提升不等于机器人操作成功率提升，更不是证明 EGO 数据能直接产生可执行策略。连续生成优于原有方法也不意味着它的指标不会随着生成次数增加而下降：论文数据恰好显示连续片段的视觉指标恶化。

【论文】在 50 段 EgoCap 真实视频上，30 段用于适配、20 段测试，微调 50 步；在 AgiBot 则对比**总步数相同**的人类预训练 + 机器人微调与仅机器人训练。两个实验的目标、环境和预算不同，不应汇总成“少量数据即可工业部署”。

## 8. 面向工业长流程/失败恢复的研究方案

假设工序为取件—移动—装配，对每个动作候选 $a^{(j)}$ 生成一个**可分支的**世界状态版本而非覆盖主状态：

$$
\hat O^{(j)}=G_\theta(S_t,a^{(j)}),\qquad
\hat S^{(j)}=\mathcal U(S_t,\hat O^{(j)}).
$$

$$
a^*=\arg\max_j\left[
\alpha R_{\rm progress}(\hat S^{(j)})
-\beta C_{\rm collision}(\hat S^{(j)})
-\gamma U_{\rm epistemic}(\hat S^{(j)})
\right].
$$

**这是本站提出的候选方案，不是 EgoSim 已公开的策略优化代码**。工业使用必须注意：视频世界模型对力、插接公差、滑移可能完全不可靠，所以碰撞/接触代价需要独立几何或物理求解器，且预测结果只能作为策略**候选排序与观测异常检测**的辅助信号。

建议做四组对照：A 直接 π₀.₅ LoRA；B 加 EGO 世界目标预训练；C 推理期候选视频筛选；D C+保守状态门控。统一动作数据规模、算力预算、随机种子与 rollout，报告阶段成功率、总体成功率、失败恢复率、推理延迟和状态漂移，**分别归因世界先验与更多数据本身的收益**。不要把生成视频作为真实环境的监督 GT。

## 9. 公开资源与可复现性清单

| 来源 | 核查结论 | 研究者下一步 |
| --- | --- | --- |
| [论文 v2](https://arxiv.org/html/2604.01001v2) | 明确状态循环、DiT、数据量、实验设置 | 用论文式子解释机制，不擅自补训练源码 |
| [官方 README](https://github.com/jinkun-hao/EgoSim/blob/main/README.md) | Inference/Continuous/Data 标记完成；Training 未完成 | 固定 commit，核对权重 |
| [52ch 模型配置](https://github.com/jinkun-hao/EgoSim/blob/main/egowm/inference/pipeline.py) | 16+4+16+16、DiT 40 layers | 核对输出头和 scheduler |
| [单段推理](https://github.com/jinkun-hao/EgoSim/blob/main/egowm/inference/runner.py) | Mask 首帧置 0、模型拼接与采样 | 对照每帧/Mask 形状 |
| [数据流水线](https://github.com/jinkun-hao/EgoSim/blob/main/data_process/README.md) | SAM3/DA3/HaMeR/Caption 分环境 | 确认上游依赖可安装 |
| [连续生成](https://github.com/jinkun-hao/EgoSim/blob/main/continuous_simulation/README.md) | 场景后端另有环境和权重 | 先跑 recon_visualize 再 full |
| [CPU 测试](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egosim_world_state_contract_cpu.py) | 本站原创、仅合成数据 | 不要替代 GPU/物理精度复现 |

### 思考与追问

1. 两次相机重定位之间存在未知 Sim3 尺度，如果物体始终不动却在图像中有明显位移，你如何区分视差与物体运动？
2. 如果视频模型在第一段把夹爪与工件“画成已接触”，但第二段的状态重建已经写回，究竟该回滚场景版本还是纠正策略？
3. 人类 EGO 预训练后，视频 LPIPS 下降，但 VLA 的 success rate 不变，应该先排查数据接口、任务标签还是“世界模型到策略”的优化桥梁？

**阅读下一站：** [EgoWAM：训练时的未来世界目标](../01-foundations/world-models/egowam-world-targets-hpt-cotraining/) → [τ₀-VLA：世界模型辅助任务搜索](../01-foundations/world-models/tau0-vla-ttc-memory-search-source-audit/) → [MINERVA：LIBERO 评测协议](./minerva-libero-capacity-and-generalization/)。
