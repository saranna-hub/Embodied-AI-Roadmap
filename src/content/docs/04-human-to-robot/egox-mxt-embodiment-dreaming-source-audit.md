---
title: "EgoX 深度源码教程：MXT+ 跨本体 Transformer 与 EMA Embodiment Dreaming"
description: "2026-11-12 EgoX 原论文、MED 开源源码、模态配置、Smooth-L1 动作头、未来 EMA token 监督和 GPU/仿真复现路线"
sidebar:
  order: 3
---

> **核查：2026-11-18。**【论文】[EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming，arXiv:2611.16728](https://arxiv.org/abs/2611.16728)，2026-11-12 发布。【源码】[egox-dream/egox 官方仓库](https://github.com/egox-dream/egox) 与 [公开数据集](https://huggingface.co/datasets/binghong0727/egox)。【教学推导】为数学说明；【建议实验】尚未执行。本站只作源码/配置核对与独立标准库 CPU 合同验证，**未训练 GPU 模型、未运行 Isaac Sim、未复测真实任务，也未获取作者训练权重**。

# 1. 核心问题：同一个抓取语义，不等于同一套关节控制

工业人类 EGO、UMI、双臂和移动本体的观测与控制维数根本不同。假如将手指位置、夹爪开合、底盘速度全部塞到统一高维 action vector，不同字段会被大量无意义的零填充掩盖。EgoX 的主张不是消灭本体差异，而是**分本体、分模态编码器及动作专家，共享操作 Transformer 主干；训练额外预测未来的本体状态潜特征**。

![EgoX 系统结构与训练推理分支](/Embodied-AI-Roadmap/diagrams/egox-mxt-system.svg)

*图1｜本站原创源码结构图：每个 embodiment 保留各自 observation/action 字段，主干共享；EMA future teacher 和 dreaming head 仅用于训练。*

EgoX 论文使用 MXT+ 名称；[当前开源代码](https://github.com/egox-dream/egox/blob/main/src/egox/models/med/model.py) 中的对应模型称 **MED / Modular Embodiment Dreamer**。不要据“Dream”误以为推理会生成视频，也不要把它误称为 Flow Matching VLA：**官方 ActionHead 使用 masked Smooth-L1 回归，不使用 π₀.₅ 那种随机噪声速度场与 ODE 采样。**

## 1.1 作者究竟报告了哪些指标？

【论文】跨本体预训练实验覆盖 13 个任务-本体组合、五项操作任务、五种真实本体与一款仿真人形。与目标任务从零训练相比，平均**归一化任务进展**从 0.63 增至 0.80，分布外从 0.53 增至 0.77。**0.80 是 normalized progress，并非 80% 成功率**。论文消融表明 broader pretraining 及 dreaming 均有帮助；作者还报告 latent geometry 平均距离降低 38.52%，这不能直接当作任务成功率提升。

【源码开放边界】[README 清单](https://github.com/egox-dream/egox/blob/main/README.md) 说明训练实现、HDF5 数据、G1-Sim-Dex3 仿真 rollout 已提供，**policy checkpoints 尚未公开**；VR 采集及多款真实本体的控制整合尚未全部放出。因而可以复刻训练流程，但不能声称拿下载好的权重即可重现实验曲线。

# 2. 输入输出：先读懂 HDF5 与本体配置

【源码】每个 episode 通过根属性标注 embodiment，数据由 [data.py](https://github.com/egox-dream/egox/blob/main/src/egox/data/data.py) 与 [mapping.py](https://github.com/egox-dream/egox/blob/main/src/egox/data/mapping.py) 解析，经过按本体分桶的 collate_by_embodiment 形成 batch：

| 输入输出键 | 典型 shape | 物理含义及陷阱 |
| --- | --- | --- |
| obs.main_image | (B,3,H,W) | RGB uint8，经模型转换到 0~1，不能把双目拼接当单目内参 |
| obs.proprio 字段 | (B,D_m) | 身体姿态、末端姿态、手部/关节，坐标系与单位由本体定义 |
| obs_mask | (B,) 或 (B,D_m) | 相机缺失或通道零填充不等于真实 0 |
| action[field] | (B,H_a,D_a) | 每字段专有未来动作块；默认 H_a=60 |
| action_mask[field] | (B,D_a) | 无效维度不参与 Smooth-L1 |
| is_pad | (B,H_a) | episode 结束后的补零时间步不计损失 |
| dream[field] | (B,K,...) | 未来 K 帧的图像/状态，训练期才用 |
| dream_pad | (B,K) | 未来帧超出 episode 的 Mask |
| output[field] | (B,H_a,D_a) | 推理输出各字段动作，按该本体统计量反归一化 |

[Human-Dex5.yaml](https://github.com/egox-dream/egox/blob/main/src/egox/configs/robots/Human-Dex5.yaml) 使用前视 RGB、6D body pose、12D 双手 EEF pose、左右 6D 相对 EEF pose、左右各 15D link 表征；并不代表纯 RGB 视频可直接提供这些标签。[G1-Sim-Dex3.yaml](https://github.com/egox-dream/egox/blob/main/src/egox/configs/robots/G1-Sim-Dex3.yaml) 还配置双腕图像、29D 主体关节、左右各 7D 手关节和 3D 底盘速度。**控制维数只是协议字段，不能在两个本体间按数值直接对齐。**

## 2.1 Episode 尾部的“双 Mask”必须独立

源代码的默认 dream_offsets 为 [20,40,60] **帧偏移**，不是秒数。数据集从 t 取 60 帧动作，不足处填零，另给 is_pad；对于越界的未来观测，使用 episode 最后一帧作数组占位，但同时设置 dream_pad=True。这最后一帧不能当未来真实标签。

例如 episode 长 100，当前 t=85：动作块仅 15 帧有效，而 t+20、t+40、t+60 均越界。动作 loss 只计 15 帧；**未来 latent 监督一项也不能算**。

另一个跨数据源陷阱是采样率：对 30Hz EGO 的 20 帧是约 0.67 秒；对 10Hz 本体则是 2 秒。应记录相机时间戳、动作时间戳、Frame Tree、fps 和标定版本。若统一未来真实时长，需重采样和同步，不得直接把整数 offsets 强加于多种设备。

# 3. 网络架构：512D 的 DETR 式 encoder-decoder

![EgoX query/action/future 三种长度以及 mask](/Embodied-AI-Roadmap/diagrams/egox-mxt-detail.svg)

*图2｜主干 query 的数量、ActionHead 预测的时间长度、DreamHead 的未来偏移，是独立的三条轴。*

【源码】[transformer.yaml](https://github.com/egox-dream/egox/blob/main/src/egox/configs/model/transformer.yaml)：hidden_dim=512、nheads=8、FFN=1024、encoder/decoder 各6层。主干不是自回归语言 Transformer，而是基于 DETR 的 encoder-decoder；视觉图像经 ResNet18 与 query pooling，状态经 MLP/tokenizer。图像 tokenizer 是否共享取决于 image_sharing 配置（none / encoder / tokenizer），并非一律全共享。

对本体 e 的模态 m，tokenizer 得到：

$$
Z_{e,m}=f_{e,m}(o_{e,m})\in\mathbb{R}^{B\times N_{e,m}\times512}.
$$

拼接合法 tokens 并编码：

$$
M_e=\mathrm{Encoder}_{\theta}\left(\mathrm{Concat}_m Z_{e,m},\ \mathrm{obsMask}_e\right).
$$

每个动作字段 k 由自己分配的 learnable queries 经 shared decoder 和本体专属 head：

$$
U_{e,k}=\mathrm{Decoder}_{\theta}(Q_{e,k},M_e),\qquad
\hat A_{e,k}=g_{e,k}(U_{e,k})\in\mathbb R^{B\times H_a\times D_{e,k}}.
$$

【源码】trunk 的 output_len 设置 body_pose=24、左右 EEF 相关字段=24、左右 link 字段=30、velocity_commands=30。这是**decoder memory query 预算**，不代表动作只有24或30帧。ActionHead 自己为每个未来动作步准备时间 query，因此可以使用 24 个 trunk tokens 预测 60×D 的 chunk。请核对 [transformer.py](https://github.com/egox-dream/egox/blob/main/src/egox/models/med/transformer.py) 与 [decoders.py](https://github.com/egox-dream/egox/blob/main/src/egox/models/med/decoders.py)。

## 3.1 Smooth-L1 的分母必须是有效元素数

官方动作回归采用 Smooth-L1，beta=0.05：

$$
\rho_\beta(r)=\begin{cases}\frac{r^2}{2\beta},&|r|<\beta\\ |r|-\frac{\beta}{2},&|r|\ge\beta.\end{cases}
$$

Mask 由有效时间步和有效通道做 AND：

$$
V_{b,t,d}=\neg \mathrm{isPad}_{b,t}\ \land\ \mathrm{actionMask}_{b,d}.
$$

从而字段损失为：

$$
\mathcal L_{e,k}=
\frac{\sum_{b,t,d}V_{b,t,d}\rho_{0.05}(\hat A_{b,t,d}-A_{b,t,d})}
{\max(1,\sum_{b,t,d}V_{b,t,d})}.
$$

模型先按本体/字段统计量做 $(a-\mu)/\sigma$ 标准化，推理再反归一化。一个被 Mask 的关节维度，即使其文件里有极大异常数，也不应影响 loss；如果固定用 B×H×D 作分母，会让缺失多的本体被系统性低估。

# 4. Embodiment Dreaming：训练时预测未来“本体潜状态”，不输出未来图像

【源码】[dreaming.py](https://github.com/egox-dream/egox/blob/main/src/egox/models/med/dreaming.py) 中的 EmbodimentTokenizer 将未来多模态信息压成少量 tokens；DreamHead 从**当前 shared trunk** 输出预测未来 EMA encoders 产生的 latent targets。默认 dream_offsets=[20,40,60]，每未来帧 token_num=8，维度512。

$$
Y^{\mathrm{EMA}},\hat Y\in \mathbb R^{B\times3\times8\times512}.
$$

人类 Human-Dex5 的 dream modalities 仅用 main_image；G1-Sim-Dex3 的 dreaming 汇聚 main_image、机器人关节与左右手关节。这意味着未来潜状态**本体相关**，不是随意抽象成一个与本体无关的机器人 3D pose。

EMA 更新（官方 decay=0.999）：

$$
\bar\phi_{s+1}=0.999\bar\phi_s+0.001\phi_{s+1}.
$$

EMA teacher 在 no_grad 中生成标签，dream loss 的梯度穿过 DreamHead 和 online trunk/tokenizer，却不能直接更新 teacher。EMA 以训练后的 online 参数平滑更新，**teacher 与 student 不能从同一未来观测直接互相抄答案**。

【源码】dream_loss 同时比较 token 的方向与模长：

$$
d_{\rm dir}=1-\frac{\hat y^\top y}{\|\hat y\|_2\|y\|_2},\quad
d_{\rm mag}=\mathrm{SmoothL1}(\|\hat y\|_2-\|y\|_2).
$$

$$
\mathcal L_{\rm dream}=\frac{1}{\max(1,|\mathcal V|)}
\sum_{i\in\mathcal V}\left[d_{\rm dir}(\hat y_i,y_i)+0.01d_{\rm mag}(\hat y_i,y_i)\right],
$$

其中有效集合 $\mathcal V$ 是 dream_pad 与模态有效性共同约束；G1-Sim-Dex3 yaml 中的 dreaming loss_weight=0.3：

$$
\mathcal L_e=\sum_k w_{e,k}\mathcal L_{e,k}+0.3\mathcal L_{\rm dream}+\mathcal L_{\rm norm}.
$$

【教学推导】如果 $\hat y=(3,0)$、$y=(6,0)$，则 cosine=1、方向损失0，但模长损失大于0；仅用方向 loss 会忽略缩放误差。当未来帧无效时，其两个分项都必须被 Mask 掉。

**推理接口**：官方 MED.forward 默认返回动作字典；只有显式 infer_dream=True 才另输出预测 latent，用于调试。真实未来图像在推理时**绝不能**作为输入。dream head 可以不参与部署实时控制，不能把此法误当成在线视频世界模型搜索。

# 5. 源码关键路径与数值合同

| 路径 | 关键功能 | 读源码重点 |
| --- | --- | --- |
| src/egox/data/data.py | HDF5、未来 offsets、Mask、按本体 collate | 处理 episode 越界与没有相机的样本 |
| src/egox/configs/robots/*.yaml | 本体字段与未来模态 | 15D/29D/3D 不是可互换单位 |
| src/egox/configs/model/transformer.yaml | query 数/宽度/层数 | output_len 不等于 H_a |
| src/egox/models/med/transformer.py | DETR Encoder+Decoder、query slices | Attention mask 的 True/False 语义 |
| src/egox/models/med/decoders.py | Smooth-L1/动作 head | 有效维度数作为分母 |
| src/egox/models/med/dreaming.py | EMA latent 的 cosine+norm loss | future padding 和方向/模长损失 |
| src/egox/models/med/model.py | 两个损失和 EMA 更新、推理边界 | action 不依赖训练未来真值 |
| scripts/training_scripts/G1-Sim-Dex3/ | 预训练、微调、from-scratch | 训练集混合与 target task 泄漏 |

本站提供**独立于 PyTorch 的 CPU 合同**：

~~~bash
python3 examples/egox_ema_mask_contract_cpu.py
~~~

检查有效动作元素计数、通道遮蔽、未来偏移越界、cosine 与模长、EMA 0.999、query 与 horizon 分离，以及未来真值不进入推理输入。这是**数学与样本契约验证**，不是官方网络前向推理。

# 6. 从下载数据到训练、仿真评测的真实复现入口

以下为官方 README/脚本的公开步骤，尚未由本站实际运行。建议先记录 commit、许可、数据磁盘空间、CUDA/Isaac Sim 版本：

~~~bash
git clone https://github.com/egox-dream/egox.git
cd egox
git rev-parse HEAD
bash scripts/setup_conda_policy.sh --name egox
conda activate egox
python scripts/download_data.py --embodiment g1-sim-dex3 --dry-run
python scripts/download_data.py --embodiment g1-sim-dex3
~~~

全本体预训练前使用下载器拉取其他 embodiment 的 HDF5 数据，然后运行作者的 G1-Sim-Dex3/Kitchen-Bread 配方：

~~~bash
python scripts/download_data.py
bash scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/pretrain.sh --no-wandb
bash scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/finetune.sh \
  /path/to/pretrained/policy_last.ckpt --no-wandb
~~~

【源码】预训练排除目标任务 G1-Sim-Dex3/Kitchen-Bread，但保留同一本体的 Kitchen-Cleanup，并混合其他六种 embodiment；脚本默认 **image_sharing=encoder、per_emb_batch_size=48、bf16、100000 steps**，同时排除预训练阶段两路腕相机。微调时使用目标任务数据和腕相机（batch_size=32），必须检查预训练权重哪些组件成功加载、哪些是新初始化。**同名 checkpoint 能载入，并不代表每个 action expert 已被充分训练**。

仿真 rollout 需要作者的 Isaac Sim/Isaac Lab 环境和完整模型目录：

~~~bash
bash scripts/setup_conda_policy.sh --sim --name egox-sim
conda activate egox-sim
bash scripts/rollout_scripts/G1-Sim-Dex3/kitchen-Bread/headless.sh \
  --model_path /path/to/finetune/run
~~~

最终模型目录必须包含 policy_last.ckpt、all_configs.json 和 dataset_stats.pkl。缺失后两者会损坏本体动作接口/标准化语义。官方真实本体控制与 VR 采集仍有开放限制；不能宣称本教程已具备端到端工业现场落地能力。

# 7. 工业 EGO/UMI + π₀.₅：哪些可以迁移、哪些不能照搬？

**可迁移的是训练范式而不是控制头权重。** 数据接入时先按本体建立 observation 类型、动作字段、局部与世界坐标、时间戳、标定版本、缺失通道、可靠性等级；再尝试共享视觉表征与未来变化监督。UMI 的六维末端轨迹、夹爪开合与 EGO 的手指 link 状态不应只凭 shape 合并。标定变了必须更新 T_world_camera / T_world_EEF；FPS 不同必须按物理时间定义未来状态。

若大量 EGO **只有 RGB、没有可执行动作标签**，官方 MED 无法直接以这些数据计算 ActionHead 的真实行为克隆损失。可以探索将 EMA future latent head 作为 π₀.₅ 视觉 Adapter 的训练期辅助监督，但这是本站**建议研究方案**，不是论文已实现的 openpi 功能。π₀.₅ 原生动作专家使用 Flow Matching，EgoX 使用 Smooth-L1；不能直接交换权重或推理采样器。

**六组可归因消融**：A 目标本体从零训练；B 跨本体无 Dream；C 跨本体带 Dream；D 去掉人类 EGO；E 粗暴统一动作头；F 置信度/质量分桶的人类 EGO。保持微调样本、训练更新次数、模型参数预算、随机种子与摄像头配置一致，报告成功率、任务进展、接管率、未见工件 OOD、不同人示教 OOD，至少多 seed 并有区间。否则无法区分收益究竟来自更多数据、更大训练步数还是未来监督。

# 8. 用四道题自检是否真正理解

**问题1：** trunk 中 body_pose 只有 24 个 query，为何预测 60 帧不矛盾？**答：**24 是 decoder 中该字段 memory 的 token 数；ActionHead 的 60 个动作时间 query 通过 cross-attention 读取这些 memory。

**问题2：** 训练 clip 长 100、起点 85、future offsets=[20,40,60]，能否把最后一帧当未来？**答：**只能数组占位，dream_pad 全为 True，dream loss 应为0。

**问题3：** 方向完全一致而模长偏差很大，cosine loss 是否惩罚？**答：**不能；额外 magnitude Smooth-L1 正是为此设计。

**问题4：** 纯人类 RGB 能否直接按官方 Human-Dex5 训练？**答：**不能；缺少 body/EEF/link 动作监督，必须另构造合法动作标注或设计不依赖动作标签的辅助训练目标。

---

**来源**：[论文](https://arxiv.org/abs/2611.16728) · [官方项目](https://egox-dream.github.io/) · [官方仓库与开放清单](https://github.com/egox-dream/egox) · [开源 HDF5](https://huggingface.co/datasets/binghong0727/egox) · [训练脚本](https://github.com/egox-dream/egox/tree/main/scripts/training_scripts) · [原生源码](https://github.com/egox-dream/egox/tree/main/src/egox/models/med)。论文实验属作者结果；本站没有宣称 GPU/仿真/真实硬件复现。
