---
title: "ACE-Ego-0 源码级深读：相机动作、形态与可信 EGO 伪标签"
description: "从6K小时人机数据到VLA后训练，辨析22D论文动作与23D公开Common23、URDF形态、动态物理时间、分层Huber和真实开源边界"
sidebar:
  order: 14
---

> **核对日期 2026-11-14。** 原始资料：[ACE-Ego-0 论文（2026-06-15）](https://arxiv.org/html/2606.17200v1)、[作者 GitHub](https://github.com/ACERobotics-VLA/ACE-Ego-0)、[官方公开 SFT 代码说明](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/docs/sft_training_implementation.md)。本文明确标注「论文」「已发布源码」「教学推导」「建议实验」。**本站只提供 CPU 合成数据数值合同，未复现作者 GPU 预训练、仿真成功率或真机实验。**

# 为什么 EGO 数据不能直接变成机器人动作标签？

今天的关键问题：工业现场可以采集大量 EGO 视频，但由相机得到的手腕伪轨迹并非机器人可执行的末端真值。ACE-Ego-0 通过 **相机坐标、形态条件、时间对齐、可信度权重**四项合同进行人机联合训练，和本站前一天整理的 [SPAR 无动作 EGO 视觉监督](../../01-foundations/world-models/ego-guidance-spar-actionfree-source-guide/) 构成两条互补路线。

![ACE-Ego-0 人类与机器人数据联合训练架构](/Embodied-AI-Roadmap/diagrams/ace-ego0-system.svg)

*图1｜根据论文研究路线和官方公开代码重绘。论文完整预训练路线与仓库当前的 Common23 SFT 不是同一个可复现实验。*

## 1. 输入、输出与总体架构

【论文】使用 **4.53K 小时机器人/仿真数据**和 **1.48K 小时 EGO 人类视频伪动作**。人类视频来源 Ego4D、EgoExo4D、EPIC-KITCHENS-100、HOI4D、EgoDex 和 Xperience-10M。论文模型由 Qwen3-VL-4B-Instruct 视觉语言主干及 Flow Matching DiT Action Expert 组成；Action Expert 36 层、hidden 1024、16 heads，推理以 4 步积分形成动作块。

| 数据路由 | 模型可见输入 | 标签来源 | 训练目标 | 不应误解 |
| --- | --- | --- | --- | --- |
| Robot/Sim | 当前 RGB、文本、机器人 state、URDF token | 传感器记录的 EEF 真动作 | 有效维度上的 Flow MSE | 目标还须满足控制约束 |
| Human EGO | 当前 EGO RGB、文本、人类 surrogate token | 重建的手腕与开合伪动作 | 可靠性加权 Huber 辅助项 | 不等于机器人真动作 |
| 推理 | 目标本体当前观测与语言、shape、state | 不使用未来标签 | 未来 camera-frame 动作块 | 需逆变换和 IK |

**四个跨域风险**：不同本体空间系不统一；相同 EEF 位姿可能不可达；相同 30 帧对应不同物理时长；手部重建噪声可能使 Action Expert 学偏。方法把四件事分别约束，而不是盲目混合样本。

论文给出的 RoboCasa GR1 TableTop 72.8%、RoboTwin2.0 Easy/Hard 91.12%/90.62%、真实双臂 ARX 平均 78.3% 均为**作者不同协议的结果**，不可相互排名，也不是本站复测。

## 2. 相机空间统一：从世界、手腕到目标本体

定义世界或底座 S、头相机 C、末端 E，用明确的坐标变换方向表示：

$$
T_{C\leftarrow S}=\begin{bmatrix}R_{C\leftarrow S}&t_{C\leftarrow S}\\0&1\end{bmatrix},
\quad p_C=R_{C\leftarrow S}p_S+t_{C\leftarrow S}.
$$

姿态（不同于位置）只乘旋转：

$$
R_{C\leftarrow E}=R_{C\leftarrow S}R_{S\leftarrow E}.
$$

目标本体真正执行前，必须反变换：

$$
T_{S\leftarrow E}=T_{S\leftarrow C}T_{C\leftarrow E},
\qquad T_{S\leftarrow C}=T_{C\leftarrow S}^{-1}.
$$

【手算反例】如果 source→camera 是绕 Z 正旋 90°，source 点 (1,0,0) 在 camera 是 (0,1,0)。用逆旋转误算为 (0,-1,0)，将造成左右方向相反的监督。

**重要工程边界**：头戴相机每帧都在动，变换应记作 \(T_{C_t\leftarrow S}\)，不能假设整个 episode 使用第一帧外参。固定相机若重新挪动也必须重新标定；camera-frame 一致性无法消除单目米制尺度、TCP 和碰撞/动力学误差。

### 2.1 姿态 6D 不是六个独立角

模型用旋转矩阵前两列表示 rot6d：

$$
r_{6D}(R)=[R_{:,1};R_{:,2}]\in \mathbb R^6.
$$

如果模型输出两个 3D 向量 u、v，重建合法旋转需做正交化：

$$
b_1=\frac{u}{\|u\|},\quad
b_2=\frac{v-(b_1^\top v)b_1}{\|v-(b_1^\top v)b_1\|},
\quad b_3=b_1\times b_2,\quad R=[b_1,b_2,b_3].
$$

当 u/v 共线或范数近零时，旋转退化必须拒绝。投影点还要先检查 Z>0，不能只凭二维 overlay 判断几何有效：

$$
u_\mathrm{pix}=f_xX/Z+c_x,\qquad v_\mathrm{pix}=f_yY/Z+c_y.
$$

## 3. 最有价值的源码纠错：论文22D vs 公共SFT Common23

【论文 Appendix】每手落盘为 xyz3+Euler3+gripper1+activity1 = 8D，两手原始 **16D**；训练转换为 xyz3+rot6d6+gripper1+activity1 = 11D/手，合计 **22D**：

$$
a_\mathrm{paper}=[p_L^3,r_L^6,g_L,\alpha_L,p_R^3,r_R^6,g_R,\alpha_R]\in\mathbb R^{22}.
$$

activity 用于表示手是否在工作，和“手被检测到”的视觉置信度不是一个概念。不能把活动手无效时的坐标写零当成真实零位。

【已发布源码】[common23.py](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/src/ace_ego_0/data/common23.py) 的 **Common23** 固定为：

| Slice | Common23 字段 | 注意 |
| --- | --- | --- |
| 0:3 / 3:9 | 左 EEF xyz / rot6d | 前 9 维 |
| 9:12 / 12:18 | 右 EEF xyz / rot6d | 论文原索引已变化 |
| 18 / 19 | 左右 gripper | 论文中的 gripper 在每只手后 |
| 20:23 | waist3 | 论文 22D 并无 waist |
| 缺失 | 左右 activity | 需另用有效掩码和元数据 |

**Common23 不是论文22D再补1维；直接填零或 reshape 会制造“shape正确、语义错误”的监督。** 公共 SFT 把 RoboCasa Raw33、ARX Raw20 映射至 Common23；RoboCasa 将 6D 手部关节信息简化为每手 1D gripper；ARX waist3 是零并在 loss 中 mask=False。此处 gripper 数值并不等价于真实夹持力。

### 3.1 Absolute 和 Delta 动作也不同

【官方 recipe】[RoboCasa delta_sft.yaml](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/configs/robocasa24/delta_sft.yaml) 设定 state_anchor、first_state、delta_xyz/rot=true、delta_quat_mode=subtract。假设 anchor=(1,2,3)，未来=(1.4,2.1,3.5)，相对标签=(0.4,0.1,0.5)。训练阶段顺序应为：

1. 读取原始列与对应 camera role；
2. 按 recipe 做 absolute/delta；
3. 按对应数据集附带的 q99 统计归一化；
4. 组装 Common23 并构造可训练维度 mask；
5. 推理将输出反归一化、还原 anchor 后变换至控制接口。

**rot6d 的逐元素 subtract 是该公开 recipe 的数值定义，不等于 SO(3) 上的真正旋转差 \(\log(R_t^\top R_{t+k})\)。** 不能将 absolute checkpoint、delta dataset、不同 norm manifest 混用。

## 4. Morphology token：统一动作还需知道执行结构

【论文】视觉看到的“目标点在相机右侧 10cm”不能告诉模型某种 6 轴或 7 轴机械臂是否能伸到。ACE-Ego-0 为 Action Expert 注入机器人本体 URDF 形态 token，图节点包含关节类型、轴、限位和左右链等信息，每个节点使用约 29D 结构特征；通过图消息传递聚合：

$$
H^{(\ell+1)}=H^{(\ell)}+\phi_\ell([H^{(\ell)};\bar A H^{(\ell)}]).
$$

经过 body node pooling 和左右操作链 pooling，投影成 body / chain tokens。人类数据没有机器人 URDF，使用每来源可学习的 surrogate embedding，而非声称推断了真实人体动力学。

【源码】[urdf_encoder.py](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/src/ace_ego_0/model/modules/urdf/urdf_encoder.py) 可检查残差 message passing 与两个 pooling；公开 SFT 使用预构建的 URDF graph cache，要求 **urdf_auto_build_cache=false**。作者区分 GR1 训练的 34-node cache 与 IK/渲染的 58-joint with-hands URDF。**能解析一个新 URDF，不等于能把它的 cache 直接替换训练的形态输入。**

## 5. 时间对齐：30帧不等于同一个物理任务

【论文】统一目标时长 \(T^\star=2s\)，不同来源以采样率 \(f_d\) 决定 chunk：

$$
H_d=\operatorname{round}(f_dT^\star).
$$

| FPS | 同样2秒的预测窗口 | 如果固定 H=30 |
| --- | ---: | ---: |
| 10 | 20 steps | 3.0s |
| 20 | 40 steps | 1.5s |
| 30 | 60 steps | 1.0s |

训练可按 task cluster、episode phase、horizon buckets 构造相近批次，减少模型对互相矛盾时间刻度的拟合。**future action 是训练标签，不是推理可提前看到的观测。** 多相机需对齐真实曝光时间、设备时间偏移和执行时间，片段末端不应无条件 clip 未来索引到最后一帧。

![ACE-Ego-0 时间与动作接口、分层损失细节](/Embodied-AI-Roadmap/diagrams/ace-ego0-detail.svg)

*图2｜固定秒数的不同 FPS，论文22D/公开23D 的对照及 Human loss 可靠性通道。*

**论文 vs 公开 SFT**：官方 [trainer.py](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/src/ace_ego_0/training/trainer.py) 强制 **H=30，future=29，past=0**，并拒绝 dynamic_action_chunking。它是下游任务 SFT 的特定约束，并没有公开完整论文 2s 多源动态 chunk 预训练。不能把 SFT 运行通过宣传为完整预训练复现。

## 6. Flow Matching：真实动作 MSE，人类伪动作分层 Huber

令采集时间为 t，Flow Matching 的**插值时间**为 s（两者不能混为一谈）。\(\epsilon\sim\mathcal N(0,I)\)，真动作 a：

$$
x_s=(1-s)\epsilon+sa,\qquad v^\star=a-\epsilon.
$$

动作专家要回归速度 \(\hat v\)，机器人主损失是有效动作维度掩码 M 的 MSE：

$$
\mathcal L_\mathrm{robot}=
\frac{\sum_{t,j}M_{t,j}(\hat v_{t,j}-v^\star_{t,j})^2}
{\max(1,\sum_{t,j}M_{t,j})}.
$$

【教学手算】a=0.8、noise=-0.2、s=0.3 时，x=0.1、target velocity=1.0；**不是**直接回归 a=0.8。部署积分得到动作块。

人类伪动作 \(\tilde a\) 由视觉重建，强遮挡、相机尺度漂移和手部翻转产生长尾误差，因此采用 Huber 并按动作维 j、数据源 d、手侧 h、时间 t 赋权：

$$
W_{t,j}=\rho_j\,w_\mathrm{data}(d,h(j))\,w_\mathrm{step}(t,h(j)).
$$

动态权重由位置一阶与二阶差分构成：

$$
d_{t,h}=\|p_{t,h}-p_{t-1,h}\|,
\quad
j_{t,h}=\|p_{t+1,h}-2p_{t,h}+p_{t-1,h}\|,
\quad
q_{t,h}=\max(d_{t,h}/\tau_d,j_{t,h}/\tau_j).
$$

$$
w_\mathrm{step}=\begin{cases}1,&q\le1\\
\max(0.2,\exp[-1.5(q-1)]),&q>1.
\end{cases}
$$

【论文参数】手位置 xyz 通道 \(\rho=1\)；旋转和 gripper \(\rho=0.001\)；source prior 在 0.25–1 范围；阈值基于各来源/每只手的 95 分位；人类伪动作先做 3 帧平滑；Huber 取 1.0；**Human loss 的总系数为 0.1**。这不等于让所有伪标签都等价降权：高可信手位置还可以影响 Action Expert；未知的旋转与开合几乎不贡献人类梯度。

$$
Z=\sum M_{t,j}W_{t,j},\qquad
\mathcal L_\mathrm{human}=
\begin{cases}
\frac1Z\sum M_{t,j}W_{t,j}\operatorname{Huber}_1(\hat v_{t,j}-\tilde v^\star_{t,j}),&Z>0\\
0,&Z=0
\end{cases}
$$

$$
\mathcal L=\mathcal L_\mathrm{robot}+0.1\mathcal L_\mathrm{human}.
$$

【工程易错】全部 mask=False 时必须返回零损失并记录空监督，不允许 0/0；NaN 乘 mask=0 仍可能污染计算图，应先 finite check 和安全填补。Human Huber 与 Robot MSE 量纲不同，不能只比较原始 loss 数值就得出“EGO 比机器人学得更好”的结论。

**真实快速接触也会造成大二阶差分。** 纯 jump/jerk 抑制可能错杀插装、重抓和碰撞恢复最有价值的瞬间，应结合物体运动、任务状态、夹爪事件和 SLAM 重投影残差判断。

## 7. 六来源 EGO 的五阶段数据生产线

【论文及其 Appendix】从 RGB 到动作不是一次 HaMeR 推理：

| 阶段 | 方法与输入输出 | 留存/质检 |
| --- | --- | --- |
| 1 视频筛源/切片 | 六来源 EGO，4–30s 动作片段 | 视频 ID、session、许可、split |
| 2 操作语义过滤 | 人脸排除、动词+物体名词联合判定 | 高脸置信度>0.5过滤，操作意图 |
| 3 时空重建 | SAM3 手追踪、HaMeR MANO、VIPE 相机、时序优化 | kp≥0.4、轨迹≥15帧、跟踪丢失 |
| 4 伪动作生成 | 世界平滑→相机系腕位姿→夹爪代理 | 16D落盘、22D训练、代理距离0.04–0.10m |
| 5 异常治理 | NaN、尖峰、静止、二手关联和帧一致性 | quat norm误差≤1e-3、>3σ尖峰比例约5% |

时序优化常用平滑正则（教学形式）：

$$
\mathcal L_\mathrm{smooth}=\sum_t\|p_{t+1}-2p_t+p_{t-1}\|^2.
$$

这项会减少 tracker 抖动，也可能抹去真实接触阶段；不能只追求轨迹平滑。先在全局系重建、时序优化，再利用当前帧的 \(T_{C_t\leftarrow W}\) 转回相机系。不得把 head-camera 与 world action、meters 与 centimeters 混在同一 episode。

【建议工程 manifest】至少明确 source_clip_id、calibration_version、timestamp_clock、action_layout（paper22/common23）、rotation_encoding、metric_unit、norm_manifest_hash、per-hand-valid-mask、per-channel confidence、tracking gaps、source session split、hard reject reason。质量审计与训练权重必须版本化，才能解释任务成功率波动。

## 8. 开源实现真实边界：论文6K小时不能一键复现

**截至2026-11-14 作者 main**，官方 README、[SFT 实现声明](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/docs/sft_training_implementation.md) 和训练目录表明：已经公开 GR1 RoboCasa24 推理评测、RoboCasa24/ARX Common23 SFT、Absolute/Delta 初始化权重等。**未公开**完整 449-mixture 预训练训练器、人类辅助损失实现、norm 计算/在线 merge、RoboCasa365、RoboTwin evaluator、ARX 真机 serving、dynamic chunking、optimizer resume。论文效果有独立学术价值，但不意味着任何人 clone 就能重复同样 6K 小时训练。

| 源码 | 实际支持 | 不支持/易误判 |
| --- | --- | --- |
| [common_sft.yaml](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/configs/common_sft.yaml) | action/state23、H30、DiT36×1024、4步、VLM BF16/expert FP32 | human aux 未配置 |
| [data/lerobot.py](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/src/ace_ego_0/data/lerobot.py) | LeRobot v2.x、RoboCasa ego_view、ARX high+双腕相机 | 不是通用 EGO loader/v3兼容 |
| [common23.py](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/src/ace_ego_0/data/common23.py) | Raw33/Raw20→Common23，读取已提供 q99 norm | 不生成 norm，不解析论文22D |
| [trainer.py](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/src/ace_ego_0/training/trainer.py) | 严格 H30、URDF cache、delta/absolute、SFT optimizer | 不含动态时间混训 |
| [urdf_encoder.py](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/src/ace_ego_0/model/modules/urdf/urdf_encoder.py) | 残差图传播与链聚合 | 不等于human surrogate预训练已开源 |

论文公式中的 s 可取均匀分布；公共 SFT 的 Flow 时间使用 Beta(1.5,1) 变换，且执行 4 步 repeated diffusion。记录训练 recipe 必须分开版本化，不能用公开 downstream recipe 直接复现论文的所有实验结论。

### 8.1 无 GPU 先验算动作合同

本站新增 [ACE-Ego-0 CPU 合成检查脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/ace_ego0_camera_reliability_contract_cpu.py)，标准库运行：

~~~bash
python3 examples/ace_ego0_camera_reliability_contract_cpu.py
~~~

验证 camera/world 正反变换、H/FPS物理时长、22/23D误配拒绝、state anchor delta、Flow 速度目标、通道权重、Huber 归一化和全 mask=0。**合成检查并非官方模型运行或精度复测。**

### 8.2 官方下游 SFT 复现步骤

先按 [作者 README](https://github.com/ACERobotics-VLA/ACE-Ego-0) 在 Python3.10/CUDA12.4 环境安装 PyTorch 2.6、Flash Attention、项目依赖，下载对应的预训练初始化权重、RoboCasa24 数据和 norm manifest。用下面阶梯逐层确认：

~~~bash
git clone https://github.com/ACERobotics-VLA/ACE-Ego-0.git
cd ACE-Ego-0
python -m pip install -e ".[train]"
bash scripts/install_robocasa.sh
python scripts/check_environment.py
python scripts/train.py --config configs/robocasa24/delta_sft.yaml --dry-run
accelerate launch --config_file configs/accelerate/zero2.yaml --mixed_precision no \
  scripts/train.py --config configs/robocasa24/delta_sft.yaml --max-steps 20
~~~

命令取自官方 README 的 SFT/环境流程，使用时仍须核对所克隆 commit 的参数及路径。数据或 checkpoint 未齐，dry-run 和训练都会有校验错误；不可用空文件绕过。四张 RTX 6000 Ada 或 H100 **不保证可跑完整 6K 小时 pretrain**；应先测显存峰值、tokens/s、ZeRO 通信开销，必要时缩小规模。**不能直接拿 ACE Common23 checkpoint 当 π₀.₅ 权重加载。**

## 9. 工业 EGO/UMI 的五组可控实验

建议先抽取 200–500 段含抓取、接触、重抓的 EGO 视频与配套 UMI 轨迹，以相同机器人真实数据小时数、seed、训练预算为前提：

| 组 | 方案 | 要测的假说 |
| --- | --- | --- |
| A | Robot/UMI 真动作 baseline | 控制变量 |
| B | 额外 EGO 全通道等权伪动作 | 噪声是否导致负迁移 |
| C | 额外 EGO xyz 强监督、旋转/夹爪弱监督 | 通道可信性是否有效 |
| D | C + 时间和空间 QC、任务阶段过滤 | 数据治理独立收益 |
| E | A + SPAR 式未来视觉特征监督 | EGO 纯视觉目标与伪动作比较 |

评估记录成功率、接管率、完成时间、重抓/打滑失败率、camera-frame p95误差、来源分布、有效 EGO 小时、每通道 mask 比例及 GPUh。对“抓早/抓高/失败后无效重复”不能直接宣称该论文已解决：它主要是跨本体预训练与数据策略，**执行反馈、任务恢复、在线 RL 仍需独立研究闭环**。

## 10. 自测题与结论

1. 论文22D与 Common23 差异？**论文每手含 activity；公共 Common23无 activity、多waist3，且索引排列不同。**
2. 10Hz/30Hz 固定 H30 时间跨度？**3s/1s，不能统一当2s。**
3. 旋转 rho=0.001 是否表示旋转完全不学？**否，机器人真动作仍监督有效旋转，人类旋转仅贡献极弱梯度。**
4. 人类 loss 所有 mask=False？**0，记 reject 并避免 NaN 与除0。**
5. 开源 Common23 SFT 能证明完全复现原论文6K小时吗？**不能，关键 pretrain + human loss 流程缺失。**

**总结**：ACE-Ego-0 值得吸收的是四重数据合同与可信人类伪监督，而不是机械照搬 Common23。先厘清动作 ABI 和质量来源，再做可解释的 VLA 实验。

**原始参考**：[原论文](https://arxiv.org/html/2606.17200v1) · [官方实现](https://github.com/ACERobotics-VLA/ACE-Ego-0) · [官方 SFT 发布边界](https://github.com/ACERobotics-VLA/ACE-Ego-0/blob/main/docs/sft_training_implementation.md) · [本站 HuRo 人类数据机器人化](./huro-robotization-to-vla/) · [UMI 时间与坐标合同](../../02-data-engineering/mcap-lerobot-v3-temporal-action-contract/)。
