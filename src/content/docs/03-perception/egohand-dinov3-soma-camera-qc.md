---
title: "EgoHand 2026 源码级教程：DINOv3 + SOMA-Hand，从单目 EGO 到世界坐标手部标签"
description: "核对 NVIDIA EgoHand 论文、IsaacCapture 推理代码与质量控制：相机内参与 Crop、RayPE、SOMA-25、2D/3D 双分支、平移闭式推导、SLAM 坐标变换、GPU 复现、CPU 合成契约及工业双手数据质检"
sidebar:
  order: 2
---

> **2026-11-12 调研。** 论文于 **2026-10-21** 发布（[arXiv:2610.30293](https://arxiv.org/abs/2610.30293)）；本页逐项核对 [NVIDIA 官方推理指南](https://nvidia.github.io/IsaacCapture/main/ego_post_processing/egohand.html)、[IsaacCapture 官方代码](https://github.com/NVIDIA/IsaacCapture/tree/main/src/python/isaaccapture_egopost)、[官方权重 EgoHand-v1](https://huggingface.co/nvidia/EgoHand-v1) 与 [GEM-X 训练代码入口](https://github.com/NVlabs/GEM-X)。**【论文】**是作者报告；**【代码】**是官方现有接口；**【推导】**是本站为理解原理做的独立推导；**【实验方案】**是建议而非已完成复现。本站附纯 CPU 数学/数据契约检查，**未运行官方 GPU 权重、未完成手部精度复测，也未证明能直接产出机器人可执行动作**。

# 从 EGO 视频中恢复可靠的手部几何：EgoHand 能解决什么，不能解决什么？

如果你已经学过 [HaMeR 的单帧手网格](./hamer/) 和 [HaWoR 的时序世界重建](./hawor/)，下一步需要回答工程上更严苛的问题：同一名工人戴头部摄像头做双手装配，换一个相机 FOV、改变手在画面中的位置、产生遮挡，**模型能否稳定给出相机系下有米制含义的双手位姿，并在十万小时数据规模上留下可复核的质量证据？**

EgoHand 的思路不是重新发明 SLAM，也不是直接训练 VLA。它把**手部定位、相机条件化几何回归、SOMA-Hand 形状先验、2D 关键点校正和时序质检**拆为可追踪步骤；世界坐标必须依赖**额外提供且同步的相机轨迹**。

![图一：从 EGO 视频、内参、手框到 SOMA、校正、QC 与世界坐标的系统图](/Embodied-AI-Roadmap/diagrams/egohand-system.svg)

*图 1｜原创教学图：灰色虚线表示外部输入；检测、推理、后处理、SLAM 与 VLA 标注之间的责任边界不可省略。*

## 0. 先纠正五个容易造成工程事故的认识

1. **【论文】模型用约 1400 万张图像训练，SOMA-Hand 有 25 个关节**。这不是“MANO-21 关节直接替换”：拓扑、手型参数和尺度接口都不同。
2. **【代码】网络输出的是相机坐标下的腕位置 `cam_t`，而 `joints`/`vertices` 是腕相对局部坐标**。忘记加 `cam_t` 会让所有手落在相机原点。
3. **【代码】输出 `frame_names` 只包含成功预测的帧**，左右手可能缺帧且彼此不同。不能拿同一数组行号直接和视频/SLAM 的第 t 帧相乘。
4. **【论文】约 9 ms 是特定 GPU/配置下的模型推理耗时**，不包括视频解码、手检测、预处理、SLAM 和后处理。不能据此宣布整套管线 100 FPS。
5. **【推导】2D 投影对齐好不意味着绝对深度真实。** 若给错焦距，得到的深度可能成比例偏移，且用同一错误内参渲染的 Overlay 依旧看起来对齐。**必须有独立米制标定或参考真值。**

## 1. 输入与输出的数据合同

| 阶段 | 必须输入 | 中间表示 / shape | 容易犯的错误 |
| --- | --- | --- | --- |
| 视频解码 | EGO `.mp4`、FPS、帧 ID | 原始图像 [H,W,3] | 以 decode 序号代替曝光时间 |
| 双手检测 | 2D 手框或 RT-DETR | `[cx,cy,side]`，**原图像素** | 将左/右按画面方位而非解剖手分组 |
| 相机模型 | `K=(fx,fy,cx,cy)`、畸变类型 | PINHOLE / KB4 / FISHEYE624 / RATIONAL_POLYNOMIAL | 鱼眼当针孔直接 Crop，或将 resize 后 K 当原 K |
| 单手推理 | 256×256 Crop 与相机几何 | DINOv3 特征、3D/2D 两个 Decoder | 丢失 Crop 原图位置 |
| SOMA 重建 | 25 关节旋转、形状系数、尺度 | `hand_poses`: [N,25,3,3]；`joints`: [N,25,3] | 当作 MANO-21 或欧拉角 |
| 相机位置 | Camera Head + 2D 对应点 | `cam_t`: [N,3]，**米** | 将腕相对 `joints` 直接视为相机系 |
| 世界坐标 | 逐帧 `T_WC(t)`、时间对齐 | 双手世界坐标轨迹 | 误把 `T_CW` 当作 `T_WC` |
| 训练样本 | QC + 任务语义 + 动作适配 | hand/object 观测、伪标签、置信字段 | 把人手关节当作机器人夹爪命令 |

官方默认模型采用 **DINOv3 ViT-H/16+** 特征编码器，初始化路径与 SAM 3D Body 有关；解码使用 3D 手部查询和 2D 关节查询，两者共享图像特征但各有任务。重要区别是 **SOMA-25 ≠ 25 个独立控制自由度**，而 `identity_coeffs` 与 `scale_params` 也不是每帧都应该变化的真实人体参数。

### 1.1 官方 NPZ 字段特别容易误读

[官方输出字段定义](https://nvidia.github.io/IsaacCapture/main/ego_post_processing/egohand.html#result-arrays) 中：`left_results.npz`、`right_results.npz` 分开保存；`frame_names` 是零基**源帧索引字符串**，缺失帧不补零；`hand_poses` 为 25×3×3 的 SO(3) 矩阵；`identity_coeffs` 为 [N,20] 或 [N,5]；`global_scale` 为 [N]；可选 `scale_params` [N,26]；`vertices` 为 [N,Vh,3] 且模型决定 Vh；`joints` 为 [N,25,3]；`cam_t` 为 [N,3]。双手帧列表可能完全不同。

相机约定 **x 向右、y 向下、z 向前**。得到相机系关节的公式是：

$$
P_{C,t,j}=J^{\mathrm{wrist}}_{t,j}+t^{\mathrm{cam}}_t,\qquad j=0,\dots,24.
$$

指根与指尖次序按官方 SOMA-25 定义（腕 0，拇指 1–4，食指 5–9，中指 10–14，无名指 15–19，小指 20–24）。**不要直接用 MANO 的 joint index 训练同一个模型或计算误差**；论文评测对齐 MANO 时单独进行拓扑映射/拟合，存在额外约定。

## 2. 架构：为什么 Crop 还需要 CLIFF 和 RayPE？

图像 Crop 将一只手剪出来并统一缩放至 256×256，确实让视觉编码器关注手指。代价是：手在画面左上或右下、手框距离主点多远、原来的占像素大小等信息被抹去。**只看归一化 Crop，预测同一个 3D 手可能对应很多物理深度。**

![图二：相机内参、Crop 方位、2D/3D 双分支与侧向平移重投影校正](/Embodied-AI-Roadmap/diagrams/egohand-geometry-detail.svg)

*图 2｜原创教学图：2D 分支只修正侧向 `Tx,Ty`，深度 `Z` 仍由回归头给出；示意方程假设针孔与零 skew。*

**【论文】几何条件化分三条路**：

- **CLIFF-style box conditioning**：把手框中心及尺度相对于相机的关系编码到初始 Hand Query，让网络知道“这个相似手形占了多少角度/离主点多远”。
- **Ray positional encoding (RayPE)**：将每个视觉位置关联到通过内参计算出的**视线方向**。针孔近似时，像素 `(u,v)` 对应方向与 `K^{-1}[u,v,1]^T` 成比例，而不是仅有 Patch 序号。
- **独立 2D 关节查询**：通过 Cross-Attention 输出 Crop 内关键点和不确定性，成为下一步 3D 与图像的软对应关系。

令视觉 token 为 `F=E_{\rm DINO}(I_{\rm crop})`，条件化后示意为：

$$
\widetilde F_i=F_i+\mathrm{RayPE}(K^{-1}\widetilde p_i),\quad
h_{\rm 3D}=\mathrm{Dec}_{3D}(q_{\rm hand}(b,K),\widetilde F),\quad
h_{\rm 2D}=\mathrm{Dec}_{2D}(q_{1:25},\widetilde F).
$$

这里是**教学表示**，不是声称作者将 RayPE 简化为逐点加线性向量；真正代码还涉及相机模式、Crop 到全图坐标变换及隐藏特征结构。

### 2.1 参数化的手：SOMA-Hand 与 MANO 为何不能混写

EgoHand 将手势关节旋转 `R_{1:25}`、全局尺度 `s`、手部身份/形态参数 `\beta` 输入 SOMA-Hand 的 MHR shape backend，解码为局部网格和腕相对关节：

$$
(V^{\rm wrist},J^{\rm wrist}) =
\mathrm{SOMA\!-\!Hand}(R_{1:25},\beta,s).
$$

MANO 是另一套参数化的 Mesh/骨架定义，具有自己的 shape/pose 空间与 landmark 导出；**两套模型即使渲染看起来相同，参数与拓扑也不一一对应**。跨模型比较，应先确定评价关节集、统一单位、右手镜像/变换方向，再计算 MPJPE 等指标。

SOMA-Hand 能表达形态与显式骨长，但不能凭单帧恢复真正的身份尺度。官方训练设计借助**跨时间共享手型**对这种歧义加约束，下节展开。

## 3. 相机平移深度：重点读懂论文公式

设原始图像中手的方形框中心 `(c_x,c_y)`，框边长 `b`，针孔相机的主点 `(p_x,p_y)`，焦距 `(f_x,f_y)`。Camera Head 预测标准化 log-depth 残差 `\ell` 以及侧向残差 `(d_x,d_y)`。论文采用的形式可以写作：

$$
Z=\frac{f_x}{b}\exp(\ell).
$$

$$
T^{\mathrm{reg}}_x=d_x+\frac{c_x-p_x}{f_x}Z,\qquad
T^{\mathrm{reg}}_y=d_y+\frac{c_y-p_y}{f_y}Z.
$$

**【推导】** 投影关系 `u=f_x X/Z+p_x` 说明占像素尺度 `b` 近似与 `f_x/Z` 成正比；回归器学习其无量纲修正 `\ell`，相机内参再恢复米制尺度。若保持同一 `b,\ell`，却将 `f_x` 从 900 错填为 1200，模型导出的 `Z` 会从原估计的 1 倍变成 **4/3 倍**。这里 `\ell` 是网络预测量，不意味着它不随错误相机信息变化；4/3 是**固定 `\ell` 条件下的几何敏感性反例**。

为何画面上可能看不出？假设手模型的位置/尺度也随错误 K 一起缩放，错误投影器仍可把几何投影回相同像素。**同一个 K 下自洽的 Overlay 不是独立标定证据。** 最好用独立尺规、已知工件尺寸、多视角/动捕或机器人基准坐标交叉验证。

### 3.1 2D 分支不是“再预测一个深度”

【论文】网络还预测 `\widehat q_j=(u_j,v_j)` 和对应存在概率/不确定性。对于可靠关节集合 `\mathcal V`，固定 Camera Head 给出的深度 `Z`，求：

$$
(\widehat T_x,\widehat T_y)=
\underset{T_x,T_y}{\arg\min}\;\sum_{j\in\mathcal V}
w_j\left\|
\pi_K\left(J_j+(T_x,T_y,Z)\right)-\widehat q_j
\right\|_2^2+\rho(T_x^2+T_y^2).
$$

这给出**两个侧向自由度**的校正：由于使用了 2D 关键点与 3D 关节对应信息，同一帧的重投影能更好落在真实手部位置。论文的有效性判定不通过时回退 `T_{\rm reg}`，防止把单帧错误的 2D 预测强行反映到 3D。

### 3.2 【推导】把“优化一下相机”化成可检查的加权最小二乘

对零 skew 的针孔相机，令 `J_j=(X_j,Y_j,Z_j)`，`a^x_j=f_x/(Z_j+Z)`，那么：

$$
u_j-p_x=a^x_j(X_j+T_x).
$$

只优化 `T_x` 时，目标函数为

$$
L_x(T_x)=\sum_j w_j\left[a^x_j(X_j+T_x)-(u_j-p_x)\right]^2+\rho T_x^2.
$$

导数置零得到**教学版本的解析解**：

$$
\widehat T_x=
\frac{\sum_j w_j a^x_j[(u_j-p_x)-a^x_j X_j]}
{\sum_j w_j(a^x_j)^2+\rho}.
$$

`T_y` 同理，将 `x,u,f_x,p_x` 分别替换为 `y,v,f_y,p_y`。这个推导假设每点有效 `Z_j+Z>0`、图像点已恢复为原相机像素、固定 3D joints 与 `Z`、损失仅为加权重投影平方误差。官方真正的有效点选择、RLE 不确定性、畸变/透视 Crop 流程与反传细节，仍要以作者代码/论文为准。

**为什么正则项不能无限大？** 当 `\rho\to\infty`，两轴都被拉向零，2D 分支无法纠正 Crop 偏差；当 `\rho=0` 且点不足或不确定，拟合就可能不稳定。工程上应记录有效关节数、残差、深度正值与回退率，而不只是输出最终 Overlay。

## 4. 训练语义：1400 万图像为何不是简单地拼起来

【论文】训练数据大体来自**约 49 小时合成手运动、63 小时授权的真实 EGO、16 小时多机位 MoCap**，通过数据增广/采样等组成约 **1400 万图像**；不要把三类源小时数直接乘 30 FPS 就当“去重独立帧数”。多源场景提供手套/裸手、工件、鱼眼、不同骨长、不同摄像机的覆盖，但不同数据的 3D 标签质量并不等价。

核心是将不同来源整理为同一 SOMA-Hand 的监督：合成数据天然有姿态/骨架，EGO 伪标签来自 3D landmarks、SAM 3 手部 mask 和时序拟合；多机位带微型反光标记的 MoCap 提供更可信的 3D 参考。

### 4.1 为什么一个视频应该共享手型？

同一名人的手不会在 0.1 秒内突然增长 5%。设片段共 `T` 帧，共享身份参数 `\beta`，逐帧关节旋转 `\theta_t` 和位置 `p_t`，优化可概括为：

$$
\min_{\beta,\{\theta_t,p_t\}}\sum_{t=1}^{T}
\left[\lambda_J L_{\rm joints}(t)+\lambda_M L_{\rm mask}(t)\right]
+\lambda_S L_{\rm shape}(\beta)
+\lambda_T\sum_{t=2}^{T}\|p_t-p_{t-1}\|^2.
$$

这是**【教学抽象】**，不是作者完整损失原式。论文采用选取可见且姿态多样的帧先拟合共享形态、然后固定形态拟合全片动作；困难区间再单独细化并经对齐与运动检查。这样比每帧独立估计手型更不容易产生“呼吸式手掌伸缩”。

### 4.2 两个输出分支对应哪些训练梯度？

【论文】3D 路径监督包含 2D 重投影 L1、腕相对 3D 关节 L1、SOMA pose/shape 平方误差以及 Camera Head 的 log-depth Smooth L1 与侧向 offset 平方误差；2D 路径使用带不确定性的 Residual Log-Likelihood Estimation (RLE)、关节 presence BCE、位置 evidence 的 CRPS 边缘监督；有效的重投影校正还计算侧向平移损失，并允许梯度穿过求解过程。

抽象地，模型同时优化：

$$
L=\lambda_{3D}L_{3D}+\lambda_{\rm cam}L_{\rm cam}
+\lambda_{2D}L_{2D}+\lambda_{\rm solve}L_{\rm solve}.
$$

**不要误写**为只有 MPJPE 监督，或认为 2D Head 只在后处理阶段使用。实际设计让视觉证据、姿态几何与相机条件共同参与训练。权重、有效标注 mask 和数值稳定性，要从当前 pin 的训练配置逐项核对；不能从这一示意损失推断全部默认超参。

## 5. 从相机系走向世界系：SLAM 是外部依赖

【代码】给定某帧相机到世界的变换 `T_{WC}(t)=[R_{WC}(t),p_{WC}(t)]`，其中输出的 Wrist 位置和局部关节都在相机坐标系：

$$
P_{W,t,j}=R_{WC}(t)\left(J^{\rm wrist}_{t,j}+t_t^{\rm cam}\right)+p_{WC}(t).
$$

齐次形式 `\bar P_W=T_{WC}\bar P_C`。假如使用 `T_{CW}`，应先逆变换。**相机位姿必须使用与手图像曝光时刻匹配的轨迹，而不是仅靠数组行号。** 若 SLAM 输出单目相似变换而无米制尺度，EgoHand 的相机系米制坐标不意味着组合后全局轨迹就有可信米制尺度。

官方 [`world_coordinate_projection.py`](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture_egopost/inference/postprocessing/world_coordinate_projection.py) 可结合 ViPE 风格的 `pose.npz` 做相机轨迹稳定化，其关键思路是：

1. 先用 `T_{WC}(t)` 将左右手相机系 3D 关节放到世界系；
2. 对世界系局部轨迹做滤波，构造较稳定的目标；
3. 对每帧用双手关节的 Kabsch 刚性配准拟合新相机姿态，循环优化；
4. 可额外对最终世界关键点做 Gaussian smoothing；这一步**会改变图像重投影**，而仅修改 `T_{WC}` 不会。

对双臂真实操作，**两个手可能本来就同时朝一个方向加速**，算法可能错误地把共同手运动当相机抖动。特别是工人一起搬起重物时，应记录原始 `poses_raw`、稳定化后的 `poses`、公共/相对运动指标，和真实桌面静止标志物对照，而不是盲目采用优化后曲线。

### 5.1 官方 NPZ 世界系精化的隐性风险

[官方文档](https://nvidia.github.io/IsaacCapture/main/ego_post_processing/egohand.html#world-coordinates-and-camera-pose-refinement) 明确：`pose.npz` 支持 `data` [F,4,4] + `inds`；稳定化脚本**只按位置配对并检查左右手与 pose 的帧数相等**，不会自动依据 `frame_names` 对齐。若左右手各漏掉不同一帧，却数量恰好相等，直接运行就可能产生**无报错的错误世界轨迹**。工业管线必须先做“源帧 ID 精确内连接”并统计丢帧比例，无法匹配的时刻不能伪造完整轨迹。

正确的诊断步骤是：先读 `frame_names` 得到原视频帧号，再由 `fps` 与时钟信息恢复曝光时间，匹配相机位姿 `inds` 或其时间戳；检查每个区间的外参版本、帧单调性和米制尺度；最后才将配对后的同一时刻两只手交给世界系精化脚本。**不可通过复制最后一个手姿态来让矩阵长度凑齐。**

## 6. 后处理与 QC：漂亮的 Overlay 并不等于可训练

**【代码】**官方 [`adaptive_smoothing.py`](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture_egopost/inference/postprocessing/adaptive_smoothing.py) 默认开启，针对 SOMA rotation 在 SO(3) 上、相机平移和形状参数分开处理，然后重建 `joints/vertices`，不会对网格点直接粗暴滑动平均。采用物理时间间隔，并在非有限值、长间断处分段。

**【代码】** [`clip_quality_control.py`](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture_egopost/inference/postprocessing/clip_quality_control.py) 检查过大速度/加速度、手根越界、手的尺寸突变、左右手 chirality、双手异常近距离、双手尺寸差、覆盖帧数和无效值。命令行 `--postprocessing_qc --qc_max_bad_frac 0.02` 允许一定比例标记帧；质量报告**只给 keep/drop 决策，不删除原始预测**。

生产应用建议把自动 QC 拆成四组，不以总通过率取代单项：

| QC 类别 | 可自动计算 | 无法凭单目自动确认 |
| --- | --- | --- |
| 输入有效性 | 图像/框/内参/时钟与 FPS、左右手帧覆盖率 | 绝对真实手型 |
| 图像空间 | 重投影 EPE、检测框越界、模糊/遮挡比例 | 工件背面手指的真实接触 |
| 几何运动 | `R^TR≈I`、`det R≈1`、深度正值、尺寸稳定、速度/加速度 p95 | 人和物体之间真实作用力 |
| 时序世界系 | `T_WC` 方向、源帧 ID 内连接、外参版本、SLAM 轨迹抖动 | 未定位的单目尺度真实值 |

**特别注意单手视频**：官方 `--postprocessing_qc` 要求 `--hand both`，且两手覆盖一致是交付规则之一；真正只有一只手的片段可能被标作 drop，这不一定意味着算法失败。数据管线要先把单手、双手、遮挡、高速操作分层评测，不能用默认两手交付标准筛掉所有单手有效技能。

## 7. 如何复现：从官方示例走到自己的 EGO 数据

### 7.1 官方软件与模型依赖（不是本站已执行的命令）

[环境要求](https://nvidia.github.io/IsaacCapture/main/ego_post_processing/egohand.html#system-requirements)：Ubuntu 24.04、Python 3.12、**32 GB 系统内存、25 GB 磁盘**，推荐 GPU 包括 RTX 6000 Ada/L40/L20/RTX PRO 6000 Blackwell。数据较长时考虑 `--frame_batch_size`。官方 [requirements.txt](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture_egopost/requirements.txt) 固定如 OpenCV 4.11、NumPy 2.4.4、ONNX Runtime GPU 1.28；安装 DINOv3 另有单独的 `requirements-dinov3.txt`。环境严格按官方当前依赖，并记录 Git SHA。

```bash
git clone https://github.com/NVIDIA/IsaacCapture.git
cd IsaacCapture/src/python/isaaccapture_egopost
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -r requirements-dinov3.txt
.venv/bin/python -c "from dinov3.models.vision_transformer import DinoVisionTransformer"
```

准备 `intrinsics.json`：严格填原始视频宽高、镜头模型、焦距、主点与畸变参数。不要把手机/相机 EXIF 的等效 35 mm 焦距直接当成像素焦距 `f_x`。鱼眼推荐 `--perspective_cropping`；它根据手框生成**虚拟针孔视角**，输出结果再转换回原相机坐标。

### 7.2 最小 GPU 推理指令

```bash
PYOPENGL_PLATFORM=egl PYTHONPATH="$(pwd)" .venv/bin/python \
  inference/scripts/run_egohand_inference_pipeline.py \
  --input /path/to/ego_clip.mp4 \
  --intrinsics_json /path/to/intrinsics.json \
  --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \
  --perspective_cropping --draw_skeleton --no_mesh \
  --dtype bfloat16 \
  --postprocessing_qc --qc_max_bad_frac 0.02 \
  --output_dir outputs/egohand_ego
```

官方默认从 Hugging Face 下载 `nvidia/EgoHand-v1` 权重以及 SOMA-X 模型资产；RT-DETR 手框来自另一个可选权重 `nvidia/RT-DETR-Hand-Detector-v1`。受控网络可先用 `hf download` 下载，再通过 `--egohand_ckpt`、`--soma_hand_assets` 和检测器 YAML 的本地 `weights` 路径离线使用。查阅各组件的**实际 LICENSE/模型卡与依赖数据许可**，不能只根据主论文中的“允许商用”概括全部组成资产。

如果自己有 2D 手框，不必使用 RT-DETR。将原始视频同名目录内左右手 JSON 保存为 `000000: [center_x,center_y,side]`；设置 `BBOX_DIR` 后使用 `inference/hand_detector_2d/configs/saved_bboxes.yaml`。手框使用**解剖左右手**，原始未预先拉直的帧坐标；视频旋转/resize 时务必变换框并配套修正 K。

### 7.3 从模型输出到世界系

```bash
PYTHONPATH="$(pwd)" .venv/bin/python \
  inference/postprocessing/world_coordinate_projection.py \
  --left outputs/egohand_ego/ego_clip/left_results.npz \
  --right outputs/egohand_ego/ego_clip/right_results.npz \
  --pose /path/to/pose_aligned.npz \
  --out outputs/egohand_ego/ego_clip/pose_stabilized.npz \
  --smooth_post 0
```

**运行前必须先自己对齐 `frame_names` 和 `inds`，双手缺帧时不可直接运行。** `--smooth_post 0` 是为了让首轮验证保持关键点像素重投影严格不变，再检查是否需要额外平滑。

### 7.4 可在 CI 执行的 CPU 契约（与模型精度严格分开）

本仓库提供 [`examples/egohand_camera_frame_contract_cpu.py`](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egohand_camera_frame_contract_cpu.py)，使用 Python 标准库测试：

- 把手根/腕相对关节正确放进相机系，验证 `T_WC` 正反方向；
- 推导并恢复固定深度下的 2D→侧向平移，构造颠倒变换的负例；
- 检查焦距比例改变会改变导出的深度，不能用自洽 Overlay 当米制验证；
- 用故意不一致的左右手帧号证明**行数相同不等于帧对齐**；
- 验证含间断帧序列不能错误平滑成一条“连续轨迹”。

它只验证**几何和数据处理假设**，没有 DINOv3、RT-DETR、SOMA 前向，因此不构成 GPU 推理或复现实验。

## 8. 性能与对比：读表比读新闻稿重要

【论文】在**未进入作者训练集**的 H2O 上使用 GT 手框、30 FPS、256×256 Crop 评估；EgoHand 60k step + smoothing 的**均值 MPJPE 13.12 mm、Root Translation Error 25.63 mm、加速度误差 1.90 mm/frame²**。同一对照表中，HaMeR 的 MPJPE 19.35 mm、HaWoR 17.78 mm、WiLoR 17.77 mm；但 **WiLoR 的 PA-MPJPE 与 2D EPE 更低**。所以正确结论是**部分 3D/时序指标明显改善，不是所有指标全面第一**。

同一篇论文 HOT3D/ARCTIC 的基线筛选不同：由于 WiLoR/HaWoR 训练数据涉及这些数据集，作者将其排除对应比较；HaWoR/Dyn-HaMR 获得 GT camera pose，且对 Dyn-HaMR 的 HMP 后处理并未成功运行。**不能将这份表格外推为“实际任意工业采集系统里领先全部端到端方法”**。

论文不同位置/版本存在 **约 9.0 ms 与 9.69 ms** 的模型推理数字，数值依 GPU 和设置变化。以官方实测配置为准，生产吞吐应另外报告：

$$
T_{\rm end-to-end}=
T_{\rm decode}+T_{\rm detector}+T_{\rm crop}
+T_{\rm EgoHand}+T_{\rm post}+T_{\rm encode}+T_{\rm queue}.
$$

不要把这些阶段简单相加为“可并行流水线稳态延迟”：若有异步流水线，还应分别报**每阶段单批耗时、吞吐、排队 p95、峰值显存和输出覆盖率**。10 万小时级数据规模更关心每小时原视频成本和有效样本数量，而不是孤立的纯模型 FPS。

## 9. 工业 EGO / π₀.₅ 项目如何做真正有说服力的验证？

在数据标签中建议增加 `source_video_id`、`frame_id`、`exposure_ns`、`camera_model`、`K_version`、`T_WC_version`、`handedness`、`detector_bbox`、`joint_3d_camera_m`、`joint_3d_world_m`、`validity_mask`、`QC_reason`，保留原始与平滑后结果。数据格式可以落在 MCAP/Parquet/LeRobot 扩展键，但**要把输出动作语义和机器人 Action Chunk 分开**（参见 [MCAP→LeRobot v3 契约](../../02-data-engineering/mcap-lerobot-v3-temporal-action-contract/)）。

**【实验方案】最小四组对照**：A：HaMeR + 同一检测器/同一内参；B：EgoHand 原始单帧；C：EgoHand + Adaptive smoothing；D：C + 可靠 SLAM 世界系。对每组都用相同的独立标定条件和可见手框协议，按无遮挡、严重遮挡、快速旋转、鱼眼边缘、双手交叉五类统计 MPJPE（有 3D GT 时）、2D EPE、腕部深度误差、抖动、缺帧率、跨帧形状漂移。再加“故意使用错误焦距 / 错误 `T_WC` 方向 / 左右手错位”的负例，确保质检真的能抓住错误。

工业夹爪训练应使用手-物距离、抓取意图、相对物体位姿、任务阶段与成功失败等**间接监督**；不能把 25×3×3 SOMA 旋转矩阵直接作为机器人 7DoF 或双臂 14DoF 动作监督。单目缺少接触力、物体惯性和机器人可达性。要把人类动作变成机器人可执行动作，仍需物体跟踪、目标 TCP、坐标标定、重定向/可达性约束与机器人侧少样本标注。

### 推荐顺序与停止准则

1. **第一天：接口**。拿 10 段已知内参 EGO 视频走通官方样例；逐段确认左右手 `frame_names`、SOMA shape、Camera Head、Overlay 与 `K`。
2. **第二天：精度**。在 2–3 个固定已知尺寸物体/动作上测真实深度与末端误差；有真值优先用真值，没真值只报告投影/一致性，**绝不写“毫米级 3D 精度”**。
3. **第三天：批量**。引入外部 SLAM/时间对齐，按帧 ID join；抽检失败样本，比较双手 SLAM 精化前后静止场景上的世界漂移。
4. **第四天：训练价值**。仅把 QC 合格片段作为增强监督；对 π₀.₅ LoRA 训练做“无手特征 / 只用 2D / 加 3D 相对物体特征”的同数据同算力对照，并用新工件成功率、接管率评估收益。

**停止/隔离规则【实验建议】**：外参或时钟未知、真实焦距缺失、尺度未定义、source frame 无法匹配、左右手身份失配的片段不进入有米制意义的动作监督。优先保留原片和失败原因，而不是猜测、插值、平滑后掩盖原始错误。

## 10. 论文与源码查阅路线（按因果顺序）

| 先读什么 | 重点核查的源码/事实 |
| --- | --- |
| [论文 `3](https://arxiv.org/abs/2610.30293) | DINOv3 + RayPE + CLIFF；SOMA 与 2D 两分支；平移求解与损失 |
| [官方接口指南](https://nvidia.github.io/IsaacCapture/main/ego_post_processing/egohand.html) | 系统要求、K、手框 JSON、输入视频、输出数组、CPU/GPU 边界 |
| [`run_egohand_inference_pipeline.py`](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture_egopost/inference/scripts/run_egohand_inference_pipeline.py) | `--perspective_cropping`、`--intrinsics_json`、检测器与输出 |
| [`adaptive_smoothing.py`](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture_egopost/inference/postprocessing/adaptive_smoothing.py) | SO(3) 旋转平滑、时间间隙、重建顶点 |
| [`world_coordinate_projection.py`](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture_egopost/inference/postprocessing/world_coordinate_projection.py) | 双手共同运动、Kabsch 轨迹修正、缺帧错位风险 |
| [`clip_quality_control.py`](https://github.com/NVIDIA/IsaacCapture/blob/main/src/python/isaaccapture_egopost/inference/postprocessing/clip_quality_control.py) | 速度/几何/左右手/覆盖率统计与 drop 语义 |
| [NVIDIA GEM-X](https://github.com/NVlabs/GEM-X) | 预训练/微调方案与发布内容，和 EgoHand 推理仓库区分 |

**结论：** 对大规模 EGO 视频，EgoHand 是值得优先运行的**相机系手部伪标签候选**，尤其适合与现有相机标定、SLAM、数据质检和 LeRobot 数据管线衔接。最值得做的原创研究不是再包一层网格可视化，而是**内参/时钟/SLAM 联合误差如何传播到人机动作监督，以及如何用可靠性权重抑制伪标签误导 π₀.₅**。本文尚未完成真实 GPU 复现或训练效果验证。
