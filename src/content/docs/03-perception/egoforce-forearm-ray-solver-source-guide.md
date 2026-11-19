---
title: "EgoForce 源码级教程：前臂引导的多相机射线定位和工业 EGO 数据合同"
description: "SIGGRAPH 2026：HALO、FARM、Crop Intrinsics Tokens、RSS 加权射线闭式解、鲁棒性与退化、世界坐标接入、开源边界、双图和 CPU 复现"
sidebar:
  order: 3
---

> **2026-11-19 源码专题。** 核查 [EgoForce 论文（2026-05-12）](https://arxiv.org/abs/2605.12498)、[DFKI 项目](https://dfki-av.github.io/EgoForce/)、[官方代码](https://github.com/dfki-av/EgoForce)，重点检查 HALO、FARM、RSS、Demo 和公开评测。文内【论文】为作者实验，【源码】为公开实现，【推导】为本站独立数学解释，【建议】仅是复现方案；本站完成合成 CPU 合同，不等于完成 GPU 推理、真实精度复核或工业策略训练。

# 从 2D EGO 手部图像到相机系 3D：为何前臂是关键？

在 [EgoHand](./egohand-dinov3-soma-camera-qc/) 一文中，视觉特征、手部形态先验和 2D 几何分支共同约束单帧手的位置。EgoForce 走另一条值得对照的路线：**手部和前臂分别 Crop，HALO 联合预测，使用真实相机模型的逆投影光线和局部 3D 骨架，在射线空间闭式求解相机平移**。这对存在大视角、镜头畸变和腕部遮挡的 EGO 数据很有价值，但**相机系 3D 不是世界系 3D，更不代表机器人可执行动作**。

![EgoForce 采集、网络、射线定位及数据治理的原创链路图](/Embodied-AI-Roadmap/diagrams/egoforce-system.svg)

*图 1｜绿色、橙色为 EgoForce 模型与射线定位；下方虚线范围是工业数据管线的额外工程，不应算作论文已有能力。*

## 1. 开源资产和复现边界

| 资产 | 官方可核对内容 | 本站审计结论 |
| --- | --- | --- |
| 论文 | [arXiv 2605.12498](https://arxiv.org/abs/2605.12498)，SIGGRAPH 2026 | 作者称 HOT3D 相机系关节误差较已有方法最高下降 28%；不等于本站实测 |
| 模型 | [models/halo.py](https://github.com/dfki-av/EgoForce/blob/main/models/halo.py)、[models/arm_model.py](https://github.com/dfki-av/EgoForce/blob/main/models/arm_model.py) | HALO、FARM 相关实现公开，包含关闭 CIT/Arm 的消融类 |
| 几何 | [core/rss.py](https://github.com/dfki-av/EgoForce/blob/main/core/rss.py) | 确有多模型逆投影、3D ray-space 线性求解和可选鲁棒流程 |
| 推理 | [demo/inference.py](https://github.com/dfki-av/EgoForce/blob/main/demo/inference.py) | HALO → LimbModel → Camera-Space Mesh → 可选 Kalman → 重投影 |
| 权重 | [下载脚本](https://github.com/dfki-av/EgoForce/blob/main/scripts/download_model_weights.sh) | README 给出 HF 下载和本地目录；本站未下载并加载验证 |
| 数据与评测 | [experiments](https://github.com/dfki-av/EgoForce/tree/main/experiments)、[数据脚本](https://github.com/dfki-av/EgoForce/blob/main/scripts/download_datasets.sh) | 已公开预测、评测、相机扰动、前臂遮挡与尺度分析路径 |
| 从零训练 | 仓库可读的现有文件结构 | 未找到完整一键训练入口，不声称权重可从头复训 |
| 许可 | [官方 README](https://github.com/dfki-av/EgoForce) | **CC BY-NC 4.0，含预训练权重；商业工业应用必须先确认授权** |

上游代码较早已公开，本篇是 **2026-11-19 新增的源码级数学/工程教学**，不是声称论文刚发布。不要把模型可推理、作者报告准确率、本站有 CPU 脚本这三件事混为一谈。

## 2. 输入和输出 ABI：数据形状与坐标

| 模块 | 输入 | 输出 | 易错点 |
| --- | --- | --- | --- |
| 数据采集 | 头戴 RGB、曝光时间戳、镜头类型及已标定内参 | 原始 H×W×3 帧 | 将视频帧编号代替时间戳 |
| Crop | 左/右手框、各自前臂框 | 手 Crop（论文图示 224²）、臂 Crop（112²） | 二者独立缩放，不能共享 bbox |
| CIT + HALO | Crop、相机及 Crop Intrinsics Token | 手/臂 2D joints、置信度、局部 3D 参数 | Crop 位置丢失则全图方向错误 |
| FARM + MANO | 手和前臂姿态、形态 | 相对于根的 3D joint/mesh | 骨架索引和尺度不能自行重定义 |
| RSS | 各关节单位相机射线、局部 3D 关节、置信度 | 相机平移（米）与相机系 hand mesh | 平行光线、畸变、错误点导致假深度 |
| 工业数据管线（扩展） | RSS + 外部 SLAM/VIO + 时间同步 | 世界系双手轨迹与有效性标记 | 不是 EgoForce 原生 VLA 动作接口 |

【源码】在 [unproject_unit_rays](https://github.com/dfki-av/EgoForce/blob/main/core/rss.py) 中，先利用 hand_bbox/hand_crop_size 和 arm_bbox/arm_crop_size 分别将网络的 2D 输出映射回原图，再按 Camera Type 反投影；手部取前 21 个 2D 点，余下属于前臂。相机模型枚举在此源码为 pinhole=0、rational=2、fisheye624=3、KannalaBrandt K3=4、equisolid=5、equirectangular=6、stereographic=7。不能把鱼眼模型强行假设为针孔模型。

【推导】令手 Crop 的某个二维点为 \(\mathbf p_i^h\)，手框左上角为 \(\mathbf b_h\)，原图手框尺寸为 \(\mathbf s_h\)，网络 Crop 规格为 \(\mathbf s_{\rm in}\)：

\[
\mathbf p_i^{raw}=\mathbf b_h+\mathbf p_i^h\odot\mathbf s_h/\mathbf s_{\rm in}.
\]

对于零畸变的针孔相机：

\[
\widetilde{\mathbf d}_i=K^{-1}[u_i,v_i,1]^\top,\qquad
\mathbf d_i=\frac{\widetilde{\mathbf d}_i}{\|\widetilde{\mathbf d}_i\|_2}.
\]

但鱼眼应调用对应相机模型的 \(\pi_{\mathcal C}^{-1}\)，否则广角边缘产生系统角误差。相机标定版本与 Crop 缩放必须成为数据资产的一部分。

## 3. HALO 与 FARM：为什么前臂先验有帮助？

【论文】FARM（Forearm Representation Model）用可微的前臂几何代理表示肘、前臂中点、腕及形状；HALO 同时读取手和前臂的视觉特征。与只预测手指的局部手型相比，前臂提供更长的相对尺度及关节方向约束，在部分遮挡和复杂相机角度下可能有益。

【源码】[models/halo.py](https://github.com/dfki-av/EgoForce/blob/main/models/halo.py) 中包含手臂分支特征、CIT 和手臂先验相关开关，HALOAblations 提供 no-cit、no-arm-prior、no-arm-input 对照的支撑。手/前臂不可见时模型使用 missing-arm token 与手条件化变分先验。**先验产生的是合理预测，不是被实际观测到的手臂真值**；不可见帧要携带质量标记。

【推导】根相对骨架的相机定位可写为：

\[
\mathbf X_{C,i}=\mathbf J_i(\theta,\beta)+\mathbf t_C.
\]

即使局部网格像一只自然的手，也不能从此推出 \(\mathbf t_C\) 的绝对误差很小。形态尺度、标定和可观测关节共同决定深度是否可信。

## 4. RSS：把射线法变成可执行的线性代数

【源码】[core/rss.py](https://github.com/dfki-av/EgoForce/blob/main/core/rss.py) 明确包含 _proj_from_dirs、_rdls_solve 和 solve_translation_full_ray_robust。对每一个根相对 3D 关节 \(\mathbf J_i\)、相机单位射线 \(\mathbf d_i\) 和可靠性权重 \(w_i\)，物理上希望：

\[
\mathbf J_i+\mathbf t\parallel \mathbf d_i.
\]

定义投影到垂直视线平面的矩阵：

\[
P_i=I-\mathbf d_i\mathbf d_i^\top,\quad P_i^\top=P_i,\quad P_i^2=P_i.
\]

因此点到射线的垂距平方为：

\[
\mathcal L(\mathbf t)=\sum_{i=1}^N w_i\|P_i(\mathbf J_i+\mathbf t)\|_2^2.
\]

求导令零，得到：

\[
M\mathbf t=-\mathbf m,\quad
M=\sum_iw_iP_i,\quad \mathbf m=\sum_iw_iP_i\mathbf J_i.
\]

再使用数值稳定的线性求解而不是手工矩阵求逆：

\[
\boxed{\mathbf t^*=\operatorname{solve}(M+\varepsilon I,-\mathbf m).}
\]

这里是对源码线性系统的【推导】。完整模型还有手与臂两阶段 refinement、可靠性选点、可选 IRLS/PROSAC、二次平移限幅等逻辑，不能把上述公式错误声称为整套最终推理的全部。

![RSS 的光线约束、平行退化、错误内参与异常关节](/Embodied-AI-Roadmap/diagrams/egoforce-ray-detail.svg)

*图 2｜原创建模图：左侧示意垂直于射线的三维残差，不是像素误差；右侧展示三类工业采集失败。*

### 4.1 不能只看模型能否算出一个数字

当所有射线近乎平行时：

\[
M=\left(\sum_iw_i\right)(I-\mathbf d\mathbf d^\top).
\]

该矩阵在共同射线方向的特征值为零，\(\operatorname{rank}(M)=2\)，因此视线深度不可观测。阻尼项 \(\varepsilon I\) 可以让求解器吐出数值，却**不能创造信息**。需要记录：

\[
\kappa(M)=\sigma_{\max}(M)/\sigma_{\min}(M).
\]

建议对条件数、正深度、离群点比例、有效手/臂点数、相机变更与可见时长建立拒绝规则。每段数据保存实际阈值而非事后只保存成功标签。

### 4.2 置信度和鲁棒估计不是免费的真实性保证

源码还包含 leverage 加权、Tukey/Huber IRLS、可选 PROSAC 和条件数检查；**函数有开关不意味着默认全部打开**。一个遮挡的指尖如果预测到错误位置，射线可能十分离谱。IRLS 下示例 Huber 权重：

\[
w_i^{eff}=w_i\min(1,c/\|P_i(J_i+t)\|_2).
\]

**值得警惕的源码细节**：_rdls_residuals 算的是点到三维射线的长度，_robust_weights_scalar 的注释却出现“pixel units”。移植到自己的数据管线时，务必查单位和阈值，不能把常数 4.685 直接当成 4.685 像素或毫米。应先在有米制参考的采样上做阈值标定。

## 5. 世界坐标不是网络自动输出

【建议】若有外部同步 SLAM/VIO 的相机世界位姿，才可将相机系结果变换到世界系：

\[
T_{WC}(t)=\begin{bmatrix}R_{WC}(t)&t_{WC}(t)\\0&1\end{bmatrix},
\quad X_W(t)=R_{WC}(t)X_C(t)+t_{WC}(t).
\]

需要额外区分：camera-to-world 还是 world-to-camera、轨迹是否有尺度、SLAM 重定位是否跳变、头戴相机与采集手柄/工件的外参、视频与定位的时钟漂移。世界坐标下的手腕或指尖轨迹也**不能直接当作机器人夹爪控制命令**：要加入本体动作接口、工具接触点、工件位姿、夹爪状态、可达性验证与失效数据过滤。

建议的帧级数据合同包括 source_clip_id、frame_id、exposure_timestamp_ns、hand_side、raw_hand_bbox、raw_arm_bbox、camera_model_id、intrinsic_hash、root_camera_m、joint_camera_m、valid_joint_mask、ray_residual_m、rss_condition、camera_pose_valid、world_map_id、pseudo_label_version。保持 invalid 和 missing；**不使用零位姿冒充有效手的位置**。

## 6. 路线复现：从官方脚本到自己的工业视频

【官方命令，本站未执行】：

~~~bash
git clone https://github.com/dfki-av/EgoForce.git
cd EgoForce
conda create -n egoforce python=3.10 -y
conda activate egoforce
bash scripts/install.sh
bash scripts/download_model_weights.sh
bash scripts/download_datasets.sh --data-root /path/to/datasets
# 在 settings.py 设置 DATASET.DIR
python experiments/save_predictions.py --test-dataset-name ARCTIC --checkpoint-path _DATA/model_weights.pth
python experiments/evaluate_predictions.py --test-dataset-name ARCTIC
python experiments/save_predictions.py --test-dataset-name ARCTIC --checkpoint-path _DATA/model_weights.pth --no-arm-input
python experiments/evaluate_predictions.py --test-dataset-name ARCTIC --no-arm-input
~~~

该仓库安装脚本涉及 CUDA 12.6、PyTorch 2.8、TensorRT、MMCV、AnyCalib、PyTorch3D、Project Aria；权重和数据另有下载。先验证版本、许可证、下载是否完成与缓存文件路径，**不能因 README 有命令就宣称已经完整部署**。

【CPU 合同】本站提供：[egoforce_ray_contract_cpu.py](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/egoforce_ray_contract_cpu.py)。不安装 PyTorch 即可做正反例：

~~~bash
python3 examples/egoforce_ray_contract_cpu.py
~~~

其最小实验覆盖已知相机光线/平移回归、低置信异常指尖屏蔽、平行射线秩退化、错误内参负例、相机/世界刚体变换可逆性。**脚本是独立合成数学验证，不是调用 EgoForce 神经网络。**

| 消融 | 官方入口与控制变量 | 推荐观察 | 有效结论范围 |
| --- | --- | --- | --- |
| 原始链路 | save_predictions → evaluate_predictions | CS-MPJPE、RR-MPJPE、有效率 | 基准精度，不等于工业效果 |
| 无前臂输入 | --no-arm-input | 遮挡区间精度、coverage | 对前臂视觉信息的依赖 |
| 无 CIT | --no-cit | 不同 FOV、图像边缘误差 | 相机几何条件编码增益 |
| 内参扰动 | save_noisy_intrinsic_predictions.py | 射线角误差、深度误差 | 特定相机噪声的鲁棒性 |
| 无 Kalman | evaluate_predictions.py --disable-kalman-filter | 加速度误差、接触延迟 | 平滑与响应之间的取舍 |
| 工厂采集样本 | 已知尺寸工件 + 独立参考跟踪 | 毫米误差 p95、漂移、漏帧 | 才能检验自有 EGO 数据是否可训练 |

## 7. 四个容易产生虚假结论的指标问题

**相机系误差和根相对误差是不同任务。** 若真值为 \(X_{C,i}\)，则：

\[
\operatorname{CS\!-\!MPJPE}=\frac1N\sum_i\|\widehat X_{C,i}-X_{C,i}\|_2.
\]

根相对误差会在求差前将每套骨架各自减去手腕位置，**自动消除了相机平移误差**，因此不能据 RR-MPJPE 优秀来宣称米制世界定位优秀。Procrustes/Sim(3) 对齐还能消除旋转/尺度误差，更不该用来判断工业动作绝对精度。

**平滑误差不等于执行误差。** Kalman 滤波抑制抖动，但可能在快速碰触时产生延迟，需要报两者。

**投影看起来对齐不等于 K 正确。** 使用同一个错误相机模型进行重建和重投影可能自洽，但真实 3D 坐标偏移。

**有重建模型不等于有动作标签。** 手腕、人手、工具、机器人末端是不同刚体；接触时刻与驱动空间需要单独建模与评测。

## 8. 与现有学习路线的关系和应用选择

| 方法 | 主要约束 | 本期最适合的任务 | 限制 |
| --- | --- | --- | --- |
| [HaMeR](./hamer/) | MANO + 单帧图像 | 人手局部 3D 基线 | 米制根定位薄弱 |
| [HaWoR](./hawor/) | 跨帧运动和相机轨迹 | 连续人手运动 | 时序漂移/尺度依赖 |
| [EgoHand](./egohand-dinov3-soma-camera-qc/) | DINOv3 + SOMA + 2D 几何 | 细粒度手指与 25 关节 | 与 MANO ABI 不同 |
| **EgoForce** | 前臂 FARM + HALO + RSS | 广角头戴 EGO、相机系根定位 | 非商用许可证、退化与遮挡 |
| [MonoEgo](../../02-data-engineering/monoego-deep/) | 被动标记 + 单目 SLAM | 无主动手腕仪器的米制轨迹 | 标记物、可见性和标定 |
| [EGO/UMI 标定](../../02-data-engineering/ego-umi-spatiotemporal-calibration/) | 外参与时序标定 | 大规模数据质量治理 | 需要真实参考数据 |

推荐沿 **EgoHand → EgoForce RSS → MonoEgo → EGO/UMI 时空标定 → [MCAP/LeRobot v3](../../02-data-engineering/mcap-lerobot-v3-temporal-action-contract/)** 学习。工业数据工程优先保留真实世界参考、失效 Mask 和版本化标定。先测数据可信度，再讨论跨本体动作映射和 VLA 的成功率提升。

**阶段结论：** 本篇完成论文及官方代码核验、架构/方程/ABI 讲解和可执行的 CPU 数学负例；作者论文的 28% 为作者报告。本站**尚未执行 GPU 推理、真实 EGO 定位精度复测及工业策略训练**。
