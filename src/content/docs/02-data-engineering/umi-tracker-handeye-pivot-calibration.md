---
title: "UMI 双 Vive Tracker 外参与 TCP 标定：从 AX=XB 到 Pivot 最小二乘"
description: "OpenCV 手眼标定、OpenVR 位姿接口、TCP pivot、双设备坐标与时钟、可观测性及真实数据验收；附两张原理图与可执行 CPU 合成契约"
sidebar:
  order: 8
---

> **2026-11-05 核对；工程专题而非新论文。** 本文以 [OpenCV calib3d 官方 hand-eye 接口](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)、[OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)、[Valve OpenVR 位姿接口](https://github.com/ValveSoftware/openvr/wiki/IVRSystem%3A%3AGetDeviceToAbsoluteTrackingPose) 和 [TRTK Pivot Calibration](https://rwthmeditec.github.io/TRTK/html/class_t_r_t_k_1_1_pivot_calibration_two_step.html) 为一手依据。公式、管线与 CPU 合成测试是本站教学实现，**不等于已经标定实际 Vive Tracker、Franka、UMI 或真机 Replay**。
>
> **前置阅读**：[EGO/UMI 时空标定及 QC](./ego-umi-spatiotemporal-calibration/) 负责相机/IMU/Tag/PnP 和世界锚；本篇专门解决 **Tracker 固定到 UMI 后，怎样把 Lighthouse 的刚体位姿转成可执行的双手 TCP 动作，并做独立验收**。两者不要混为同一套标定。

# 1. 先区分：手眼外参与工具 TCP 不是同一个未知数

UMI 常见输出为 Vive Tracker 的 6DoF 刚体姿态，但机器人需要的是夹爪操作点的 6DoF 姿态：

$$
\boxed{\mathbf T_{WE}(t)=\mathbf T_{WL}\,\mathbf T_{LU}(t)\,\mathbf T_{UE}}.
$$

采用列向量左乘：$\mathbf T_{AB}$ 把 **B 系坐标变成 A 系**。其中 $L$ 是 SteamVR/Lighthouse 追踪世界、$U$ 是 Tracker/UMI 刚体系、$E$ 是夹爪 TCP、$W$ 是工业工位世界。固定的 $\mathbf T_{WL}$ 用于将追踪世界对齐桌面标记或机器人 base；固定的 $\mathbf T_{UE}$ 是从 Tracker 到 TCP 的工具标定。**两者不能用一个偏置代替。**

![UMI Tracker 到工位机器人动作的完整坐标链](/Embodied-AI-Roadmap/diagrams/umi-tracker-system.svg)

*图 1｜左、右 UMI 必须分别保存自己的工具外参；在同一 Lighthouse 原点下追踪，仍不代表两个 TCP 的外参相同。*

| 数据 | 典型来源 | 形状 | 作用 | 常见混淆 |
| --- | --- | --- | --- | --- |
| $\mathbf T_{LU}(t)$ | OpenVR Tracker | 4×4 / float64 | 运动刚体观测 | 把原点当工具尖 |
| $\mathbf T_{UE}$ | Pivot / CAD / 辅助摄像头 | 4×4 | 工具几何 | Pivot 只恢复平移 |
| $\mathbf T_{WL}$ | 桌面世界锚 + 追踪世界配准 | 4×4 | 工位落地 | 重置原点后沿用旧外参 |
| $g(t)$ | 夹爪编码器 | 标量 | 开合监督 | 与图像曝光错帧 |
| $m(t)$ | 校准健康度 | bool/quality | 决定能否用于训练 | 用高置信度掩盖系统误差 |

OpenVR 的 `GetDeviceToAbsoluteTrackingPose(origin,prediction,...)` 返回相对于指定 TrackingUniverseOrigin 的设备位姿；官方说明 **prediction=0** 表示当前调用时刻的估计，正值是预测将来的位姿。它既不是自动与相机曝光时间对齐的历史状态，也不保证直接具有精确的同源硬件时间戳。[Valve 官方接口](https://github.com/ValveSoftware/openvr/wiki/IVRSystem%3A%3AGetDeviceToAbsoluteTrackingPose)。采集程序必须写下所选 origin、预测参数、设备 index/serial、采样时间、置信度，并检测 SteamVR 重置和丢跟踪。

## 1.1 为什么有两类“标定”？

**A：Tracker→辅助相机手眼外参**。为了用视觉 AprilGrid 核查 Tracker 运动方向或将视觉轨迹接到 Lighthouse 系，可以将一台**标定期间刚性固定在 UMI/Tracker 支架上的相机** $C$，观察固定标定板 $M$，估计 $\mathbf T_{UC}$。这是 OpenCV 标准 eye-in-hand 配置。

**B：Tracker→工具尖位置（Pivot）**。将 UMI 工具的同一尖点放在固定凹窝中心，改变整个工具的朝向，估计点在 Tracker 局部系中的位置 $\mathbf p_U^E$。不需要辅助摄像头，**但不能从单个点恢复完整的 TCP 旋转轴**。完整抓取方向仍需 CAD、带方向的夹具或可观测的多点几何。

**重要反例**：如果 EGO 相机戴在人的头上，而 Vive Tracker 固定在手持 UMI，二者**不是刚体**。这时不能直接套用下面的 $AX=XB$ 求一个永远不变的 $\mathbf T_{UC}$；人的颈部、手臂使外参随时间变化。真正的 H2R 闭环要把两者各自定位后在世界系融合。

# 2. 标准手眼：为什么是 $AX=XB$ 而非拟合一个平均平移？

固定标定板 $M$；追踪世界记作 $L$；每个时刻记录 Tracker 位姿 $A_i=\mathbf T_{LU_i}$ 和标定相机的 PnP 位姿 $B_i=\mathbf T_{C_iM}$；相机固定于 Tracker 支架，待求 $X=\mathbf T_{UC}$。按变换链有：

$$
\boxed{A_i\,X\,B_i=Y,\quad Y=\mathbf T_{LM}\ \text{(常量)}}.
$$

在第二时刻 $j$ 同样满足 $A_jXB_j=Y$。消去未知常量 $Y$：

$$
A_j^{-1}A_iX=XB_jB_i^{-1}
\quad\Longleftrightarrow\quad
\boxed{A_{ji}X=XB_{ji}}.
$$

注意相对运动取逆的位置。代码中把 $A_i$ 与 $B_i$ **同时做逆**通常并不会“自动成为 eye-to-hand”，反而会改变求解问题。先做三个 pose 的手推演算，再跑十几组空间姿态。

![相对运动方程、旋转和平移的可观测性与 Pivot 求解](/Embodied-AI-Roadmap/diagrams/umi-tracker-solver-detail.svg)

*图 2｜上半部分是移动相机对固定板的 hand–eye；下半部分是固定工具尖的 pivot。两种实验的数据采集动作完全不同。*

### 2.1 旋转先求，平移再求

将相对运动写作 $A_{ji}=[R_A,t_A]$、$B_{ji}=[R_B,t_B]$、$X=[R_X,t_X]$，则：

$$
R_A R_X=R_X R_B,
\qquad
(R_A-I)t_X=R_Xt_B-t_A.
$$

第一式表示两个坐标系看到的是同一个物理旋转，只差固定基变换。若用 $\log:SO(3)\to\mathbb R^3$ 对非退化转动提取旋转向量 $\alpha,\beta$，则 $\alpha\simeq R_X\beta$；足够多的**非平行转轴**才能稳健估计 $R_X$。得到 $R_X$ 后，堆叠第二式做线性最小二乘即可求 $t_X$。这是教学推导，Park/Horaud/Tsai/Daniilidis 在数值细节上有所差别。

对平移观测矩阵 $H=[R_{A_1}-I;\cdots;R_{A_n}-I]$，若奇异值 $\sigma_{\min}(H)\approx0$，就表示某方向外参平移不可观测或极易受噪声影响。单一旋转轴、仅平移运动都不适合完整手眼标定。OpenCV 官方要求至少两次具有**不平行旋转轴**的运动，建议使用更多姿态，而非刚好 3 帧。

### 2.2 OpenCV 的方向逐个核对

调用接口：

~~~python
R_UC, t_UC = cv2.calibrateHandEye(
    R_gripper2base=R_LU_list, t_gripper2base=t_LU_list,
    R_target2cam=R_CM_list,  t_target2cam=t_CM_list,
    method=cv2.CALIB_HAND_EYE_PARK,
)
~~~

- `gripper2base` 实际输入为 $\mathbf T_{LU}$：将 Tracker/夹具局部坐标转为追踪世界。
- `target2cam` 实际输入为 $\mathbf T_{CM}$：将标定板坐标转为**移动相机**。PnP 输出正好是这个方向；不要再自动取逆。
- 返回 `cam2gripper` 为 $\mathbf T_{UC}$，**并不直接是** $\mathbf T_{UE}$。
- 源码依据：[OpenCV calib3d/calibrateHandEye](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。OpenCV 提供 Tsai、Park、Horaud、Andreff、Daniilidis 多种方法；多方法一致仅是交叉检查，不能替代独立地面真值。

校准后验证 $\hat Y_i=A_i\hat X B_i$ 应为常量。留出未参与求解的姿态，统计

$$
e_t(i)=\left\|\hat t_{Y_i}-\bar t_Y\right\|_2,
\qquad
e_R(i)=\left\|\operatorname{Log}(\bar R_Y^\top\hat R_{Y_i})\right\|_2.
$$

这里使用了一个固定的平均参考变换 $\bar Y$；旋转平均建议采用 SO(3) 几何均值。**仅统计优化用帧的残差会乐观偏差**；真实系统还应与不共用观测的第三方量具比对。

# 3. TCP Pivot：最适合手持 UMI 的低成本工具原点标定

若 UMI 上的 Tracker 与末端相对位置固定，夹爪尖始终放在同一工位物理点 $P$，针对每个改变了方向的采样姿态：

$$
\mathbf p_L^P=R_{LU_i}\,\mathbf p_U^E+t_{LU_i}.
$$

未知量有六个：局部 TCP 三维位置 $p_U^E$ 和固定世界点 $p_L^P$。整理：

$$
\underbrace{\begin{bmatrix}
R_{LU_1}&-I\\R_{LU_2}&-I\\\vdots&\vdots\\R_{LU_n}&-I
\end{bmatrix}}_{H\in\mathbb R^{3n\times6}}
\begin{bmatrix}p_U^E\\p_L^P\end{bmatrix}
=
\underbrace{\begin{bmatrix}-t_{LU_1}\\-t_{LU_2}\\\vdots\\-t_{LU_n}\end{bmatrix}}_{b}.
$$

用 SVD 解最小二乘 $x^*=H^+b$，不建议形成 $H^TH$ 后硬求逆，因为病态时条件数会被平方。若所有姿态旋转相同，$H$ 的秩不足六，**增加静止重复帧不能解决不可观测**；需改变滚转、俯仰与偏航角。

Pivot 得到的只有 $p_U^E$。$\mathbf T_{UE}$ 的旋转部分 $R_{UE}$ 仍需另设标定方法：例如用能限定前进方向和手指张合轴的几何夹具、多点测量或机械 CAD 结合实物验证。将 $R_{UE}=I$ 写入配置可以作为**明确的临时约定**，但绝不能声称已完成 6DoF 工具标定。

## 3.1 双手不能共享一个 TCP

左右手分别记录：

~~~yaml
calibration_id: umi-tools-2026-11-05
tracking_universe: Standing
prediction_seconds: 0.0
left:
  serial: "<左 Tracker 实际序列号>"
  T_UE: "<左工具完整4x4矩阵，旋转待独立验证>"
right:
  serial: "<右 Tracker 实际序列号>"
  T_UE: "<右工具完整4x4矩阵，旋转待独立验证>"
T_WL: "<工位世界相对于追踪世界的固定外参>"
valid_from: "<校准版本生效时间>"
~~~

**Tracker id 不是跨次启动必然稳定的 serial**；应用层应遍历设备并核对序列号、connected、pose valid、tracking result，不能按数组 1/2 永久认定左/右。更换 Tracker、夹爪、转接板、重新安装、撞击松动后需重新验外参。

# 4. 无法同时看到世界标记时，怎样对齐 Lighthouse 和工位？

OpenVR 的 `Standing`、`Seated` 等原点是**追踪系统的定义**，不等同于桌面世界。选定工位世界 $W$，用足够多空间位置的**共同物理点/刚体轨迹**估计 $\mathbf T_{WL}$。同一刚体两系点对应满足：

$$
\mathbf p_W=R_{WL}\mathbf p_L+t_{WL},
\quad R_{WL}\in SO(3).
$$

最小化 $\sum_i\|p_{W,i}-(Rp_{L,i}+t)\|_2^2$，采用刚性 Procrustes/Umeyama（**固定尺度为 1**，除非已经通过独立计量证实尺度错误）；点的空间分布不能共线。若两个坐标系存在不同手性（镜像）或 L 的轴向被错误解释，需先规范轴定义再强制 $\det R=+1$，不能任意拟合一个反射矩阵掩盖错误。

世界原点重新设定后的 $T_{WL}$ 版本必须跟随每段示范写入；跨天模型训练若把两个不同 origin 的世界动作拼接，数据会发生系统性偏移。对双手，最终转换为机器人 base：

$$
\boxed{T_{RE}^{(h)}(t)=T_{RW}T_{WL}T_{LU}^{(h)}(t)T_{UE}^{(h)}},\quad h\in\{L,R\}.
$$

其中 $T_{RW}$ 不能假设恒为单位阵；若机器人基座相对工位移动，必须重新估计或实时更新。

# 5. 时间配准：OpenVR 的“预测位姿”不是图像的真值

记录视频帧的**曝光中点** $t_c$、Tracker 查询时刻 $t_L$、编码器时刻 $t_g$，对设备时钟建立映射，最后在同一时刻插值位姿：

$$
t_c=a\,t_L+b;
\qquad
T_{LU}(t_c)={\rm Interp}_{SE(3)}(T_{LU,i},T_{LU,i+1}).
$$

时间戳跨时钟域时，必须记录硬件 clock ID 和接收 clock ID；$a$ 为相对时钟漂移，$b$ 为偏移。位姿平移线性插值，旋转用 SO(3) SLERP；不要对矩阵元素直接线性插值后不正交化，也不要对欧拉角跨 $\pm\pi$ 生硬线性插值。存在短时丢包可以设严格外推上限，并把缺失区间标记为 **invalid**，不能无条件平滑成“完美轨迹”。

例：工具速度 0.30 m/s，视频与 Tracker 错开 15 ms，即使设备空间完全准确，也可能产生约 $4.5$ mm 的平移偏差：

$$
e_{\rm sync}\approx v\Delta t=0.30\times 0.015=0.0045\ \mathrm m.
$$

再加上 TCP 杠杆臂 $\ell$ 与角速度 $\omega$，姿态错配引起的几何偏差一阶界约 $\ell|\omega\Delta t|$。两项相关、可部分抵消，不能把它们机械加成独立误差真值。

# 6. CPU 合成验证：验证接口方向，而非模拟“真实标定成功”

仓库提供 [`examples/umi_tracker_handeye_pivot_contract.py`](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/umi_tracker_handeye_pivot_contract.py)，使用 NumPy 与 OpenCV（安装了 OpenCV 的现有 CI 环境可直接运行）：

~~~bash
python3 examples/umi_tracker_handeye_pivot_contract.py
~~~

它构造**已知真值** $X=T_{UC}$、静止板 $Y=T_{LM}$、多组 Tracker 运动 $A_i$，按 $B_i=X^{-1}A_i^{-1}Y$ 合成相机观测；然后调用 `cv2.calibrateHandEye` 恢复 $X$。最后用固定点接触的多姿态数据解 Pivot，并故意提供单姿态“退化”数据检验不应通过。

必须看三种输出：`PASS handeye`、`PASS pivot`、`PASS degeneracy`。该脚本**没有**合成鱼眼像素、相机畸变、软结构变形、真实 SteamVR 噪声或人体动作；通过只证明数学接口契约和标定方向写对，**不报告真实设备毫米级精度**。

## 6.1 数据和张量字段：怎样落到 MCAP/LeRobot？

~~~text
episode_id / task_id / umi_device_serial / calibration_id
camera/exposure_mid_ns, rgb_left[H,W,3], rgb_right[H,W,3]
lighthouse/timestamp_ns, T_LU_left[4,4], T_LU_right[4,4], tracking_valid[2]
gripper/timestamp_ns, openness[2]
calibration/T_WL[4,4], T_UE_left[4,4], T_UE_right[4,4], T_RW[4,4]
derived/T_RE_left[T,4,4], T_RE_right[T,4,4]
derived/action_delta_left[T,H,6], action_delta_right[T,H,6], action_mask[T,H,2]
qc/sync_error_ms, qc/pose_closure_mm, qc/pose_closure_deg
~~~

上面是**建议 schema，不是 OpenPI 或 LeRobot 固有字段**。如果模型的动作空间是关节角/Delta Joint，就不能把 TCP 的 SE(3) 增量直接塞入其输出维度；需要 IK/重定向和动作空间一致性检查。姿态增量可选：

$$
\Delta T^{\rm body}_{t,k}=T_{RE}(t)^{-1}T_{RE}(t+k),
\quad
\xi_{t,k}=\log_{SE(3)}(\Delta T^{\rm body}_{t,k})\in\mathbb R^6.
$$

这里 $\xi$ 是 SE(3) 李代数向量（具体角-线顺序及角度单位必须在 schema 中固定），**不是**把矩阵的前三列直接展平成“6维姿态”；OpenPI 动作填充、归一化和 mask 仍按对应 robot policy 配置处理。

# 7. 工位半天复现：每一步都留证据

| 阶段 | 操作 | 验收证据 | 停止条件 |
| --- | --- | --- | --- |
| 0. 确认追踪 | 同时发现两枚 Tracker，固化 serial 与 TrackingUniverseOrigin | 录制两条有不同 serial 的 pose stream | 某 Tracker 丢跟踪/换 id 未重绑定 |
| 1. 验坐标轴 | 对单个 Tracker 沿三根实际尺子轴各走 10 cm | 对应轨迹方向和尺度 | 运动方向反向或尺度异常 |
| 2. 手眼辅助校准 | **临时刚性安装**相机到被跟踪支架，静止 AprilGrid，做多轴旋转 | $T_{UC}$、各方法结果、held-out closure | 相机与 UMI 无刚性连接 |
| 3. 左右 Pivot | 每只手工具尖固定于硬质凹窝，分别采十余种旋转 | $p_U^E$、SVD 奇异值、残差和独立复测 | 移动接触点、转轴退化 |
| 4. 工位对齐 | 量测不共线的共同参考点，估计 $T_{WL}$ | 物理距离误差、det(R)=1、重启复测 | 追踪原点重置未更新版本 |
| 5. 时序检查 | 快速摆动与夹爪开闭，同时采视频/Tracker/编码器 | 时钟域、曝光时间、配准残差 p95 | 仅有主机收包时间不能估曝光偏置 |
| 6. 冻结验证 | 左右工具各移 10 cm、原地转动并检查 TCP 轨迹 | 独立尺/运动捕捉真值、重定位一致性 | held-out 偏差超现场门限 |
| 7. 训练导出 | 每段固定校准版本、标出 invalid 动作 | SE(3) chunk、g、mask、图像对照 | 不可执行动作未经重定向 |

**门限如何订？** 建议先根据抓取任务的几何间隙、夹爪宽度、机器人闭环控制能力确定容忍误差。例如可用“目标工位允许的横向定位误差 / 2”作整体感知+控制预算的起点，并分别为标定、追踪、控制留份额；**不能直接把合成实验的 $10^{-6}$ 米误差当真实工业设备规格**。

# 8. 失败模式、局限与下一篇读什么？

| 观察 | 可能原因 | 最低成本的定位实验 |
| --- | --- | --- |
| 双 Tracker 坐标方向不一致 | 两设备来源/原点混用，或局部轴映射不同 | 对同一物理位移看 $L$ 系结果 |
| 追踪点轨迹准、夹爪尖误差大 | $T_{UE}$ 未标或支架弯曲 | 固定点 Pivot + 换角度保持尖端不动 |
| 手眼恢复随采集批次大幅变 | 缺少非平行转轴/相机支架松动 | SVD 退化检查、重复多轴采样 |
| 图像目标静止但 TCP 随头部移动 | 把非刚体 EGO 头相机硬当 UMI 手眼相机 | 去掉错误 $T_{UC}$，分别定位再融合 |
| 低速 Replay 好、高速抓取错 | 图像曝光与 Tracker/编码器错时钟 | 加速运动时扫描时差残差 |
| 跨天同任务坐标整体偏置 | Lighthouse 或世界锚原点重置 | 从 calibration_id 找回 $T_{WL}$ 版本 |
| 训练 loss 低但机器人手爪反向 | body/world 轴、Action Delta 与绝对值混淆 | 用单轴 10 cm 动作离线回放 |

**与论文研究的衔接：** 这一套质量门控能给 [HiFi-UMI](./hifi-umi-deep/)、[RGB-only EGO 到 π₀.₅](../../04-human-to-robot/ego-rgb-only-to-pi05/) 和 [HuRo 机器人化](../../04-human-to-robot/huro-robotization-to-vla/) 提供**可验证的位姿/动作标签来源**。但几何精确不等于机器人可执行；接触、速度、碰撞和动作重定向仍需独立闭环评测。

## 参考文献及证据分级

1. [OpenCV Hand–Eye Calibration 官方 API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)：`R_gripper2base`、`R_target2cam`、返回的 `R_cam2gripper`，以及最少两次非平行旋转运动要求。**一手接口依据**。
2. [OpenCV Perspective-n-Point 官方](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)：物体/板 → 相机的变换方向、平面多解。**一手接口依据**。
3. [ValveSoftware OpenVR Wiki：GetDeviceToAbsoluteTrackingPose](https://github.com/ValveSoftware/openvr/wiki/IVRSystem%3A%3AGetDeviceToAbsoluteTrackingPose)：追踪原点、prediction、pose valid。**一手接口依据**。
4. [TRTK Pivot Calibration](https://rwthmeditec.github.io/TRTK/html/class_t_r_t_k_1_1_pivot_calibration_two_step.html) 与 [ROS tool_point_calibration](https://github.com/Jmeyer1292/tool_point_calibration)：工具尖保持固定、改变方位估计 TCP。**算法/工程实例**。
5. [CPU 合成契约脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/umi_tracker_handeye_pivot_contract.py)：站内**可执行但不是真机测量**的例子。

**没有验证的事实不得扩写为成果**：真实 Vive Tracker 精度、SteamVR 多 Tracker 硬件时钟一致性、实际夹爪受力变形、特定机器人 Replay 成功率都不在本次 CPU 实验的覆盖范围；这些留待工位现场记录与统计。
