---
title: "EGO/UMI 双目鱼眼＋IMU＋AprilTag：从标定到世界系动作标签（源码和误差实证）"
description: "统一讲透 Kalibr、鱼眼 PnP、空间/时间外参、工具标定、误差预算、质量门控和面向 VLA 的 Action Chunk，附可运行的合成数据反例"
sidebar:
  order: 7
---

> **资料复核：2026-10-28。性质：对 Kalibr、OpenVINS、AprilTag 3、OpenCV 一手技术文档的交叉整理＋工程方案；不是某篇新论文的结果复现。** 本页的 Python 合成实验已在本地 OpenCV 4.13 环境实际执行，**真实设备相机/IMU 采集、真实标定与机器人闭环未执行**。文中的参考阈值是建议起步标准，均非论文承诺精度。
>
> **建议先读**：[HiFi-UMI](./hifi-umi-deep/) → [MonoEgo](./monoego-deep/) → 本文 → [RGB-only EGO 怎样训练 π₀.₅](../../04-human-to-robot/ego-rgb-only-to-pi05/)。前两篇讨论设备/论文路线，这一篇解决实际采集时**该标定什么、怎么求、怎么证明轨迹可信、怎样写成动作标签**。

# 一、今天要解决的真实问题

假定你有一套**头戴双目鱼眼＋IMU**，桌面固定了几个 AprilTag，手持 UMI 上也贴了一块 Tag；夹爪有磁编码器。采集平台需要输出：

- 某时刻头部/相机的世界系 6DoF 位姿；
- 同一时刻 UMI 工具（夹爪 TCP）的世界系位姿；
- 夹爪开度、未来 20～50 个采样时刻的相对运动；
- 给每个监督维度一个有效性标记，以供机器人 VLA 训练。

“能在画面里看到 Tag”还远远不够。你需要分别解决五种未知：**镜头成像参数、双目刚体外参、相机/IMU 的空间时间关系、UMI 标签到 TCP 的工具外参、世界系锚定**。只差其中一项，轨迹也可能看起来连贯但物理上是错的。

![世界锚、运动相机、手持 UMI、工具 TCP 的变换关系](/Embodied-AI-Roadmap/diagrams/ego-umi-calibration-chain.svg)

*图 1｜蓝色框是世界锚点与相机，绿色框是被测 UMI；PnP 返回坐标变换的方向不能猜。图中系数、帧名均在下面定义。*

# 二、统一帧定义：为什么必须先约定 T_AB？

采用**列向量左乘**约定：

$$
{}^{A}\mathbf p=\mathbf T_{AB}\,{}^{B}\mathbf p,\qquad
\mathbf T_{AB}=
\begin{bmatrix}\mathbf R_{AB}&\mathbf t_{AB}\\
\mathbf 0^\top&1\end{bmatrix}.
$$

这里 $\mathbf T_{AB}$ 的含义是“把 B 系坐标变换到 A 系”，不是文字中的“从 A 看 B”这种容易歧义的说法。

| 符号 | 含义 | 如何获得 |
| --- | --- | --- |
| W | 桌面固定 AprilGrid/Marker 定义的世界系 | 选定锚点原点与轴向 |
| C_L、C_R | 双目左/右鱼眼光学系 | 图像内参、双目外参 |
| I | 头戴 IMU 的惯性传感器系 | Kalibr CAM-IMU |
| U | 手持 UMI 标签系 | 已知标签几何，单帧 PnP |
| E | 夹爪工具 TCP（操作点）系 | UMI 刚体工具外参 |
| R | 机器人 base 或工位 base 系 | 世界→机器人基座标定 |

一个容易出错的事实：**OpenCV solvePnP 输出的旋转和平移把“标记/物体点”变换到“相机点”**。如果你将 W 固定在桌面标定板上，则 PnP 给出 $\mathbf T_{CW}$；相机世界位姿需要求逆：

$$
\boxed{\mathbf T_{WC}=(\mathbf T_{CW})^{-1}}.
$$

若相机同时看到固定世界锚 W 和手持标签 U，则

$$
\boxed{\mathbf T_{WE}
=\mathbf T_{WC}\,\mathbf T_{CU}\,\mathbf T_{UE}
=(\mathbf T_{CW})^{-1}\mathbf T_{CU}\mathbf T_{UE}.}
$$

**手算 sanity check**：设相机世界系平移 $(0,0,-0.80)$ m，旋转为单位阵，UMI 标签在相机前方 $(0.20,0.05,0.90)$ m，工具相对于 U 再向局部 z 前进 $0.10$ m。那么工具世界位置就是 $(0.20,0.05,0.20)$ m。若把 $\mathbf T_{CW}$ 当作 $\mathbf T_{WC}$，z 会出现米级偏差；而单张图上的 PnP 重投影损失仍可能很小。

## 2.1 双目分开看到世界锚与工具怎么办？

左眼看得到 W、右眼看得到 U。双目标定已知 $\mathbf T_{C_LC_R}$ 时：

$$
\mathbf T_{WC_R}=\mathbf T_{WC_L}\mathbf T_{C_LC_R},
\qquad
\mathbf T_{WE}=\mathbf T_{WC_R}\mathbf T_{C_RU}\mathbf T_{UE}.
$$

这里两幅图必须对应**同一曝光时刻**，或者先根据正确的时间模型插值到相同时间；不能只凭两个图像文件的序号恰好一致。头部高速转动时，左右眼即使相差几毫秒，几何转换也可能产生显著误差。

## 2.2 世界系还要转到机器人 base 吗？

**需要，除非世界系 W 本来就是机器人 base。** 有固定工位、世界锚和机器人 base 的外参时：

$$
\mathbf T_{RE}=\mathbf T_{RW}\mathbf T_{WE}.
$$

没有 $\mathbf T_{RW}$ 就不能把“桌面世界系动作”直接当成“机器人基座系动作”。对于 EGO 人类演示，**获得几何上的机器人基座坐标仍不意味着动作一定可执行**；后续还需 IK、碰撞、速度和接触约束。

# 三、鱼眼模型：为什么直接套普通 PnP 可能错误？

鱼眼图像像素位置不是标准小孔模型加零畸变。OpenCV fisheye 的等距投影式模型定义（省略非零 skew 情形的部分记法）：

$$
r=\sqrt{(X/Z)^2+(Y/Z)^2},\quad
\theta=\arctan(r),
$$

$$
\theta_d=\theta(1+k_1\theta^2+k_2\theta^4+k_3\theta^6+k_4\theta^8),
$$

$$
x'=\frac{\theta_d}{r}\frac XZ,\qquad
y'=\frac{\theta_d}{r}\frac YZ,\qquad
u=f_xx'+c_x,\quad v=f_yy'+c_y.
$$

当 $r\to0$ 时按连续极限处理，不能数值上直接除以 0。鱼眼参数包含 $\mathbf K$ 和 $\mathbf D=(k_1,\ldots,k_4)$。这**不等同**于传统 OpenCV pinhole 的 5 参数 radial-tangential 模型；Kalibr 还支持多种相机投影模型，包括 pinhole-equi、omni 等。**必须用标定时相同的模型解析参数。**

### 正确地用 AprilTag 角点求鱼眼 PnP

一种稳妥且可验证的实现顺序：

1. 在**原始鱼眼图像**检测 4 个标签角点。
2. 通过 <code>cv2.fisheye.undistortPoints</code> 将角点变为归一化相机平面坐标。
3. 使用单位内参、零畸变执行 <code>cv2.solvePnPGeneric(..., flags=cv2.SOLVEPNP_IPPE_SQUARE)</code>。
4. 遍历 IPPE 的候选位姿，在**原鱼眼成像模型**下重新计算像素重投影误差，并检查角点深度为正。
5. 多帧连续性、标记法向、双目观测或非共面多标签约束一起消歧；**不能以最小重投影误差就声称得到了唯一的真实姿态**。

OpenCV IPPE_SQUARE 要求按其规定的**四角顺序**提供物体角点。在标签局部系中、边长为 $s$ 时：

$$
\mathbf P_0=(-s/2,+s/2,0),\;
\mathbf P_1=(+s/2,+s/2,0),\;
\mathbf P_2=(+s/2,-s/2,0),\;
\mathbf P_3=(-s/2,-s/2,0).
$$

不要把 AprilTag 检测器的角点数组顺序直接当作满足此规则；先用可视化逐一确认索引与标签方向。

**AprilTag 官方特别强调**：tagSize 是**实际检测角点之间的边长**，不一定是整张标签纸外围的尺寸。假设真实边长 60 mm，误填 65 mm，就会把尺度误差传进 PnP 的平移估计。贴纸皱折、打印机自动缩放、玻璃反射同样可能破坏已知几何。

**原始证据**：[OpenCV fisheye](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html) · [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) · [AprilTag 3 官方](https://github.com/AprilRobotics/apriltag)。

# 四、两种板不要混为一谈：标定板与工位世界锚

| 用途 | 推荐 | 理由 | 是否在运行时出现 |
| --- | --- | --- | --- |
| 双目内参/外参标定 | AprilGrid 多标签板 | 多角度、多距离覆盖整个 FOV | 不要求 |
| 相机/IMU 标定 | AprilGrid + 动态多轴激励 | 同时估计 CAM-IMU 外参与时间偏移 | 不要求 |
| 工位世界系定义 | 已量测的刚性 AprilTag/AprilGrid 锚 | 每次采集可统一世界原点与尺度 | 需要可观测或可追溯 |
| UMI 工具追踪 | 带已知尺寸的 Tag/多面标记 | 估计标签姿态并处理遮挡 | 需要至少部分可见 |
| 独立精度验收 | CMM、光学动捕或机械量具/标靶 | 不和被测系统共用误差来源 | 专门验证时使用 |

AprilGrid 的 <code>tagSize</code> 表示每个标签边长（米），<code>tagSpacing</code> 通常表示标签间净间距与边长的比例。**不要把 0.3 当作 0.3 m**。检查打印后实测尺寸，并记录标定板版本、纸张是否拉伸、固定底板是否翘曲。

### 示例：Kalibr 标定板 YAML

~~~yaml
target_type: 'aprilgrid'
tagRows: 6
tagCols: 6
tagSize: 0.06
tagSpacing: 0.30
~~~

这是**自行制作板件的示意参数**，不是 OpenVINS 示例板的实测值。边长 60 mm、tagSpacing=0.30 表示相邻标签边到边约 18 mm；必须用现场打印件复核。

# 五、实际标定顺序：别一次把未知量全部丢进优化

![设备空间/时间校准、质量门控、动作监督的分层流水线](/Embodied-AI-Roadmap/diagrams/ego-umi-calibration-qc.svg)

*图 2｜训练样本的产生方式：先把空间和时间关系证实，再决定哪几帧能成为 VLA 动作真值。*

## 5.1 第一步：锁定硬件与元数据

先固定镜头焦距与对焦环，锁死相机分辨率、裁切、曝光模式，记录 IMU 采样频率、相机标称/实际 FPS、触发线与设备 clock domain。**更换镜头、改分辨率或对相机做未知缩放裁切后，原内参不能直接复用。**

建议按设备原始流保存：

~~~text
/camera/left/image_raw    + exposure_midpoint_ns + frame_id
/camera/right/image_raw   + exposure_midpoint_ns + frame_id
/imu                      + timestamp_ns + accel, gyro
/umi/gripper              + timestamp_ns + openness
/calibration              + version, K, D, T_LR, T_CI, time_offset
~~~

先确认设备时间戳单调、是否存在重复帧、左右帧差、露出延时和 IMU 丢包。只有主机“收到图像的时刻”不一定等于“传感器曝光时刻”。

## 5.2 第二步：标双目鱼眼 K、D、T_LR

Kalibr 的 [多相机标定官方 Wiki](https://github.com/ethz-asl/kalibr/wiki/multiple-camera-calibration) 给出：

~~~bash
kalibr_calibrate_cameras \
  --bag stereo_calib.bag \
  --topics /camera/left/image_raw /camera/right/image_raw \
  --models pinhole-equi pinhole-equi \
  --target april_6x6.yaml
~~~

上面仅适用于**与你的实际成像模型相容的 pinhole-equi** 的例子；极大 FOV、镜头畸变与投影适配性需用残差和外部测试决定，不能见“鱼眼”就认定必选 pinhole-equi。

多相机标定要覆盖左右共同观测区域、图像中心到边缘、多种倾角和距离。官方建议多相机标定时**相机刚性装好，移动标定板**采集，避免在标定过程中改变相机之间刚体关系。

输出文件通常包含 <code>camchain-*.yaml</code>、报告与每相机重投影统计。检查校准报告中是否存在某一侧边缘残差系统偏高、外参突变、两路相机高于预期的旋转差；分别留出另一段数据做复投影，不能只看训练标定时的最小 RMS。

## 5.3 第三步：标 IMU 噪声和 CAM-IMU 空间/时间外参

Kalibr 支持 IMU-Camera 的**空间变换**与**时间偏移**标定。命令结构如下：

~~~bash
kalibr_calibrate_imu_camera \
  --bag cam_imu_motion.bag \
  --cam camchain-stereo_calib.yaml \
  --imu imu.yaml \
  --target april_6x6.yaml \
  --time-calibration
~~~

<code>--time-calibration</code> 需要根据具体 Kalibr 版本帮助确认；官方经典 Wiki 记载它用于开启时间标定，不能把默认行为理解为必然估出时间偏移。

在 **cam-imu** 阶段要**使刚体相机+IMU 一起运动，标定板保持静止**，和上一步“动板不动相机”的多相机外参采集不是同一种操作。运动应包含多方向转动及平移，避免过度运动模糊；OpenVINS 建议约 20～30 Hz 图像、200～500 Hz IMU，30～60 秒多轴激励数据（是其示例经验，不保证所有设备适用）。

### 为什么不能只摇一摇设备？

对于纯匀速直线或缺乏旋转激励的数据，有的外参与时间参数在观测上高度相关，优化器可以给出数值解，却无法证实它唯一、稳定。OpenVINS 指出至少需要平移和两个独立旋转自由度，使所估参数具备可观测性；尽量减少运动模糊，又要充分激励全部轴。

检查输出：IMU 噪声参数、拟合残差、bias/3σ、估计外参与卷尺测值是否相符。**不要仅靠优化“converged”就接受标定。**

一手资料：[Kalibr 官方仓库](https://github.com/ethz-asl/kalibr) · [OpenVINS 官方校准指南](https://docs.openvins.com/gs-calibration.html)。

# 六、同步为什么要用曝光时刻，而不是图像收到时刻？

令相机曝光中点为 $t_c$，IMU 时刻为 $t_i$，定义统一时钟上的偏置：

$$
t_i=t_c+\delta_{ci}.
$$

这是**本文定义的符号方向**，不同软件配置可能相反，必须在接口记录中注明。对高速夹爪动作，如果图片与姿态相差 $\Delta t$，局部恒速近似下：

$$
\Vert\Delta\mathbf p\Vert\approx\Vert\mathbf v\Vert\vert\Delta t\vert,\qquad
\vert\Delta\theta\vert\approx\Vert\boldsymbol\omega\Vert\vert\Delta t\vert.
$$

**手算**：夹爪平移速度 0.4 m/s，错开 15 ms，位置就偏约 6 mm；角速度 120°/s 时，同样偏移导致约 1.8° 姿态错配。假设工具从标记中心到 TCP 有 0.12 m 杠杆臂，单纯因姿态错配造成的额外 TCP 位移一阶近似：

$$
\Vert\Delta \mathbf p_{tcp}\Vert
\lesssim L\Vert\boldsymbol\omega\Vert\vert\Delta t\vert
\approx 0.12\times(120\pi/180)\times0.015
\approx 3.8\text{ mm}.
$$

这项误差与平移误差可能同向也可能抵消，**不能简单加起来当作严格总误差**。此推导也不表示 IMU 误差可以靠某个固定 $\delta$ 完全补偿：主机收包延迟、传感器漂移和曝光滚动效应需要分别处理。

### 怎样定量测试时钟？

从一段包含快速重复运动的标定数据估计时间偏置，例如对视觉角速度 $\omega_v(t)$ 和 IMU 去偏角速度 $\omega_i(t)$ 做互相关粗估：

$$
\hat\delta=
\underset{\delta}{\arg\max}\ 
\int\boldsymbol\omega_v(t)^\top
\boldsymbol\omega_i(t+\delta)\,dt,
$$

随后在滑窗或连续时间批量优化中联合求外参、时间与轨迹。互相关只提供**初值**，高频高噪、转动轴混乱或持续等速运动会产生虚假峰值。

真实量产优先硬件共同触发或能确认时钟同步的链路。Kalibr 的软件时间外参是误差校准手段，不是“标完以后异步设备就自动变硬同步”。

# 七、固定 AprilTag 与 UMI Marker 的 PnP：为什么会有双解？

单个共面方形 Tag 的四个角点只能提供有限的透视约束。近正面、小目标或远距离时，两个不同的姿态候选可能都有极小重投影误差。IPPE 系列方法会返回多个候选，让使用者按几何与时间上下文消歧。

**一个非常反常、但已经实际用合成程序验证的现象：**

~~~text
DEMO: 纯合成鱼眼角点，不是任何真实设备验收数据
frames=80 valid=77 valid_ratio=0.963
p95 reprojection=0.176px
p95 world tool translation=27.14mm
p95 world tool rotation=6.61deg
~~~

在这个确定性测试中，**只给四角增加 0.1 px 高斯噪声、重投影 p95 仅约 0.18 px，但最终 TCP 世界位姿 p95 竟可偏 27 mm、6.6°**。其中一个主要原因是平面候选歧义：最小角点误差的解不一定对应真实三维姿态。更大的测试噪声可使结果明显恶化。这组数字完全来自本章教学合成几何，**绝不能拿来描述你的设备精度**。

完整可运行脚本：[ego_umi_pose_qc_demo.py](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/ego_umi_pose_qc_demo.py)。

~~~bash
python -m pip install numpy opencv-python
python examples/ego_umi_pose_qc_demo.py
python examples/ego_umi_pose_qc_demo.py --noise-px 0.4
~~~

脚本包含真实的 OpenCV <code>fisheye.projectPoints</code>、<code>fisheye.undistortPoints</code>、<code>solvePnPGeneric(IPPE_SQUARE)</code> 调用，输出有效帧比例、重投影 p95、世界系 TCP 平移/旋转 p95。它**故意保留“只选最低 reprojection 候选”的危险做法作为反例**，而不是把它包装成鲁棒定位算法。下一步真实系统必须补多帧连续性/双目与多标签几何约束，拒绝双解无法区分的帧。

### 真实系统怎么减少双解问题？

| 方法 | 改善点 | 代价与风险 |
| --- | --- | --- |
| 多面 3D 标签支架 | 非共面角点增加深度约束 | 支架体积、制造精度 |
| 双目同时观测同一 Tag | 相机间基线为深度提供独立约束 | 遮挡和边缘鱼眼误差 |
| 连续时间运动先验 | 约束帧间旋转突然跳变 | 错误初始化可能持续 |
| 多个世界锚 Tag 联合估计 | 增加覆盖和锚定稳定性 | 板件布局须准确测量 |
| 融合 IMU 预测 | 短遮挡提供旋转/运动预测 | bias 漂移、缺少绝对约束 |

“重投影误差 < 1 px”只能证明**模型和观测图像相容**，并不能单独证明世界轨迹是真值。还必须看独立三维精度、IPPE 解分离度、重定位一致性与工具姿态分布。

# 八、Tag → 工具 TCP：工具外参不是贴上去就自动知道

UMI Marker 对应的坐标系是 U，机器人真正执行抓取的位置通常是夹爪指尖中点/工具 TCP，即 E。你需要标定 $\mathbf T_{UE}$。

### 方法 A：结构测量 + 重复装配验证

使用 CAD 得到初值，手工实测 Marker 中心到 TCP 的位移与旋转，再在不同抓取姿态下检查数据。**换一个 Tag 支架、旋紧螺丝、改变夹爪开合机械中心，工具外参可能改变**。不要拿上周的 $\mathbf T_{UE}$ 直接用于新结构件。

### 方法 B：固定尖端枢轴标定（pivot calibration）

若可以让工具尖端 E 在空间固定一点 P，旋转手柄而保持尖端不动，则每个姿态满足：

$$
\mathbf R_{WU}^{(i)}\mathbf p_{UE}
+\mathbf p_{WU}^{(i)}
=\mathbf p_{WP}.
$$

未知量是局部偏移 $\mathbf p_{UE}$ 和空间固定点 $\mathbf p_{WP}$。把多个不同旋转姿态堆叠：

$$
\begin{bmatrix}
\mathbf R_{WU}^{(1)}&-\mathbf I\\
\vdots&\vdots\\
\mathbf R_{WU}^{(N)}&-\mathbf I
\end{bmatrix}
\begin{bmatrix}\mathbf p_{UE}\\\mathbf p_{WP}\end{bmatrix}
=
-\begin{bmatrix}\mathbf p_{WU}^{(1)}\\\vdots\\\mathbf p_{WU}^{(N)}\end{bmatrix}.
$$

最小二乘解能估工具**平移外参**；但尖端点本身没有姿态，**不能仅靠 pivot 标定完整 6DoF 的旋转外参**。想得到完整 $\mathbf T_{UE}$，需要已知朝向的几何参考、手眼标定或专门的机械基准。转动方向太少时系统病态，应检查矩阵条件数。

### 方法 C：机器人 hand-eye（已知机器人姿态时）

如果存在机器人真实末端位姿，许多 hand-eye 标定可归结为：

$$
\mathbf A_i\mathbf X=\mathbf X\mathbf B_i.
$$

其中 $\mathbf A_i,\mathbf B_i$ 是两套坐标系的**相对运动**，$\mathbf X$ 为固定外参。但 AX=XB 的轴定义与左右乘顺序依硬件装配而变；不能在不知道 A、B 各代表什么的情况下直接套用通用库函数。

# 九、质量评价必须分四层，不能只看一个 RMS

## 9.1 图像层：reprojection error

对第 $i$ 帧第 $j$ 个角点：

$$
\mathbf e_{ij}=
\mathbf u_{ij}-
\pi(\mathbf K,\mathbf D,\mathbf T_{CT_i},\mathbf P_j),
\qquad
e_{\mathrm{rms}}
=\sqrt{\frac{1}{N}\sum_{i,j}\|\mathbf e_{ij}\|^2}.
$$

这里 $N$ 指实际计入的**角点观测总数**。报告全图像范围分桶残差、p50/p95 与最大值；只给训练集全局 RMS 可能掩盖图像边缘大误差。

## 9.2 标定层：外参与时间的一致性

固定安装后重复三次独立采集、分别估 $\mathbf T_{C_LC_R},\mathbf T_{CI},\delta$。对比旋转和平移分布与机械量测；给参数置信区间。若外参随着采集姿态而飘，优先检查镜头模型、板件平面度、安装刚性和时间，而不是直接用轨迹滤波掩盖。

## 9.3 轨迹层：ATE/RPE + 静止场景 + 多解歧义

ATE 衡量相对于独立真值的位置差，RPE 衡量短时间的相对运动误差。若绝对世界锚已经固定，不能任意用每条轨迹的刚体配准吸收系统性世界系偏移；应同时报告**对齐前绝对误差与必要的对齐后运动误差**。

固定 Marker 不动的测试能暴露噪声，但不能证明大幅快速运动的动态精度。建议专门测试：慢速、快速平移、多轴旋转、近距离、远距离、遮挡后重定位、重新启动设备、标签边缘视野。

## 9.4 训练层：action supervision validity

为每帧定义原因码 <code>qc_reason</code>，把“没有可信动作”的帧与“RGB 视频仍有用”的帧分开保留：

| 场景 | RGB 可以用？ | 末端动作可作真值？ |
| --- | --- | --- |
| 图像清晰、双解解决、世界锚稳定 | 是 | 是 |
| Tag 被手臂挡住，IMU 短暂预测 | 是 | 视任务与误差门限，可暂时置 false |
| 世界锚丢失且 SLAM 地图跳变 | 是 | 否，直到重新建立可信变换 |
| 图像与夹爪开合延迟不明确 | 可能 | 否，不应用于接触动作监督 |
| 工具外参版本未知 | 是 | 否 |

以下只是**首轮工程建议起点**，要通过你的设备精度要求反推，**不是“达标即毫米级”的官方阈值**：

- 至少采集三次独立标定数据，图像中心/边缘都要覆盖；
- held-out 标定视频报告重投影 p50/p95，且确认 IPPE 双解判别；
- 另用独立外部量具/动捕报告工具 TCP 平移和旋转误差的 p95；
- 视频—夹爪开度同步误差与位姿误差需单独建预算；
- 对每条拒绝轨迹保留原始观测、拒绝原因，不能从训练集中无痕删除。

# 十、误差如何传递到 TCP？公式之外还要给物理直觉

设工具位姿由三个有误差的变换相乘 $\mathbf T_{WE}=\mathbf T_{WC}\mathbf T_{CU}\mathbf T_{UE}$。在李代数一阶近似下，可写小扰动 $\delta\boldsymbol\xi\in\mathbb R^6$，由雅可比传播协方差：

$$
\boldsymbol\Sigma_{WE}\approx
\mathbf J_C\boldsymbol\Sigma_C\mathbf J_C^\top
+\mathbf J_U\boldsymbol\Sigma_U\mathbf J_U^\top
+\mathbf J_E\boldsymbol\Sigma_E\mathbf J_E^\top
+\boldsymbol\Sigma_{\mathrm{cross}}.
$$

**公式成立前提**：扰动足够小，线性化点正确；若三项误差不独立，交叉协方差不能省略。图像重投影残差并不能直接替代任何一项 6×6 位姿协方差。

**时间偏差也需要进入动作精度预算**。为一个精细插接工位假设允许末端位置误差 $\le 2$ mm、实际最大速度 0.4 m/s。即使完全忽略其他误差，仅平移时间项就要求：

$$
\vert\Delta t\vert\lesssim
\frac{0.002}{0.4}=0.005\text{ s}=5\text{ ms}.
$$

这个值**只是反推的必要条件，不是充分条件**：视觉标定、Tag 追踪、工具外参、控制延迟和动态运动误差也要分预算。若目标是 2 mm，实测 10 ms 同步错位就已经值得单独修复，无需再争论 PnP RMS 的小数点后三位。

# 十一、最后一步：从可信位姿生成 VLA 动作 Chunk

得到统一世界系工具位姿 $\mathbf T_{WE}(t)$ 后，选择动作表示前必须回答三个问题：**世界系还是工具局部系？绝对位姿还是增量？每步控制时间 $\Delta t$ 多大？**

工具局部系的下一步 SE(3) 增量定义为：

$$
\Delta\mathbf T_t^{body}
=(\mathbf T_{WE}(t))^{-1}\mathbf T_{WE}(t+\Delta t).
$$

世界系的位移增量则是：

$$
\Delta\mathbf p_t^W=
\mathbf p_{WE}(t+\Delta t)-\mathbf p_{WE}(t).
$$

两者**不等价**。若工具旋转 90°，它局部 x 的正向与世界 x 正向未必一致。训练若以工具 body-frame 的动作标签监督，推理/机器人控制也必须按同一约定解释，否则动作会朝错误方向执行。

选一种明确旋转参数化（旋转矩阵、6D rotation、axis-angle 或 quaternion），统一符号与单位。做 chunk 时令：

$$
\mathbf A_t=
[\mathbf a_t,\mathbf a_{t+1},\ldots,\mathbf a_{t+H-1}]
\in\mathbb R^{H\times D_a}.
$$

未来 $H$ 个标签如果包含无效帧，必须有逐维的 <code>action_mask</code>。**不能仅将无效动作填 0，再让模型把 0 当成真实示范。**

### 一个数据样本的最小协议

~~~json
{
  "episode_id": "demo-001",
  "timestamp_ns": 123456789000,
  "camera_exposure_mid_ns": 123456789000,
  "frame_ids": ["ego_left", "ego_right", "world", "umi_marker", "umi_tool"],
  "calibration_version": "rig-20261028-v1",
  "clock_mapping": "t_imu = t_cam + delta_ci",
  "coordinate_convention": "T_AB maps B-coordinates to A",
  "action_space": "body_SE3_delta_plus_gripper",
  "dt_s": 0.033333333,
  "action_horizon": 25,
  "action_valid_mask": [true, true, false],
  "qc_reasons": ["OK", "OK", "UMI_TAG_OCCLUDED"]
}
~~~

这只是 schema 结构示例，<code>action_valid_mask</code> 为便于阅读只展示 3 个元素，**不是完整 25 步窗口**。真实存储还需位姿数值、原始帧索引、夹爪数值、标定文件哈希、机器人坐标转换与来源信息。

进一步接 [只有 RGB 的 EGO 如何训练 π₀.₅](../../04-human-to-robot/ego-rgb-only-to-pi05/)：无工具动作的纯 RGB 样本不能直接计算标准 Flow Matching；而本教程经过标定的 EGO/UMI 则可能提供**人类/UMI 动作监督**，后续仍需明确机器人动作空间映射。

# 十二、半天完成的可执行桌面 Demo（验收不是口头说成功）

| 阶段 | 实际操作 | 交付物 | 通过证据 |
| --- | --- | --- | --- |
| 0. 锁定设备 | 固定相机、分辨率、IMU 频率与工具支架 | 硬件/固件版本清单 | 重新插拔后配置一致 |
| 1. 双目校准 | 动 AprilGrid 拍多方位图 | K、D、T_LR、PDF 报告 | 未见标定帧的重投影误差 |
| 2. Cam-IMU | 固定 AprilGrid，摇动刚体相机+IMU | T_CI、delta、噪声 | 报告残差、时间和外参稳定性 |
| 3. 工位世界锚 | 固定桌面 Tag，测真实尺寸 | W 系定义、T_CW 观测 | 重启设备轨迹锚点一致 |
| 4. UMI 标签 | 标记贴到工具刚体上 | T_CU、双解标志 | 多距离多角度均可追踪 |
| 5. 工具外参 | CAD/量测与 pivot/手眼校验 | T_UE | 末端对准已知物理点时误差 |
| 6. 时间核对 | 重复抓放/快速转动 | 光学与 IMU/夹爪对齐报告 | 明确 p95 错位时间 |
| 7. 数据导出 | SE(3) 相对动作、夹爪、质量 mask | JSON/MCAP/HDF5 或 LeRobot | 同时可视化原图/轨迹/动作 |
| 8. 训练接入 | 机器人示范或动作域桥接 | 真实含 action 的训练 batch | Tensor shape + mask + loss 核验 |

**一个强制反例检查**：保持 UMI 工具完全不动，只移动头戴相机。如果输出的世界系 UMI 仍随相机跑动，则你的变换链、世界锚识别或时间对齐必有问题。另一个检查是相机不动，手持 UMI 只沿已知尺子移动 10 cm，估计长度应接近物理量具，并在不同 Tag 朝向重复执行。

## 十三、失败排查清单：先改数据，再改模型

| 症状 | 最可能的原因 | 对应排查 |
| --- | --- | --- |
| PnP 误差很小但位姿突然翻转 | 平面标记双解 | 查看候选、姿态先验和双目结果 |
| 左眼正常，右眼世界轨迹漂 | 两路内参/外参、错帧 | 单独做右眼 held-out 校验 |
| 静止工具跟着头部移动 | T_CW 反向或时序错配 | 固定工具转动相机 sanity test |
| 远距离尺寸系统性偏大 | tagSize、打印缩放、鱼眼畸变 | 卡尺实测角点边长与边缘残差 |
| 快速抓取在视频里“晚一步” | 曝光时刻与编码器时间不同步 | 分析夹爪事件和视觉帧延迟 |
| 训练损失正常但机器人动作错方向 | 世界/局部增量混用 | 复核 body vs world action 约定 |
| SLAM 地图重置后轨迹跳变 | world/map_id 未建立合法桥接 | 不要把不同地图零点硬拼接 |
| 小样本示范模型只输出零动作 | 监督 mask 缺失或全零填充值 | 可视化每维监督样本分布 |

# 十四、参考来源与边界

1. [Kalibr 官方工具箱](https://github.com/ethz-asl/kalibr)：多相机投影、外参与 CAM-IMU 时空标定、标定报告。
2. [Kalibr 多相机官方 Wiki](https://github.com/ethz-asl/kalibr/wiki/multiple-camera-calibration)：标定采集方式、命令和 camchain 输出。
3. [Kalibr CAM-IMU 官方 Wiki](https://github.com/ethz-asl/kalibr/wiki/Camera-IMU-calibration)：IMU YAML、时间标定与报告。
4. [OpenVINS 官方标定指南](https://docs.openvins.com/gs-calibration.html)：可观测性、多轴激励、时钟、噪声检查建议。
5. [AprilRobotics / AprilTag 3](https://github.com/AprilRobotics/apriltag)：标签角点、真实尺寸、姿态估计与双解。
6. [OpenCV 官方 fisheye](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)、[solvePnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)：鱼眼模型、归一化角点、PnP 返回方向、IPPE_SQUARE。
7. [本仓库可运行 Python 合成质检示例](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/ego_umi_pose_qc_demo.py)：用于验证重投影与三维精度可能不一致；不等于真机算法。

**明确的局限**：本文未执行实际相机标定，也未报告你的双目鱼眼设备的真实精度；Kalibr 的命令示例还需要对应 ROS bag 话题、传感器 YAML、标定板和实际软件版本。脚本的 OpenCV 合成实验能够解释几何机制，但没有实现完整 VIO、连续时间批量优化和真机安全控制。
