---
title: UMI：不用机器人采集可学习的操作示范
sidebar:
  order: 2
description: UMI 的硬件、动作恢复、时空对齐与策略学习
---

**论文**：[Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots](https://arxiv.org/abs/2407.10214) · **官方代码**：[real-stanford/universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) · **项目**：[UMI](https://umi-gripper.github.io/) · **状态**：原理论文与官方公开工具链解析，本站未完成整机复现。

## 1. 为什么“视频里人抓起杯子”还不够训练机器人？

视频只记录看到的事情，机器人学习控制需要知道：在某一时刻，夹爪究竟在哪里、朝向哪里、开合多少，以及下一段如何运动。普通 EGO 虽便宜，但缺少机器人的控制标签；让工人直接遥操机器人又受到本体数量、工作空间和成本限制。

UMI 的思路是做一个**人可以拿在手里操作、但末端形态与机器人更接近**的采集接口。它把视觉、夹爪动作与空间轨迹变成同一条示范，减少数据与下游机器人之间的差距。

## 2. 端到端数据流

![图：UMI 从采集到训练](/Embodied-AI-Roadmap/diagrams/umi-pipeline.svg)

*图：UMI 从采集到训练。教学重绘；图示未包含完整算法实现。*

这不是“直接从视频估计人的完整五指关节”。采集装置本身通过夹爪约束动作，目标是让记录到的操作更接近可迁移的末端执行器示范。实际组件与配置必须以官方硬件说明为准。

## 3. 重要的不是一条轨迹，而是坐标系

假设 SLAM 在自己的世界坐标系给出每帧位姿：

$$
T_{WC}(t)=\begin{bmatrix}R_{WC}(t)&p_{WC}(t)\\0&1\end{bmatrix}\in SE(3).
$$

其中 $W$ 是 SLAM 世界，$C$ 是相机。若相机到夹爪工具坐标的刚性外参为 $T_{CG}$，则夹爪世界位姿：

$$
T_{WG}(t)=T_{WC}(t)\,T_{CG}.
$$

要得到当前夹爪坐标系下的相对运动：

$$
\Delta T_t=T_{WG}(t)^{-1}T_{WG}(t+1).
$$

**这些式子是标准刚体变换教学推导，不代表 UMI 官方策略只接受这种一种动作编码。** 对机器人训练，必须声明相对变换方向、旋转表示、时间间隔和工具中心点定义；$\Delta T$ 算错顺序，会出现旋转或平移方向反转。

### 时间戳如何对齐？

视频 30Hz 不代表 SLAM、编码器或机器人一定也按 30Hz 同步。若图像曝光时间为 $t_i$、动作测量时间为 $t_a$，策略训练应明确二者对应关系和插值方式；不要简单按数组下标对齐。固定偏差可能造成模型看见“还没接触”却对应“已夹紧”的动作，直接污染模仿学习标签。

## 4. Action Chunk 从哪来？

把当前观测 $o_t$ 与未来 $H$ 步末端动作配成训练样本：

$$
(o_t,\;A_t),\quad A_t=[a_t,\ldots,a_{t+H-1}],\quad A_t\in\mathbb{R}^{H\times D_a}.
$$

训练策略学习 $p_\theta(A_t\mid o_t)$。UMI 原工作使用 Diffusion Policy 及针对真实示范的处理链，重点是跨不同环境收集动作、再在机器人上执行。若目标机器人动作空间或夹爪结构不同，仍需要适配，而非宣称物理本体差距已消失。

## 5. 新人容易忽略的三个问题

- **尺度**：纯单目重建存在绝对尺度歧义。需要使用可靠的尺度约束或标定，不能把未经标定轨迹叫毫米级真值。
- **外参**：相机固定位置变化就会导致 $T_{CG}$ 失效；不同采集设备应保存标定版本。
- **轨迹质量**：SLAM 跟踪失败、断帧、突然跳变时，不应简单插值成“平滑正确轨迹”。要保存无效区间与质量来源。

## 6. 对工业轮式双臂的使用边界

如果只是双指夹爪，UMI 的末端 SE(3) 与开合控制可能是有用的迁移接口；如果需要灵巧手丰富接触动作，UMI 夹爪不能代表全部手部动力学。手套触觉可辅助标注抓握/接触/滑移阶段，但不能直接等同于真实机器人末端力。

建议先搭一个固定底盘的双臂取放基线，然后研究 EGO+UMI+少量 Robot 示范是否提高对新物体/新工位的泛化。轮式导航与高频接触控制先分层，不宜一开始把整机所有自由度全交给同一模仿策略。

## 7. 实验与源码

论文展示了真实环境采集与机器人策略学习的能力，但能否迁移到你的机器人，要根据相机、末端工具、频率与任务分布重新评测。官方仓库提供采集、SLAM、训练与执行入口，可阅读 [SLAM pipeline](https://github.com/real-stanford/universal_manipulation_interface/tree/main/scripts_slam_pipeline)、[训练代码](https://github.com/real-stanford/universal_manipulation_interface/blob/main/train.py)、[文档](https://github.com/real-stanford/universal_manipulation_interface)。

**最小验证**：采 30 条同一任务，检查不同 session 的轨迹起点、尺度、夹爪开合与相机帧是否一致；再记录离线轨迹误差和闭环成功率。未执行实验前只能标为“建议验证”。
