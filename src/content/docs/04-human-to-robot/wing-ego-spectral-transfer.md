---
title: "WING 深度教程：从纯 RGB EGO 中提取可迁移潜动作与频谱引导"
description: "论文与项目页核验：相机运动解耦、WING-LAM 教师学生蒸馏、时间轴 DCT、谱预测器与门控动作生成；含数学推导、张量契约、CPU 验证与工业 EGO/UMI 复现路线"
sidebar:
  order: 8
---

> **调研日期：2026-11-01｜原论文 2026-10-02，arXiv:2610.03607。** 本篇区分四类内容：**【论文/官方项目】**作者明确公布的方法和实验；**【数学推导】**对其机制的通用等价解释；**【替代性 Demo】**可在无 GPU 环境独立验证的数学与数据契约；**【工程建议】**应用到双目鱼眼、IMU、UMI 的可行路线。**作者项目页显示 Code Coming Soon；本文不是已复现的 WING 训练报告。**
>
> 一手资料：[论文原文](https://arxiv.org/abs/2610.03607) · [作者项目与方法说明](https://mikuz12.github.io/wing/) · [官方项目展示](https://mikuz12.github.io/wing/#method)；前置阅读：[只有 RGB 的 EGO 如何训练 π₀.₅](./ego-rgb-only-to-pi05/) · [Human-to-Robot Transfer](./human-robot-emergence-deep/) · [Ego-Pi](./ego-pi-deep/)。

# WING：不要求每段 EGO 都有机器人动作，但不能让视觉变化冒充动作

**一句话**：WING 将人类第一视角视频中的**交互变化**和**观察者运动**分开，学习对相机扰动更稳定的潜在动作；再提取人机更容易共享的**时间低频结构**，作为机器人动作模型的条件引导。**机器人动作仍由配对机器人示范监督**，潜在动作不直接成为机器人关节角真值。

![WING 整体系统图，EGO相机运动、WING-LAM、DCT和机器人动作训练/推理](/Embodied-AI-Roadmap/diagrams/wing-spectral-system.svg)

*图 1｜从视频到机器人：先在 EGO 侧学习交互谱信号，再使机器人动作模型使用它。图中“未来帧”只在训练中用于建立谱监督，不能进入推理输入。教学重绘，并非逐层完全复刻作者架构。*

## 0. WING 与之前三篇 Human-to-Robot 工作究竟哪里不同？

假设你在工厂录制 10,000 小时 EGO，其中大量视频**只有 RGB**，没有机器人关节动作。你不可能直接将所有帧送给标准 openpi 的 Flow Matching 损失，因为对每段动作块必须知道监督真值 $A$。但从相邻帧可以分析“手和物体如何相互变化”。

最大陷阱在于，**图像发生变化并不等于手或物体发生变化**：工人低头看零件时，整张画面移动；如果 Latent Action Model（LAM）只通过重建下一帧学习，会很容易把“头在移动”也编码进“动作”。

WING 的解决问题顺序非常明确：

| 路线 | 需要的人类监督 | 学到的东西 | 如何获得机器人 Action |
| --- | --- | --- | --- |
| [Ego-Pi](./ego-pi-deep/) | 可对齐的人类双手动作，配对机器人数据 | 跨本体任务规则与动作联合学习 | 使用机器人示范做动作监督 |
| [Human-to-Robot Transfer](./human-robot-emergence-deep/) | 3D 人手动作等表征，配对机器人数据 | 多样预训练使跨本体迁移更明显 | 使用机器人示范做动作监督 |
| [EgoEngine](./egoengine-deep/) | RGB 与数字孪生/运动估计 | 从人类视频**生成机器人观测与动作** | 重定向与执行约束构造伪机器人动作 |
| **WING（本文）** | EGO 视频；训练期运动监督可由点轨迹与手物区域构建 | 相机不变的交互 latent + 低频谱引导 | 机器人示范仍监督最终 Action |
| 纯 RGB π₀.₅ 直接 FM | 无人类动作也无机器人动作 | **监督不成立** | 无从构造 $A$ 与 $\epsilon-A$ |

所以 WING 最适合先作为**EGO 视频预训练/辅助表征路线**，而不是宣称“EGO RGB 单独训练出机器人策略”。

## 1. 整体架构与五个训练阶段

【官方项目】WING 的核心是 WING-LAM（交互中心的 Latent Action Model）与 Spectral Latent Action Guidance（谱潜动作引导），配合世界/动作建模。项目页明确介绍：

1. 用**背景点轨迹**拟合全局相机 warp；使用手物区域定位交互运动。
2. 教师网络把相机运动和交互残差分到不同潜变量分支。
3. 将交互分支蒸馏为只看两帧 RGB 的学生 WING-LAM；在相机扰动增强下维持交互表示一致性。
4. 将跨帧潜动作序列做时间轴 DCT，保留前 $K$ 个低频系数，训练当前观测条件下的谱预测器。
5. 推理时用谱预测器输出 Guidance Token，并通过门控残差的 Attention 引导机器人策略生成动作块。

【训练流程解释】按信息依赖可分为：**离线监督构造 → Teacher 运动路由 → RGB Student 蒸馏 → 视频预训练谱预测 → 机器人动作监督训练**。每阶段都要输出独立可检验的结果，绝不能只凭最后 LIBERO 分数判断是否真正解决了相机运动混叠。

~~~text
人类 EGO video
   ├──[仅离线训练/构标] 点轨迹 + 可见性 + 手/物体区域
   │       └── 估计背景 global warp W → 相机补偿交互 residual
   │                               └── 两分支 Teacher
   └── 原始 RGB(t), RGB(t+Δ) ────────┤
                                    ↓ 交互 latent 蒸馏
                            RGB-only WING-LAM
                                    ↓ 冻结编码器
                     latent trajectory [B,H,D_z]
                                    ↓ 时间轴 DCT
                       target [B,K,D_z] 仅训练可用
                                    ↓ 当前 observation / language / state 训练 Predictor
                             predicted guidance tokens
                                    ↓ gated residual attention
robot demonstrations (observation, true robot action chunk)
                                    ↓ Robot Flow Matching / policy learning
                                    ↓
                      执行机器人 action + 真实闭环验证
~~~

### 1.1 张量账本：先分清不同时间轴

以下形状用于**工程化复现设计**，其中 $H=8$、$K=4$ 来自作者介绍的频率消融示例；其余如图像分辨率和动作预测长度是示意，不是作者公开的模型全配置。

| 中间量 | 形状示例 | 含义与是否需要 GT |
| --- | --- | --- |
| 帧对 $(I_t,I_{t+\delta})$ | 各为 $[B,3,H_{\mathrm{img}},W_{\mathrm{img}}]$ | 仅学习潜动作时的观察 |
| 背景轨迹 $P_t,P_{t+\delta}$ | $[B,N,2]$ | Teacher 训练的离线监督 |
| 手/物体 mask | $[B,1,H_{\mathrm{img}},W_{\mathrm{img}}]$ | 用来区分全局与交互 |
| 相机变换 $W$ | $[B,2,3]$ | 图像平面仿射近似，**不是完整 SE(3) 运动** |
| Student 交互 latent $z_t$ | $[B,D_z]$ | 用 RGB 帧对估计，不是物理关节动作 |
| latent trajectory $Z$ | $[B,H,D_z]$ | 一段交互 latent 时间序列 |
| 频率系数 $C$ | $[B,H,D_z]$ | 对时间轴作 DCT |
| 截断谱目标 $G=C_{0:K}$ | $[B,K,D_z]$ | 用**未来帧**建立的训练监督 |
| Predictor 输出 $\widehat G$ | $[B,K,D_z]$ | 只使用**当前观测/语言/state** |
| 机器人动作块 $A$ | $[B,T_a,D_a]$ | 真实机器人 demonstration 监督 |
| 速度场预测 $v_\theta$ | $[B,T_a,D_a]$ | 与机器人动作空间一致，不是 latent |

**牢记两个 $H$ 不是一回事**：这里的 latent 时间长度 $H$ 与机器人动作块长度 $T_a$ 逻辑上独立；$K=4$ 不代表每次只能执行 4 帧动作；如果使用不同数据采样率，也不能把 DCT 系数直接一一映射到物理控制频率。

## 2. 第一性原理：为什么仅重建相邻帧会学错潜动作？

图像 $I_t$ 由相机位姿 $T_{WC}(t)$、场景几何、物体姿态与光照共同决定。即使机器人/人手完全不动，只要头戴相机旋转，图像光流依然不为零。因此：

$$
\Delta I = \Delta I_{\mathrm{camera}}
          + \Delta I_{\mathrm{interaction}}
          + \Delta I_{\mathrm{appearance}}
          + \text{高阶耦合项}.
$$

这是为说明混叠的**局部概念分解**，不是严格像素空间线性恒等式。无约束的帧重建编码器可能选择容易解释的大范围背景运动，从而压制局部手物接触信息。

例如工人把零件从桌面拿起 5 cm，同时相机前移；大面积背景像素移动通常远多于零件局部像素，单纯最小化视频重建误差可能优先学会“镜头在前移”，但这对机器人装配动作迁移没有帮助。

### 2.1 用背景运动估计全局相机 Warp

【官方方法】先用离线点跟踪与手物区域排除交互轨迹，对背景点用鲁棒估计拟合全局 warp。论文方法使用一类全局仿射近似；在 2D 坐标中写为：

$$
p' = W(p) = M p + b,\quad
M\in\mathbb R^{2\times 2},\ b\in\mathbb R^2.
$$

对背景轨迹 $\mathcal B$ 拟合：

$$
(\widehat M,\widehat b)
=\arg\min_{M,b}\sum_{i\in\mathcal B}
\rho\left(\|p'_i-(Mp_i+b)\|_2\right).
$$

$\rho$ 表示对错误跟踪更鲁棒的损失；使用 RANSAC 时可通过内点选择和重拟合近似实现。**不要把背景上的移动叉车、滑动物体当静态背景**，否则估计的“相机运动”就会混入环境运动。

### 2.2 反向补偿得到残差

将下一帧位置映射回第一帧参考系：

$$
r_i = \widehat W^{-1}(p'_i)-p_i.
$$

**数值例子**：一维情况下头部平移让背景点从 100 像素变到 120 像素，而手部点从 200 变到 225。若 $\widehat W(p)=p+20$，则背景残差 $r_{\mathrm{bg}}=(120-20)-100=0$，手部残差 $r_{\mathrm{hand}}=(225-20)-200=5$ 像素。**这 5 像素可以描述交互相关的剩余运动，但绝不是 5 mm 或机器人末端位移**。

更严格的情况是透视投影：

$$
\tilde p_{t+\delta}\sim
K\left(R_{t+\delta,t}(Z_tK^{-1}\tilde p_t)+t_{t+\delta,t}\right),
$$

其中 $Z_t$ 是深度。**单个 2D 全局仿射矩阵无法精确补偿大平移、近景/远景同时存在、明显视差和非刚体背景**。这正是强视差或工业工位大深度差场景需要优先测试的边界。

![WING详细示意：相机运动补偿与时间低频谱引导](/Embodied-AI-Roadmap/diagrams/wing-spectral-detail.svg)

*图 2｜左侧是相机运动补偿，右侧是“语义动作相同、人体与机器人的执行节奏不同”的频域处理。灰色高频被舍弃的是 **latent guidance**，不是把真实机器人控制指令强行低通。*

### 2.3 为什么要 Teacher 两条潜变量分支？

【官方方法】教师有处理相机的分支和交互分支。相机分支解码全局 warp；交互分支解码相机补偿后的局部位移；两者组合解释可见的点轨迹。与单一下一帧重建相比，它显式规定：

- **camera latent** 应该能解释背景的共享运动；
- **interaction latent** 应该对手物残差敏感、对相机扰动不敏感；
- 训练期可以有点轨迹与分割 mask 等“特权监督”；推理时不会强制要求这些贵重模态。

从机制上可以用一个**教学性简化损失**表示：

$$
\mathcal L_{\mathrm{motion}}
=\lambda_c\|\widehat W-W^\ast\|^2
+\lambda_r\sum_i m_i\|\widehat r_i-r_i^\ast\|^2
+\lambda_{\mathrm{recon}}\sum_i\|\widehat p'_i-p'_i\|^2.
$$

此式只说明“相机、残差、合成轨迹”为什么需要监督，**并非论文逐项损失或超参数的原样复制**。论文还考虑相机干预、表征一致性与其他正则化，其精确系数必须等待源码/原文配置核验。

### 2.4 为什么还要 RGB-only Student？

Teacher 在运行时要求点轨迹和 mask，对大量工厂 EGO 数据部署与在线预训练成本高。作者将 teacher 的交互 latent 蒸馏给一个只接收 **相邻 RGB 帧**的 student。即：

$$
z^{T}_{\mathrm{int}}=T_{\mathrm{interaction}}(I_t,I_{t+\delta},\mathcal T,\mathcal M),
$$

$$
z^{S}_{\mathrm{int}}=S_\phi(I_t,I_{t+\delta}),\quad
\mathcal L_{\mathrm{distill}}
=\|z^{S}_{\mathrm{int}}-\operatorname{sg}(z^{T}_{\mathrm{int}})\|_2^2.
$$

其中 $\operatorname{sg}$ 表示 stop-gradient，不允许学生损失回传修改已固定的教师目标。**这不是给机器人 action expert 做像素级蒸馏**：学生学的是中间交互表示。

作者进一步对图像施加“模拟摄像机运动”的全局变换，使原始和增强帧对表达同一交互目标。概念性的不变性正则为：

$$
\mathcal L_{\mathrm{inv}}=
\|S_\phi(I_t,I_{t+\delta})-
S_\phi(I_t,\widetilde I_{t+\delta})\|_2^2.
$$

这里需要注意：**增强必须尽量保持原有手物交互语义**。如果增强时连手物相对位置和接触关系都改变了，不变性损失就会压制真正有用的动作信息。

## 3. 第二个迁移障碍：同样任务不代表同样时间节奏

假设人用 0.8 秒把电池插入槽，机器人为了安全需要 2.0 秒；人的轨迹可能有快速手指捏合，而机器人用平缓的末端接近加夹爪闭合。即使两段任务语义相同，**逐时间点对齐动作序列仍然可能是错误监督**。

WING 提出的一条经验性发现是：跨人机对应的交互 latent 在时间低频部分更接近，较高频分量更可能承载本体相关节奏与局部动作差异。它并不表示高频“不重要”；对于插入、脱离接触、快速滑动，高频细节在具体机器人控制阶段仍然重要。

### 3.1 正交 DCT-II：从公式到张量

令一段 latent 轨迹为 $Z=[z_0,\dots,z_{H-1}]\in\mathbb R^{H\times D_z}$。沿**时间维**做 DCT-II：

$$
C_{k,d}=\alpha_k\sum_{n=0}^{H-1}
Z_{n,d}\cos\left(\frac{\pi}{H}(n+\tfrac12)k\right),
$$

$$
\alpha_0=\sqrt{\frac1H},\qquad
\alpha_k=\sqrt{\frac2H}\quad(k>0).
$$

这组基底正交，所以若保留全部频率，逆变换能够精确重建原始序列；若只保留前 $K$ 个系数：

$$
G=C_{0:K,:}\in\mathbb R^{K\times D_z},
\qquad
\widehat Z = D_K^\top G.
$$

上式的 $D_K$ 为 DCT 变换矩阵的前 $K$ 行。这是**在时间维上最小二乘投影到前 K 个余弦基底的近似**，不是沿潜变量通道 $D_z$ 做 FFT，也不是将 $D_z$ 裁切成 $K$ 个维度。

**一个简单的形状账本**：$B=2,H=8,D_z=32$ 时，latent 为 $[2,8,32]$，DCT 后仍是 $[2,8,32]$；取 $K=4$ 后谱监督为 $[2,4,32]$。它相当于每段有 128 个实数系数，但投影成几个 attention token 取决于实际网络设计，**不能直接断言一定生成4个 token 或128个 token**。

### 3.2 为什么截断低频不等于损坏机器人动作？

机器人策略最终仍通过自己的动作监督预测：

$$
\widehat A\in\mathbb R^{B\times T_a\times D_a}.
$$

DCT 截断作用于**“如何引导动作预测”的潜变量**，不是强制将 $\widehat A$ 的高频项置零。机器人低层控制仍可以学习与接触相关的快速修正。

特别容易混淆的是 [FAST](../01-foundations/vla/pi0-fast/) 的 DCT：FAST 对真实动作块做频域压缩与离散化，目标是高效动作 Tokenization；**WING 对跨本体交互潜变量做低频选择，目标是保留可迁移结构**。相同的数学变换，不代表相同的信息对象与训练目的。

### 3.3 $K=2,4,8$ 是怎么选出来的？

【官方项目】论文在 $H=8$ 的设定中，对比不同保留带宽；在其 LIBERO 四个 suite 的消融中，**$K=4$ 优于 $K=2$ 和保留全部 $K=8$**。其中 $K=2$ 过度简化，$K=8$ 引入更强的本体节奏差异。它是作者实验范围内的结论，绝不能外推为“所有工业插接都必须 K=4”。

真实工业任务应根据动作频率、任务时长、接触变化、数据 fps 做自己的扫描，例如 $K/H\in\{1/4,1/2,3/4,1\}$；如果接触短促且完成度依赖毫米级瞬态，高频也可能包含必须保留的可迁移信号。

## 4. 最关键的训练/推理区别：未来视频不能泄漏

【官方方法】训练时，谱目标来自未来帧对编码的 latent 序列；但机器人真正执行时，**未来图像未知**。因此必须训练一个预测器 $P_\psi$，从当前观察与任务条件预测谱：

$$
G^\ast_{t}=DCT_K\big[
S_\phi(I_t,I_{t+\delta}),
S_\phi(I_{t+\delta},I_{t+2\delta}),\ldots
\big],
$$

$$
\widehat G_t=P_\psi(I_t,\ell_t,s_t),
\qquad
\mathcal L_{\mathrm{spec}}
=\|\widehat G_t-\operatorname{sg}(G^\ast_t)\|_2^2.
$$

这里的 spectral MSE 是**帮助理解的监督表达**；具体 token 归一化、预测头、权重与训练细节要以作者最终代码为准。

训练时 $G^\ast$ 来源于未来帧，这是合法监督；推理时输入只能有 $(I_t,\ell_t,s_t)$，预测器生成 $\widehat G_t$ 后再送策略。将未来真值 latent 直接输入推理，会构成严重的 **future leakage**：离线表现高，真实部署不可实现。

### 4.1 如何进入 Action Expert？

谱预测器的输出先做 projection / normalization，形成与动作策略可通信的 guidance tokens。作者描述使用门控残差式的 Attention 注入。作为**机制示意**，可以写成：

$$
H_{\ell+1}=H_\ell+
g_\ell\operatorname{CrossAttn}
(Q=H_\ell,\ K=G_{\mathrm{token}},\ V=G_{\mathrm{token}}).
$$

$g_\ell$ 是门控，作用是控制辅助信号对原有模型更新的强度。**它是教学抽象，不代表论文就采用了这个精确门控参数化或共享 QKV 权重。**

为什么不是直接拼接 latent 当作 robot action？因为二者维度与物理含义不同：潜变量属于“物体和手发生了何种交互”，而机器人 Action 是关节、末端或夹爪控制量。让 action policy 用 latent 作为条件，能够在保留已有机器人动作建模能力的前提下利用人类视频中的语义结构。

### 4.2 与 π₀.₅ Flow Matching 的关系

如果在已有 π₀.₅ 上做**研究性兼容实验**，机器人动作示范 $A$ 仍为：

$$
x_\tau=(1-\tau)A+\tau\epsilon,
\qquad
u_\tau=\epsilon-A,
$$

$$
\mathcal L_{FM}=
\mathbb E\left[
\left\|v_\theta(x_\tau,\tau,o,\ell,\widehat G)-u_\tau\right\|_2^2
\right].
$$

其中约定 $\tau=0$ 是动作、$\tau=1$ 是噪声。预测器还可由 $\mathcal L_{\mathrm{spec}}$ 监督。**这不是 WING 官方公布的一份现成 π₀.₅ 修改版训练代码**，仅说明如何在保留机器人动作监督时接入谱引导。

关键梯度问题：冻结 $S_\phi$ 时，谱目标 $G^\ast$ 不应被机器人 FM loss 随意修改；而谱预测器和门控投影是否参与 joint training，应按实验设计显式设置 optimizer param groups，核对每组参数梯度是否非零。不要写“加个 embedding 就自动学会跨本体”。

## 5. 数据管线：从 EGO / UMI 到可重放的训练样本

建议一个完整 episode 保存以下字段，区分“必须有”和“可选辅助”：

| 字段 | 来源 | 校验方式 | 参与哪一阶段 |
| --- | --- | --- | --- |
| <code>rgb[t]</code> + <code>timestamp_ns</code> | EGO 头戴相机 | 单调时间、无丢帧、曝光时间一致 | Student 和视频预训练 |
| <code>camera_intrinsics/distortion</code> | 双目鱼眼标定 | 重投影 RMS、视场覆盖 | 离线几何 QC |
| <code>imu[t]</code>、<code>T_world_cam[t]</code> | IMU/VIO/SLAM | 重力方向、漂移、闭环误差 | 辅助相机运动诊断 |
| <code>point_tracks[t,n,2]</code> | 视频点跟踪 | 可见率、错误运动轨迹 | Teacher 特权监督 |
| <code>hand_object_mask[t]</code> | 手/物体分割 | 边界、漏检、遮挡 | 避免背景拟合混入手物 |
| <code>camera_warp[t]</code> | 背景 RANSAC | 内点率与残差分布 | Teacher 相机分支 |
| <code>interaction_latent[t]</code> | 冻结 Student | 相机扰动一致性、操作类别可分 | WAM / 谱引导 |
| <code>spectral_target[t,K,Dz]</code> | 未来 latent 的 DCT | 不得进入推理输入 | 训练谱 Predictor |
| <code>robot_state/action</code> | 真机示范/本体遥操 | 坐标、单位、控制模式、时间对齐 | **只在机器人监督分支** |
| <code>domain/source/quality</code> | 数据治理 | human / UMI / robot / synthetic | 混采与可追溯 |

训练数据建议明确记录 <code>calibration_version</code>、<code>frame_tree</code>、<code>track_visibility</code>、<code>warp_inlier_ratio</code>、<code>valid_frame_mask</code>。**原始 MCAP 可以保留所有传感器，离线导出 HDF5/LeRobot 时只迁移通过 QC 的片段**；不要在归一化阶段默默混合不同方向坐标的世界系轨迹。

### 5.1 EGO 只有 RGB 时到底够不够？

- **学生网络推理**：两帧 RGB 可以输出交互 latent，不必实时提供点轨迹、mask。
- **训练教师的原始流程**：需要通过视频算法离线得到点跟踪、手物区域及相机运动的监督；并非无额外计算的端到端纯 RGB 自监督。
- **策略微调**：依然需要机器人动作示范（或另一条经过严格验证的机器人动作合成路线），不能只凭 RGB 得到真机部署策略。

这三个阶段使用的“需要什么输入”不同，是本论文最容易被讲错的地方。

## 6. 可跑通的 CPU 数学契约 Demo

本站附带 [无 GPU 合成验证脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/wing_spectral_contract_demo.py)；依赖 <code>numpy</code>，不下载论文权重、不训练 WING 模型，验证四个关键数学事实：

1. 背景点满足同一个仿射 warp 时，拟合后补偿残差应接近 0。
2. 手部额外运动在补偿后保留下来；没有人手交互的“看一眼”不应变成“执行抓取”的动作信号。
3. 时间轴 DCT 全频段可逆；截断后的形状为 $[K,D_z]$。
4. 人与机器人有相同慢变化交互、不同高频抖动时，低频谱差异比整段时域差异小。
5. 合成推理接口只接受**当前状态与条件**，未来谱目标只在训练函数中产生。

在仓库根目录运行：

~~~bash
python3 -m pip install numpy
python3 examples/wing_spectral_contract_demo.py
~~~

预期成功标记包括 <code>PASS camera motion removed</code>、<code>PASS interaction residual retained</code>、<code>PASS DCT and spectral bandwidth</code>、<code>PASS no-future inference contract</code>、<code>PASS ALL</code>。

**它不是论文 LAM 的真实误差指标，也没有实测 LIBERO、RoboTwin 或真机成功率。** 合成验证的意义是先防止基础坐标、频谱维度与训练/推理时间方向写错，再考虑 GPU 训练。

### 6.1 理想的端到端研究性复现过程

**阶段 A｜motion supervision**：用 30–60 fps EGO 视频做点跟踪与手物分割，计算背景 warp 的内点比例；人工可视化 20 段“仅转头不操作”和 20 段“操作且转头”的轨迹。验收：背景 residual 显著降低；手物 motion 被保留。用固定相机或 IMU/VIO 提供第二条几何对照，但不能自动等价于论文方法。

**阶段 B｜motion-routed Teacher**：从点跟踪监督训练相机/交互双分支；每次按相机旋转、平移、视差强度分组，检查交互 latent 的相机信息可解码性（例如 camera-motion 线性 probe 的 $R^2$），以及 task-phase probe 准确率。验收不只是总 loss，还要确认不同分支各司其职。

**阶段 C｜RGB Student**：Teacher 冻结；训练帧对网络逼近交互 latent，并施加保持交互语义的 camera warp augmentation。用有/无 Student 蒸馏两组比较相机扰动敏感度，避免“模型输出全 0 也一致”的崩塌解。

**阶段 D｜spectral pretrain**：冻结 Student，构造时间 latent window，DCT 后截取 K。训练预测器仅从当前观察预测未来潜动作谱；按 $K=2/4/8$ 消融，保存它们对人机相同任务的匹配/不匹配相似度。需要覆盖不同演示员、不同动作速度和不同相机晃动强度。

**阶段 E｜policy finetuning**：先跑一个只用 robot 数据的基线；再加入冻结 latent 的谱引导和训练好的预测器。固定机器人示范数量、训练步数、分辨率、数据切分与随机种子，只改变谱监督。报告 action loss 与闭环成功率，避免将额外训练预算误认为 WING 结构增益。

**阶段 F｜真机**：先选无尖锐工具、可反复复位的低速桌面抓放，再做有接触的对齐/插接；限制速度/力、准备急停和遥操作接管。统计 20 次以上不同初始状态的成功率与失败原因，同时统计相机头动、深度变化、场景改变的性能衰减。此处“20”是工程起步建议，不代表足以得到高置信度结论。

### 6.2 最小研究矩阵与预期可证伪结论

| 设置 | Camera debias | DCT K | EGO 训练 | 主要回答 |
| --- | --- | --- | --- | --- |
| Robot-only | 无 | 无 | 无 | 现有底座真实能力 |
| Naive RGB-LAM | 无 | 全谱 | 是 | 相机运动是否干扰迁移 |
| Interaction LAM | 有 | 全谱 | 是 | 单独去相机运动是否有用 |
| WING-like 2 | 有 | 2 / 8 | 是 | 过窄带宽会否丢信息 |
| WING-like 4 | 有 | 4 / 8 | 是 | 中等低频是否最优 |
| WING-like 8 | 有 | 8 / 8 | 是 | 高时间频率是否破坏迁移 |

要同时报告**对相机扰动不变**与**对真实交互变化敏感**这两个目标。不然一个恒输出零 latent 的编码器可能在“不变性”指标上非常出色，却完全没有学习到动作语义。

## 7. 论文实验数字应该怎么读？

【官方项目】作者报告如下仿真基准**平均成功率**，不同 benchmark 的任务组成、动作空间、模型与设置不同，不应跨 benchmark 直接排序难度：

| 基准 | WING | 作者项目页列出的最强对照 | 差值 |
| --- | ---: | ---: | ---: |
| LIBERO（四个 suite 平均） | **99.20%** | LingBot-VA 98.50% | +0.70 百分点 |
| RoboTwin 2.0（clean + randomized 平均） | **93.80%** | LingBot-VA 2.0 93.60% | +0.20 百分点 |
| RoboCasa–GR1（24 个 tabletop 任务） | **57.7%** | LDA-1B 55.4% | +2.3 百分点 |

另外，作者在一套双臂机器人上测试 Pack Objects、Stack Cups、Battery Insertion、Battery Assembly 等四类任务；项目页报告**标准设置下真机平均成功率 75.0%**。要特别区分标准场景成功率、泛化场景成功率与部分任务进度（progress score），不能把“完成部分步骤”也算作完全成功。

**评价结论的证据边界**：+0.2 或 +0.7 个百分点的优势在不同随机种子、训练预算或评测采样下可能不稳定；没有独立的重复实验、置信区间与开源完整实验配置前，不宜把这些数字当成已被外部验证的普遍提升。

## 8. 与 HiFi-UMI、KIWI、MonoEgo、Ego-Pi 的组合关系

- **HiFi-UMI / KIWI / MonoEgo** 解决“如何更高质量地采集和定位”：提供更可靠图像、相机位姿、夹爪或手部追踪。它们可以改善 WING 的 teacher 标签和质量评估，但 WING 的 student 并不因此被强制要求永远输入外部跟踪数据。
- **Ego-Pi / Human-to-Robot Transfer** 更偏直接利用对齐的人类动作或手部 3D 表征进行联合训练。WING 不要求把人手关节角强行塞入机器人控制维度，而是把人类交互建模为**辅助引导**。
- **EgoEngine** 尝试输出成对的**机器人图像 + 机器人可执行动作**，可以为直接 imitation learning 提供动作监督；WING 本身没有声称生成可执行机器人示范。
- **FAST / π₀.₅** 与 WING 不冲突：FAST 更改动作编码方式、π₀.₅ 提供动作生成基线、WING 提供可选交互谱条件。但能否在某个开源主干里零改动集成并保持性能，需另行实验，不能从论文跨方法类比直接得出。

## 9. 工业落地建议：先验证三个最有信息量的反例

**反例 1：转头但手不动。** 正确的交互 latent 应维持稳定；若 latent 的范数与头部角速度高度相关，说明还泄漏 camera motion。

**反例 2：头不动但手移动零件。** 正确 latent 应能识别物体被抓取/移动；若只学到背景不变性，模型会丢失正例。

**反例 3：相同工序，人体/机器人完成速度差两倍。** 低频应能更好匹配同语义；但在插针、接触转换、故障恢复等瞬间动作上，高频不应该被当成无用噪声。

对于双目鱼眼 + IMU 系统，建议在 WING 背景 warp 估计之外记录 VIO 的 $T_{world\leftarrow camera}$、相机 IMU 时间差及视觉残差。**强视差场景**可对比：单仿射补偿、Homography、多平面补偿、深度辅助 $SE(3)$ 重投影；后一种可能更合理，但这是**本站的改进假设**，不是作者的官方消融。

## 10. 证据、开放状态和下一步

**已核实**：论文 2026-10-02 提交；作者网页解释 WING-LAM 的运动路由与 student、时间 DCT、当前上下文的谱预测器、门控残差引导，以及仿真与真机任务；项目显示 **Code Coming Soon**。截至本文编辑时未验证到完整公开训练代码、权重、官方可重现脚本或完整数据分割配置。

**本站已做**：文字与教学 SVG 机制分解、可运行的合成契约示例、数据字段和训练-推理边界设计。**本站没做**：作者原版 WING Teacher/Student 权重训练、200 小时 EGO 训练、真实 LIBERO/RoboTwin 成绩复现、真机部署。

**建议学习顺序**：先跟着第 2 节手算相机残差，接着推导第 3 节 DCT，再画出第 4 节训练/推理信息边界；最后运行第 6 节 CPU 例子。这样理解了“要迁移什么”，才有能力判断未来作者开源代码是否实现了论文宣称的方法。

**参考原始资料**：[WING 原论文](https://arxiv.org/abs/2610.03607) · [WING 官方项目（含方法与消融）](https://mikuz12.github.io/wing/) · [openpi 官方仓库（用于研究性兼容实验）](https://github.com/Physical-Intelligence/openpi) · [本手册 RGB-only EGO 指南](./ego-rgb-only-to-pi05/)。
