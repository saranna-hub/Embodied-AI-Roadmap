---
title: "MonoTag SLAM 源码审计：米制 Atlas、IPPE 双解、回溯定位与 LeRobot 有效性合同"
description: "MonoEgo 进阶专题：基于 2026-09 论文及开源代码分析米制地图状态、尺度重锚、相机/双腕位姿、动作语义与质量监督"
sidebar:
  order: 10
---

> **核验日期：2026-11-23。** [MonoEgo 原论文 arXiv:2609.34512（2026-09-28）](https://arxiv.org/abs/2609.34512)｜[作者采集仓库](https://github.com/jiejie567/MonoEgo)｜[MonoTag-SLAM 正式代码](https://github.com/jiejie567/MonoTag-SLAM)｜[官方发布与复现说明](https://github.com/jiejie567/MonoTag-SLAM/blob/main/REPRODUCIBILITY.md)。这是 [MonoEgo 入门篇](./monoego-deep/) 的源码级续篇，不重复普通 PnP/BOM；区分**论文报告、代码核对、教学推导与本站实测**。这里的 CPU 脚本不是官方运行或论文精度复测。

# 从单目 EGO 到可信米制示教：关键是保留几何证据

如果把“追踪出的连续数字”直接交给策略网络，单目尺度漂移、错并 Atlas、不同手的遮挡、标记姿态双解都有可能成为看似平滑的错误监督。MonoTag 的价值在于将相机定位、标记刚体定位、地图版本、米制判据和训练掩码拆开管理。

![MonoTag 从原始 RGB 经过静态锚点和动态腕标记形成可审计轨迹](/Embodied-AI-Roadmap/diagrams/monotag-atlas-system.svg)

*图1｜本站原创。蓝色静态 anchor 参与相机/地图米制化；绿色动态 wrist constellation 只作为佩戴件位姿观测，绝不能当成固定 ORB 地标。*

## 1. 已公开什么、缺什么？

| 一手证据 | 作用 | 严格边界 |
| --- | --- | --- |
| [论文 Section III–VI](https://arxiv.org/abs/2609.34512) | 视角/标定、IPPE 拒绝、metric Atlas、回溯、评测 | 作者结果不是本站复测 |
| [MonoTag README](https://github.com/jiejie567/MonoTag-SLAM/blob/main/README.md) | Ubuntu 构建和实际入口 | 作者**已经发布后端源码**，不可声称仅有论文 |
| [process_monotag.py](https://github.com/jiejie567/MonoTag-SLAM/blob/main/process_monotag.py) | SHA256 runtime、缓存策略与输出路径 | 迁移部署要建立本机哈希绑定 |
| [官方 profile](https://github.com/jiejie567/MonoTag-SLAM/blob/main/config/monotag_ubuntu_profile.json) | frozen Ubuntu 开关 | 不能把旧实验 fallback 当正式论文实现 |
| [LeRobot v3 exporter](https://github.com/jiejie567/MonoTag-SLAM/blob/main/tools/export_lerobot_dataset.py) | 完整状态/动作字段与有效性 | “具备 action”不等于有机器人控制真值 |
| [实验与数据范围](https://github.com/jiejie567/MonoEgo/blob/main/results/README.md) | Odin 相机参考、静止腕部误差 | 原始可辨识录制、模型权重、生成缓存未公开 |

复现说明中还记录 2026-09-16 frozen 原生库 SHA256；官方发布将“干净源码可构建、测试通过”和“与论文同一二进制”区分开。未经验证的新编译版本不应冒充那一版测量。

## 2. 观测坐标与输出接口

输入为头戴单目 RGB，论文采集配置是 **1920×1080、90 FPS、8 ms 曝光**，头戴相机和双腕几何标记共享**图像同一时钟**。但镜头焦距/畸变仍需标定，腕部标记刚体布局也须按每件装配实物校准。90 FPS 是**录制率**，不是 SLAM 可实时跑 90 FPS 的证据。实验固定标记黑框边长 **48 mm**，必须按工位实测值配置。

世界 W、相机 C、腕部标记刚体 B、待求操作端 E；T_AB 表示 B 系点到 A 系的列向量左乘变换：

$$
\mathbf T_{WC_t}\in SE(3),\qquad \mathbf T_{CW_t}=\mathbf T_{WC_t}^{-1}.
$$

$$
\mathbf T_{WB_t}=\mathbf T_{WC_t}\mathbf T_{CB_t}.
$$

$$
\mathbf T_{WE_t}=\mathbf T_{WC_t}\mathbf T_{CB_t}\mathbf T_{BE}.
$$

论文估计的是腕部**被动佩戴件**的位姿。上式 T_BE 需要另外标定；没有 T_BE 不能声称恢复工具 TCP，更不能自动恢复夹爪、接触力和手指闭合命令。

## 3. 平面 IPPE 为什么不能简单取第一解？

标记四角像素观测 u、已知标记边长构成的三维角点 p 和相机 K 给出：

$$
\mathbf r_{tjk}=\mathbf u_{tjk}-
\pi(\mathbf K\mathbf T_{CW_t}\mathbf T_{WM_j}\mathbf p_{jk}).
$$

$$
E_{\mathrm{marker}}=\sum_{tjk}\rho\left(\mathbf r_{tjk}^{\top}
\boldsymbol\Omega_{tjk}\mathbf r_{tjk}\right).
$$

IPPE 的两组正深度平面解可能具有近似相同的像素残差但不同的三维姿态。论文 IV-B 提供可审计的**未消歧**条件：两解方向差大于 **5°** 或相机中心距离超过 **20 mm**，且角点 RMS 误差差小于 **0.5 px**。在这样的观测中，不能让某个候选自行建立世界原点或米制尺度；要引入相机运动、共视 ORB 地图、其他完整标记或历史可靠测量进行筛选。

完整接入流程为：ArUco 检测→尺寸、边界/视角、畸变检查→两组 IPPE 候选→候选一致性与正深度→可信静态 marker 的接纳→才进入 BA。质量不够的临近角点可以作为 assist-only **拒绝错误分支**，不能成为强地图因子。

融合 ORB 与已接纳标记角点的目标是：

$$
E(\mathcal X)=\sum_{(t,l)\in\mathcal O}\rho_O\left(
\|\mathbf v_{tl}-\pi(\mathbf K\mathbf T_{CW_t}\mathbf X_l)\|^2_{\Omega^O}\right)
+\sum_{(t,j,k)\in\mathcal M}\rho_M\left(\|\mathbf r_{tjk}\|^2_{\Omega^M}\right).
$$

静态工位 anchor 属于地图；手腕 marker 由于随人移动，必须从背景 ORB 地标中隔离。不同静态 ID 不代表它们一定共面或刚性相连；只有已完成多面标定的**同一腕部 fixture**，才共享已知布局。

## 4. Meter gauge 与 Atlas revision 是独立的状态

单目重建具有尺度规范自由度：场景点与相机位移同时按比例 s 缩放，投影不变。

$$
\pi(\mathbf K(\mathbf R(s\mathbf X)+s\mathbf t))
=\pi(\mathbf K(\mathbf R\mathbf X+\mathbf t)).
$$

因此“单帧看到了已知边长的 marker”，不代表任意尺度的旧 Atlas 已经全局米制。至少要区分：相机 tracking 是否有效、地图是 arbitrary 还是 metric、map_id、revision、腕部观测分别是否有效。

![MonoTag 真假米制、地图重锚定、回溯与动作合同的反例](/Embodied-AI-Roadmap/diagrams/monotag-atlas-detail.svg)

*图2｜本站原创：map_id 不可混用，缺测不可内插为真值；同一世界下连续两帧才可求腕局部动作。*

| Atlas 状态 | 可以用作哪类监督 | 必须拒绝 |
| --- | --- | --- |
| camera 有效、metric 未建立 | RGB 视觉/语义、自监督 | 绝对米制 wrist translation |
| metric、相机有效、单只腕被遮挡 | RGB、另一只有效腕 | 遮挡一手的动作真值 |
| 两张未验证的 map_id | 两组独立 episode | 跨图下一步 delta |
| map revised 且历史已一致更新 | 合法重算的轨迹 | 混用 revision 之间位置 |
| retrospective recovered 有 PnP 支持 | 离线补充可用历史 | 将简单显示插值当测量 |

### 4.1 旧 Atlas 的首次米制化

论文 IV-C 不是靠一次尺度 PnP：旧 arbitrary map 的**尺度提案**要求至少 **8 次可靠观测、3 个 keyframe、约 4 cm** marker 估计平移，并须通过视差和不确定性门槛。若有锚点但视觉视差不足，允许创建一张单独 metric map，不应静默地宣称原 arbitrary 过渡图已变成米制。

教学中可用比率理解规模，但不能把下面的单个比值当成原方法：

$$
\widehat{s}=\frac{d_{\mathrm{marker,m}}}
{d_{\mathrm{ORB,unit}}},\qquad d_{\mathrm{ORB,unit}}>0.
$$

如果 ORB 基线趋近零，微小噪声会被放大；这正是论文做 parallax gating 的原因。尺度证据应由多帧几何与 BA 联合验证。

### 4.2 相隔较远的 A/B 工位如何重锚？

当 A 的 meter gauge 建立后，必须存在 ORB 连续可连通的 A→B 轨迹，才能把 B 的静态锚点拉回同一 Atlas。作者 IV-E 的 B 区闭合条件包括 **8 次可靠观测、B 区 2 个 keyframe、约 4cm 米制基线**、有效视差和不确定性验证。若尺度偏差超过 **3% 且超过3倍标准差**，可以触发有约束的 Sim(3) 重锚；若接近1，也可做优化，但不应反复全图缩放。

$$
|\widehat{s}-1|>0.03
$$

$$
|\widehat{s}-1|>3\sigma_s
$$

这两式只作判断**直觉教学**；作者实际优化在 log-scale 和图约束下进行，不是以上简单阈值代替整个后端。对 metric 地图的普通视觉回环必须限制 SE(3)，不能让视觉外观匹配任意改变尺规；跨图并图与经过验证的尺度重锚属于不同机制。

### 4.3 回溯定位不能伪造轨迹

记录视频的前几秒若相机初始化失败，最终 Atlas 可用于**重新匹配早期原始图像**。作者 IV-D 的 prefix recovery 门槛为至少 **30 个 PnP inlier**、重投影 RMS 不超过 **3px**，还要满足正深度、图像空间分布和连通性：

$$
N^{\mathrm{inliers}}_t\ge 30,\qquad
\mathrm{RMS}_{\mathrm{px},t}\le 3.
$$

**重定位 = 新的图像到地图几何证据**，不是在时间轴上给断开的点做线性插值。短缺口也必须有同图两侧支持；没有证据的长缺测仍然无效。仅在渲染中显示最终地图的早期视图不能算“成功回溯测量”。

## 5. 真实导出源码逐字段审计：147D state 与12D action

[官方 LeRobot exporter](https://github.com/jiejie567/MonoTag-SLAM/blob/main/tools/export_lerobot_dataset.py) 对图像、状态、局部动作和 validity 分别落盘：

| 字段 | 实际 shape | 重要语义 |
| --- | --- | --- |
| observation.images.ego | video, RGB | 源视频帧抽样，源和目标 FPS 独立 |
| observation.state | **[147] float32** | 相机 pose7 + 双腕 pose14 + 双手各21×3个关节点共126 |
| observation.valid_mask | **[5] bool** | camera / left_wrist / right_wrist / left_hand / right_hand |
| observation.confidence | **[3] float32** | 相机及左右腕置信度 |
| observation.map_context | **[2] int64** | map_index 与 map_revision |
| action | **[12] float32** | 左腕局部平移3+旋转向量3，右腕同样6 |
| action.valid_mask | **[2] bool** | 双腕各自能否从当前帧到下一**采样帧**构成 delta |
| source_provenance.json | 元数据 | video/FPS/原始文件 SHA256，以及 direct_robot_control=false |

关键：pose 保存顺序为 **[x,y,z,qw,qx,qy,qz]**；SciPy 的 Rotation.from_quat 需要 **xyzw**，脚本显式重排。不能拿旧版 π₀.₅ 或其他模型的动作维度定义去解释本数据的12个元素。

每只腕的局部增量：

$$
\mathbf d_t=\mathbf R_t^\top(\mathbf p_{t+1}-\mathbf p_t)
$$

$$
\mathbf R_{\Delta,t}=\mathbf R_t^\top\mathbf R_{t+1}
$$

$$
\boldsymbol\omega_t=\mathrm{Log}(\mathbf R_{\Delta,t}).
$$

这是佩戴件的局部 SE(3) 位姿增量，**不是世界系相机位姿差，更不是机器人关节控制量**。样本按输出帧率重新索引：

$$
i_k=\mathrm{round}(k f_{\mathrm{source}}/f_{\mathrm{train}}),\quad
f_{\mathrm{train}}\le f_{\mathrm{source}}.
$$

90→20FPS 后相邻样本间可能跨4或5帧；动作因此也跨多个源帧，不能再误以为每个 action 表示1/90秒。若下一样本不存在（例如最后一帧）、所属 trajectory_segment_graph 不同或 world_submap_id 不同，官方将对应腕的 action.valid_mask 设 false，动作数值置零。

$$
M^a_t=M^{\mathrm{pose}}_t\land M^{\mathrm{pose}}_{t+1}
\land M^{\mathrm{same\ segment}}_{t,t+1}
\land M^{\mathrm{same\ submap}}_{t,t+1}.
$$

**零填充+mask 不是静止真值**。做监督必须把无效维度排除在 loss 分子和有效样本计数之外。工业项目还应追加 metric-gauge、角点质量、时间误差、坐标轴和来源追溯等严格检查，不能把官方导出字段自带的简化 mask 当成全方位工厂 QC。

## 6. 可执行复现：合成契约→后端构建→训练格式

### 第一层：本站 CPU 合成合同

[可运行脚本](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/monotag_atlas_contract_cpu.py) 检查：map 隔离与 invalid 样例、IPPE 双分支拒绝、重锚观测门槛、回溯 inlier/RMS、局部增量方向、时间抽样、mask 填零/末帧。**这不执行作者原生 C++ 后端，不测真实位姿精度**。

~~~bash
python3 examples/monotag_atlas_contract_cpu.py
~~~

### 第二层：官方 Ubuntu 原生后端（本次未执行）

作者验证环境 Ubuntu24.04 / Python3.10+；按 [官方 INSTALL](https://github.com/jiejie567/MonoTag-SLAM/blob/main/docs/INSTALL.md) 提前准备 OpenCV4.10、Pangolin、ORB-SLAM3 的 ORBvoc 字典。

~~~bash
git clone https://github.com/jiejie567/MonoTag-SLAM.git
cd MonoTag-SLAM
make setup
make native DEPS_PREFIX=/path/to/dependencies JOBS=4
.venv/bin/python scripts/init_runtime.py \
  --deps-prefix /path/to/dependencies --vocabulary /path/to/ORBvoc.txt
make smoke
make native-smoke DEPS_PREFIX=/path/to/dependencies
.venv/bin/python process_monotag.py input.mp4 \
  --calib camera.json --head-slam --slam-init auto --auto-marker-map \
  --static-marker-ids 20-49 --static-marker-size-mm 48 \
  --no-hand-joints --slam-replay --save-atlas runs/demo/atlas.osa \
  --output runs/demo/actions.jsonl
~~~

上面 20–49 和48mm 是作者数据示例，**不得直接替换实测 marker 配置**。入口会核对本机 runtime hashes，拒绝输出路径覆盖；corner policy 与缓存不一致也要重算。用另一台机器的 runtime JSON 或旧角点缓存直接跑没有科学可复现性。

### 第三层：官方 LeRobot 数据导出（本次未执行）

~~~bash
.venv/bin/python tools/export_lerobot_dataset.py \
  runs/demo/actions.jsonl \
  --output runs/demo/lerobot --task "assemble component" --fps 20
~~~

源 exporter 将此任务标记为 passive-ego-demonstration，直控标志 false。后续接到 π₀.₅ 需额外解决**人类腕→机器人末端的可达约束、工具外参、夹爪、力接触及动作时序**，不要仅将12D复制进 Action Expert。建议做三组公平实验：RGB-only 辅助预训练 / 粗腕轨迹伪监督 / 带质量 Mask 的腕轨迹监督，在相同机器人数据量和评测 seed 下比较收益。

## 7. 原文结果：不能混用静止精度和动态定位

作者三段相机 Odin 对比，MonoTag 主地图 **SE(3) ATE 分别 0.302 / 0.196 / 0.150 米**，对应 **78.1% / 99.5% / 99.2% coverage**；Odin 是多传感器参考里程计，不是独立绝对真值。另四条**腕部标记固定不动**的录像得到每侧约 **1.26–2.69 mm RMS scatter**，评估的是精密性和重建稳定，不代表动态操作腕轨迹达到2mm绝对定位，也不代表人体解剖腕或工具末端误差。

需要在真实工位补充相同视频下的 ORB-only/marker-only/MonoTag，IPPE-首解/两候选、无锚/有锚、前向/回溯、raw/masked 轨迹消融；同时报告**ATE、RPE、覆盖率、map碎片、来源回溯成功率**，不可只挑有效帧计算较低误差。

## 8. 局限与合规边界

没有可见可靠标记时可能丢失米制尺度支持；反光、近正对标签、局部遮挡、背景缺纹理及快速旋转可能增加拒绝率。被动腕标记不产生手指/接触/抓握监督。对于录像编辑/遮挡修复，应先用**原始未改 RGB**定位，之后才能生成派生视频；视频隐私遮挡也不等于恢复被遮挡的真实观测。

MonoTag 后端基于修改过的 ORB-SLAM3，发布许可证为 **GPL-3.0**；硬件图纸和文档另有 **CC BY 4.0**，可选手模型/数据各自受协议约束。要用于工厂商业系统，需要独立评估许可与数据合规，不能假定复制源码后所有衍生代码均可闭源发布。

**本次站点可验证范围**：论文/作者代码交叉核对、原创 SVG、CPU 合成动作与地图契约；**未执行**官方 Ubuntu native、真实90FPS录像与外部跟踪对比、LeRobot 真实数据产出、GPU训练或下游机器人策略成功率验证。
