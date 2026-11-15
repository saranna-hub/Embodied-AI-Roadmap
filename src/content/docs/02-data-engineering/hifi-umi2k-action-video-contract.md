---
title: "HiFi-UMI-2K 真数据合同：六视角检索、20D 动作、Rot6D 与 π₀.₅ 训练"
description: "基于 HiFi-UMI 官方数据卡、LeRobot v3 和 openpi 源码审计 2000h 数据的真实 ABI：多层分片、每集视频时间偏移、逐维掩码、旋转行约定、绝对动作转相对策略与不泄漏的训练样本"
sidebar:
  order: 10
---

> **2026-11-15 实证范围**：本文审阅 [HiFi-UMI 论文](https://arxiv.org/abs/2607.25895)、[作者公开的 HiFi-UMI-2K 数据卡](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)、[LeRobot v3 官方格式](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) 与 [openpi 配置源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)。本站新增原创教学公式、分层可视化与 CPU 合成契约。**不是**完成了 2000 小时下载、官方模型训练、真实 6 视角解码或真实机器人复现。作者未在上述项目入口提供完整 HiFi-UMI 采集/训练发布栈时，不将数据发布等同于端到端复现。

本篇是 [HiFi-UMI 系统深读](./hifi-umi-deep/) 的第二讲：第一讲解释“如何采出高保真数据”，这里解决更容易让实验静默失败的问题——**怎样正确读出数据，转换为具身基础模型可信的训练样本**。

![HiFi-UMI-2K 分片、Episode、六视角与 action schema 全链路](/Embodied-AI-Roadmap/diagrams/hifi2k-data-system.svg)

## 1. 先理解论文贡献，而不是先跑训练

传统 UMI 采集的可扩展性强，但 EGO/UMI 通常用于预训练，模型后训练仍常依赖小规模机器人遥操作数据作本体锚定。HiFi-UMI 通过**头戴双目惯性 SLAM、同头相机坐标下双手标记、原生双手相对定位、共享 GPIO 触发、两手各两路广视角**提高数据可信度，尝试只用 UMI 数据完成部署策略后训练。

[官方数据卡](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)报告：源语料超过 20,000 小时、480 多场景；**公开的是整理过的 2,000 小时子集**，六路视频，LeRobot v3 风格 Parquet+MP4，CC BY 4.0。约 3 mm 工作空间局部末端精度、传感器时差小于 40 μs、回放验证约 98% 是作者测量/统计，**不是本站或任意复刻设备的验收结果**。

论文将同一模型系列的训练初始化、动作表示与评测协议控制一致，但**比较的并非等数量示教**：VLA 对照每任务约 3,200 条 HiFi-UMI 与 300 条机器人遥操作。因此它说明两种完整数据生产路线的策略效果，而不能由此推出“单条 UMI 比单条机器人数据更有效”。

| 模型家族（论文报告） | UMI-only 后训练 | 机器人遥操作后训练 | 差异 |
| --- | ---: | ---: | ---: |
| StarVLA-QwenPI | 82/160 = 51.3% | 86/160 = 53.8% | −2.5 个百分点 |
| OpenPI-π₀.₅ | 124/160 = 77.5% | 119/160 = 74.4% | +3.1 个百分点 |
| LingBot-VA | 91/160 = 56.9% | 92/160 = 57.5% | −0.6 个百分点 |

这些小样本成功率差异不能宣称统计显著。公开 2,000 h **无法覆盖**论文另用 4,000 h 语料做的完整预训练消融。作者报告在其协议下未见任务动作误差下降 41%、进一步的真实成功率提升 18.1 个百分点；二者不代表从本数据子集自动能得到同样增益。

## 2. 必须先解析“分片里的分片”

HiFi-UMI-2K 仓库顶层不是一个直接可交给普通加载器的数据目录，而是多个外层分片，示意如下：

~~~text
HiFi-UMI-2K/
  chunk-0000/
    part-0000/
      meta/info.json
      meta/modality.json
      meta/stats.json
      meta/tasks.parquet
      meta/episodes/chunk-000/file-000.parquet
      data/chunk-000/file-000.parquet
      videos/observation.images.head_main/chunk-000/file-000.mp4
      videos/observation.images.head_main_stereo_right/chunk-000/file-000.mp4
      videos/observation.images.left_hand_up/chunk-000/file-000.mp4
      videos/observation.images.left_hand_down/chunk-000/file-000.mp4
      videos/observation.images.right_hand_up/chunk-000/file-000.mp4
      videos/observation.images.right_hand_down/chunk-000/file-000.mp4
  chunk-0001/...
~~~

**层级1：外层 chunk/part；层级2：part 内 LeRobot v3 数据与视频分片；层级3：每个文件内部再包含多个 episode。** 不应假设一个 MP4 对应一条示范、一行帧表就对应一帧独立文件，也不能只靠文件名推断 episode 边界。

最小读取顺序：先定位一个 part → 读取它自己的 `meta/info.json` 以获得 `fps` 和路径模板 → 读取 `meta/episodes/**/*.parquet` → 按 `data/chunk_index` / `data/file_index` 与 `dataset_from_index` / `dataset_to_index` 定位该集帧 → 使用 episode 内 video offset 定位六路视频。**跨 part 合并时需要复合键** `(outer_chunk,part,episode_index)`，只用 `episode_index` 很可能撞号。

推荐数据源 manifest 固定 `repo_id + HF revision + outer_chunk + part + source_file_sha + schema_version`；转换物另记录 `code_sha + adapter_config_sha`。单靠“脚本今天跑过”不能重现一年后的训练集。

## 3. 样本字段：不要想当然把布尔 mask 丢掉

官方帧级表列与含义：

| 列名 | ABI / 物理语义 | 训练使用 |
| --- | --- | --- |
| `observation.state` | 20 个 float32，当前双手末端状态 | 当前 proprioception |
| `observation.state_valid` | 20 个 bool，逐维当前状态有效性 | 输入置信度、缺测处理 |
| `action` | 20 个 float32，**绝对下一状态目标** | 转为模型动作 |
| `action_valid` | 20 个 bool，逐维目标有效性 | loss mask |
| `timestamp` | float32，episode 起点起的秒 | 仅用于同集对齐 |
| `frame_index` | int64，episode 内从零开始 | 边界与视频定位 |
| `episode_index` | part/shard 范围内索引 | 必须与外层键联用 |
| `task_index` | int64，映射 `meta/tasks.parquet` | 文本条件 |
| `valid.frame` | bool，作者推荐训练与否 | 有效片段过滤 |

`valid.frame=false` 行**故意保留**以保障 Parquet 帧、MP4 和时间戳严格对齐。不能先物理删除这些行再用剩下的行号读取视频：例如原始帧 9 无效，删除后数组第 9 个元素来自原始帧 10，出现一帧错位，数据表依然可能“看起来没报错”。

训练锚点的有效性定义建议为：

$$
m_t^{\mathrm{anchor}} = \mathbf1[\mathrm{valid.frame}_t]\land
\bigwedge_{j\in J_{\mathrm{required}}}m_{t,j}^{\mathrm{state}}.
$$

未来动作在时间步 $k$、维度 $j$ 的监督是：

$$
M_{t,k,j} =
m_t^{\mathrm{anchor}}\land
m_{t+k,j}^{\mathrm{action}}\land
\mathbf1[t+k \text{ remains in same episode}]
\land \mathbf1[\mathrm{valid.frame}_{t+k}].
$$

这是**本站建议的严格监督合同**，不是作者公开损失的逐字复刻。若只要求一侧手有效，可为左、右手分别设 mask 并使用权重；也必须显式记录单手与双手任务差别。未知值先置安全占位再应用 mask，避免 NaN 乘零仍产生 NaN。

## 4. 20D 的真正布局：右手在前，旋转使用“矩阵前两行”

数据卡明确：

$$
s_t=[s^R_t,s^L_t]\in\mathbb R^{20},
\qquad
s^{\{R,L\}}=[x,y,z,r_0,\ldots,r_5,g]\in\mathbb R^{10}.
$$

`xyz` 为米，`gripper` 为弧度。这里**不是常见的 6D 旋转前两列约定**：官方描述的是旋转矩阵的 **first two rows**。设两行分别是 $u,v\in\mathbb R^3$；可用 Gram–Schmidt 先重建正交行，再得到第三行：

$$
\widehat u = \frac{u}{\|u\|_2},\qquad
\widetilde v=v-(v\cdot\widehat u)\widehat u,\qquad
\widehat v=\frac{\widetilde v}{\|\widetilde v\|_2},
$$

$$
\widehat w=\widehat u\times\widehat v,\quad
R_{\mathrm{row}}=\begin{bmatrix}\widehat u^\top\\\widehat v^\top\\\widehat w^\top\end{bmatrix},
\quad R^\top R=I,\;\det(R)=1.
$$

若把行编码当作列编码恢复，通常等效得到方向不同的矩阵（近似转置）；对于单位旋转看不出问题，换非对称旋转才发现方向颠倒。**必须做非恒等旋转的负例测试**，见 [随教程提交的 CPU 契约代码](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/hifi2k_action_video_contract_cpu.py)。

### 双手共世界系，但不是所有 episode 共享同一世界原点

作者描述的手部局部坐标：指尖为原点，$+X$ 指向手指前方、$+Y$ 向左、$+Z$ 向上；头戴左相机光学系采用 $+X$ 图像右、$+Y$ 下、$+Z$ 前。两手与头部轨迹在**同一个 episode 的共享世界系**表达，但每次录制的世界原点是任意初始化的。

$$
T_{LR}(t)=T_{WL}(t)^{-1}T_{WR}(t).
$$

同集双手相对位姿物理有意义；不同 episode 的世界平移直接比较往往没有意义。这也是为什么策略更适合采用**机器人基座、工具局部或任务参照系**动作，而不是从不同采集场景直接模仿任意世界坐标数值。

## 5. 视频检索：从 episode 相对时间映射到 MP4 内的时间

LeRobot v3 将多个 episode 连接到同一个 MP4。官方 `meta/episodes` 为每路相机保存 `chunk_index/file_index/from_timestamp/to_timestamp`。给定行 $t$：

$$
\tau_{\mathrm{video}}(e,t,c)=
\tau^{(c)}_{\mathrm{from}}(e)+\tau^{(e)}_{\mathrm{row}}(t).
$$

这里 `from_timestamp` 是**视频文件内该 episode 的起始时间**，不是“全体训练数据采集时的 UTC”。`fps` 从本 part 的 `info.json` 取，实际解码时优先按 PTS 搜索，必要时做取最近帧并验证误差；浮点时间乘以 FPS 只适合做近似帧索引，不能一概假定 VFR 视频或边界没有偏差。

正确伪代码：

~~~python
# 从 meta/episodes 找到 episode e 与 video key c 的行
video_file = part / info["video_path"].format(
    video_key=c, chunk_index=int(ep[f"videos/{c}/chunk_index"]),
    file_index=int(ep[f"videos/{c}/file_index"]))
seek_seconds = float(ep[f"videos/{c}/from_timestamp"]) + float(frame["timestamp"])
# 用解码器 seek(video_file, seek_seconds)，并验证返回帧的 PTS 与目标差
# 不允许: 先删除 invalid row，再以训练数组序号作为 MP4 帧号
~~~

对每个六路视频都检查 `video_file` 存在、`from <= t < to`（末尾允许按官方端点协议容差）、帧时间偏差、解码失败、帧 index 连贯性。**视频缺失时不应该把另一相机图像伪装成目标相机**，可声明单视角观测实验并单独统计。

![时间轴与逐维 mask：有效性、视频偏移及动作监督范围](/Embodied-AI-Roadmap/diagrams/hifi2k-data-detail.svg)

## 6. 最危险的“看似正确”：绝对 next-state vs π₀.₅ delta 动作

数据卡：`action_t` 是下一状态的**绝对目标**，形状与 state 一样。它不等于 $a_{t+1}-a_t$，也不等于以工具为原点的 $SE(3)$ 运动。对位置，如果目标控制器需要基座系平移增量，可以计算：

$$
\Delta p_t^B=R_{BW}(p^{W,\mathrm{target}}_{t+1}-p^W_t).
$$

如果是工具局部相对位姿：

$$
\Delta T_t^{E}=T_{WE}(t)^{-1}T_{WE}^{\mathrm{target}}(t+1).
$$

如果机器人动作头需要的是当前状态基准下各未来时刻增量，不应把相邻差 $\Delta T_{t+k}^{-1}\Delta T_{t+k+1}$ 与锚点差 $T_t^{-1}T_{t+k}$ 混为一谈。对旋转必须先行约定恢复 $SO(3)$，再用矩阵乘法或旋转向量转换，**不能直接对 rot6d 六个分量相减**。

[openpi 官方训练配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)的示例明确区分 raw absolute 与 delta，提供 `DeltaActions` / `AbsoluteActions`，并说明示例中夹爪维仍使用绝对量。**这不代表 HiFi-UMI 20D 可以直接套用 LIBERO 的 7D mask 或 normalization**。双臂基座映射、相机视角、状态字段、维度顺序和控制器闭环全需定制。

### Flow Matching loss 在哪里引入 mask

以 $x_s=(1-s)\epsilon+s a$，$v^*=a-\epsilon$ 为例，训练动作块长度 $H$、维度 $D$ 的加权速度损失可写为：

$$
\mathcal L_\mathrm{FM} =
\frac{\sum_{k=1}^{H}\sum_{j=1}^{D}M_{t,k,j}\,
\|\widehat v_{t,k,j}(x_s,o_t,s)-v_{t,k,j}^*\|^2}
{\max(1,\sum_{k,j}M_{t,k,j})}.
$$

这里 $s$ 是 Flow Matching 插值时间，不是视频秒。**官方 π₀.₅ 训练需对齐其具体 action transforms、flow 路径和 tokenizer**；以上是用于展示 mask 插入点的教学式，不是声称复现了官方训练器。

### 时间窗口与 terminal

对锚点 $t$，未来 $H$ 个动作必须来自同一个 episode，且满足：

$$
\mathrm{episode}(t+k)=\mathrm{episode}(t),\qquad 1\le k\le H.
$$

官方末帧的 `action` 往往复写最后可用 target，以保持行数一致；它**不能当成“最后一帧之后又发生了运动”的真实监督**。到 episode 末尾要么做 horizon mask，要么按模型协议做末帧 pad 并在报告中单列 pad 比例，不可跨下一条 episode 填充。

## 7. 训练归一化与数据切分：一处泄漏毁掉结论

设每个动作维度 $j$、有效样本集合 $\mathcal V_j$：

$$
\mu_j=\frac{1}{|\mathcal V_j|}\sum_{t\in\mathcal V_j}a_{t,j},\qquad
\sigma_j^2=\frac{1}{|\mathcal V_j|}\sum_{t\in\mathcal V_j}(a_{t,j}-\mu_j)^2.
$$

$\mathcal V_j$ 必须在**训练 split** 且 `valid.frame=true`、`action_valid_j=true` 之后计算。作者卡片也提示应用有效帧过滤后重算统计。用全量 `meta/stats.json` 的均值套到过滤后的分布，可能扭曲动作尺度；对 rot6d 更要验证所选正则化是否保持可用的几何表示。

反例：先将整个录制切成 H=32 的滑窗，随机把滑窗分成 train/val，重叠的 31 帧可以同时落在两个集合，验证误差显著虚低。建议先按**原始采集 session/场景/对象/任务段**划分，再构造窗口；即使同一任务文本不同 episode，仍需检查是否来自同一个长视频拆条。

对“零机器人后训练”实验，留出任务、对象、视角和环境的交叉评测，报告样本数/有效分钟/唯一 session 数，而不只写 2000 小时总量。

## 8. 下载与检查：可执行步骤、失败先停

最少依赖先使用 Hugging Face CLI 的 dry-run 看清容量和目录，再选择一个 part 的元数据及视频：

~~~bash
python -m pip install -U huggingface_hub pandas pyarrow numpy
hf download simple-world-lab/HiFi-UMI-2K --repo-type dataset --dry-run
# 以下 include 是示例，真实可下载路径以 hf repo 文件树为准
hf download simple-world-lab/HiFi-UMI-2K --repo-type dataset \
  --include "chunk-0000/part-0000/meta/*" \
            "chunk-0000/part-0000/data/chunk-000/file-000.parquet"
~~~

不要用 `load_dataset(...)` 就假设所有六路 MP4 与 episode offset 自动匹配：这里有外层 part 分区以及大规模视频，**需先 inspect 路径、metadata 与实际数据结构**，再决定按部分物化还是 streaming。

~~~python
import json
from pathlib import Path
import pandas as pd
part=Path("HiFi-UMI-2K/chunk-0000/part-0000")
info=json.loads((part/"meta/info.json").read_text())
frames=pd.read_parquet(part/"data/chunk-000/file-000.parquet")
episodes=pd.read_parquet(part/"meta/episodes/chunk-000/file-000.parquet")
required={"observation.state","observation.state_valid","action",
          "action_valid","timestamp","frame_index","episode_index","valid.frame"}
assert required <= set(frames.columns)
assert all(len(v)==20 for v in frames["observation.state"])
print("fps",info["fps"],"rows",len(frames),"episodes",len(episodes))
# 下一关：真实核查各 episode 的 video offset、两个20维mask及训练 split
~~~

上述片段是一个 part 的**结构化审计入口**，不是官方一键训练脚本；官方目录里的行数量巨大，读取整个 parquet 需保证内存。配套 [CPU 不依赖真实文件的反例测试](https://github.com/saranna-hub/Embodied-AI-Roadmap/blob/main/examples/hifi2k_action_video_contract_cpu.py) 可先证实实现方向，再对真实样本作抽检。

## 9. 分层错误模型与质量治理决策

| 错误 | 表面症状 | 独立判据 | 处置 |
| --- | --- | --- | --- |
| Rot6D 行列搞反 | loss 正常，方向/旋转失败 | 非恒等旋转行解码与列解码不一致 | P0 拒训，修复变换 |
| 同步删帧 | 图像与动作错一个事件 | 帧号、原始 index、视频 PTS 联查 | P0 恢复原始索引 |
| 绝对 action 误当 delta | 推理瞬间大偏移 | 零运动负例、action roundtrip | P0 阻断 |
| 视频跨 episode 拼接 | 同一训练样本出现两个任务 | episode offset、future horizon 检查 | P0 阻断 |
| 单手缺测被当双手真值 | 另一只手出现虚假动作 | per-dim mask 的统计与 loss | P1 单手监督或丢弃 |
| 全集归一化泄漏 | 验证 loss 虚低 | split 前后 q01/q99 比较 | P1 只用 train split |
| 旋转正交化退化 | NaN/怪异转动 | rot6d 近零范数、正交残差 | P1 mask 并抽检 |
| WBC 仿真通过 | 仍可能物理失败 | 分开统计 replay 与执行 SR | 不以 replay 替代任务评测 |

**质量回流优先级**：每条训练样本留存 `source part, episode id, frame id, timestamp, sensor validity, action validity, normalization version, reject reason`。失败后能够追溯是同步、定位、动作表达、视觉遮挡，还是策略本身的问题。

## 10. 真正可复刻的最小工程与实验计划

先不急于 2000 h 全量后训练。用 1 个 part、2–3 个典型任务与固定随机种子逐项交付：

1. **L1：索引可证明**。列出 `info.json` / `episodes` / `data` / 六路视频的真实路径，抽取 10 个 episode，报告每路视频 offset/PTS 偏差、缺测率；
2. **L2：动作语义可证明**。对 20D 左右手、米/弧度、前两行 Rot6D、世界系与工具系各跑 3 个正例和 3 个负例；对所有生成的 loss mask 打印有效维度数量；
3. **L3：数据加载可复现**。固定 HF revision，切分 train/val 后对每条训练样本验证不跨集；可用一条样本重新定位原视频帧、追溯原始状态和动作；
4. **L4：训练实验可比较**。冻结 checkpoint / 模型、统一 GPU 小时、H、动作接口、视频视角与 seed，对比 `raw absolute`、`correct relative`、`correct relative + strict mask`；
5. **L5：数据价值可验证**。增加视角减量、掩码放宽、不同采集场景、机器人数据混合比例等消融，以固定测试集的成功率、置信区间、接触/重抓失败和吞吐报告结果。

对 π₀.₅ 的迁移顺序是：**先保证 20D 标注 ABI → 通过坐标/控制转换器映射到目标机器人 action 维度 → 建立独立 normalization → 训练 → 评测**。未检查公开权重或官方训练端的实际输入适配器前，不能宣称“HiFi 20D 直接兼容 openpi”。

**你需要真正会答的三个问题：**

- “为什么视频文件不能按 episode_index 直接切？”——因为 v3 的视频按多个 episode 拼接，需读该集在该视频分片中的 `from_timestamp`，并加上 episode 相对时间。
- “为什么 rot6d 已经6维还会方向错？”——同样6维既可表示矩阵的两行，也可表示两列，Gram–Schmidt 排列不一致得到不同的 $R$。
- “为什么 UMI-only 的 π₀.₅ 77.5% 不能说胜过 teleop 的74.4%？”——160次试验且数据条数与采集场景不相等，+3.1 个百分点仅是该评测的观察差异，不能证明等样本效率或普适优势。

### 参考来源与相关课程

**原始证据**：[HiFi-UMI 论文](https://arxiv.org/abs/2607.25895) · [HiFi-UMI 官方项目](https://cloud.simpleai.tech/simple-world-lab/hifi-umi/) · [HiFi-UMI-2K 数据卡/许可/字段](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K) · [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) · [openpi 官方配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)。

**站内承接**：[高保真采集原理](./hifi-umi-deep/) → [MCAP/LeRobot v3/π₀.₅ 时序与转换合同](./mcap-lerobot-v3-temporal-action-contract/) → [π₀.₅ LIBERO 训练核查](../../01-foundations/vla/pi05-libero-data-loss-lora-source-guide/)。

**复现结论**：目前只完成论文/作者卡片/官方软件接口核查与 CPU 合成契约；尚无真实 HiFi-UMI-2K MP4/Parquet 全链路读取、实际 π₀.₅ 微调和闭环实验结果。