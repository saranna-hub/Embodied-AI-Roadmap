---
title: Ego-OSCAR｜开源头戴双目惯性采集完整复现
sidebar:
  order: 4
---
# Ego-OSCAR：低成本可复刻的双目惯性 EGO 采集平台

论文：[Ego-OSCAR (arXiv:2608.08285)](https://arxiv.org/abs/2608.08285) · 官方：[GitHub: fpv-labs/ego-oscar](https://github.com/fpv-labs/ego-oscar) · [技术与 BOM 官方说明](https://www.fpvlabs.ai/essays/ego-oscar) · [公开数据 stereo-550](https://huggingface.co/datasets/fpvlabs/stereo-550)。

> **公开程度**：四种方案中，Ego-OSCAR 的 CAD、固件、主机录制软件、硬件文档公开程度较高，可以跟随官方资料搭建采集系统。**官方声称物料 BOM 低于 200 美元**，不能将其直接等同于在中国采购、含税运/人工的最终落地成本。

![Ego-OSCAR 硬件同步与数据路径](/Embodied-AI-Roadmap/diagrams/ego-oscar-system.svg)

## 1. 硬件：传感器、计算板和物理指标

| 模块 | 官方型号/参数 | 数据与功能 |
| --- | --- | --- |
| 双目视觉 | Dexcin USB 双 Omnivision 全局快门模组 | 左右各 1280×720@30 FPS，同步 RGB |
| 左右几何 | 基线约 42 mm，单目 FOV 约 126° | 双目深度和校正 |
| 输出接口 | 单 UVC/USB 2.0，side-by-side MJPEG | 左右拼接图像（需拆分） |
| 惯性 | TDK ICM-20948，经 I²C | 实际采集 6 轴 acc+gyro，120 Hz |
| 实时控制 MCU | Seeed XIAO ESP32-S3 | SoE/STRB 时间戳、串口、LED、看门狗 |
| 嵌入主机 | Radxa ROCK 5C，RK3588，2 GB | 接收视频、硬件 H.264 编码、存储 |
| 电源与头戴 | 10,000 mAh USB-PD 电池、3D 打印和帽带 | 约 5–6 小时、整机约 280 g（作者报告） |

**关键区别**：全局快门抑制逐行曝光几何偏差，但不能消除运动模糊。ICM-20948 芯片支持的全部传感功能并不意味着系统实际保存九轴姿态；官方方案说明使用加速度＋陀螺仪 6 轴。

## 2. 原始数据与派生数据严格区分

| 模态 | 是否直接采集 | 用途 |
| --- | --- | --- |
| 左右 RGB 视频 | 是 | 视觉操作/双目匹配 |
| 6 轴 IMU | 是 | 惯性测量、VIO |
| MCU 曝光起始/同步事件 | 是 | 视频帧与 IMU 时间对齐 |
| 相机内外参和畸变 | 标定后得到 | 双目校正/深度 |
| 头部 6DoF | 由 VINS/SLAM 估计 | 运动轨迹；不等于真值 |
| 手部检测/三维重建 | 模型推断/标注 | 操作理解 |
| 操作语义文字段落 | 后处理标注 | 语义预训练 |
| 机器人真实关节动作/接触力 | **没有直接测量** | 不能当成直接监督标签 |

公开数据约 550 h，带约 209,315 个动作语义片段，部分手部派生标注。**不能因为有视频+手部检测，就假设已经有可执行机器人动作标签。**

## 3. 系统同步为何需要独立 ESP32？

双目模组内部能同步两眼，但 IMU 运行在另一个时间基。相机的 STRB/SoE 曝光事件接入 ESP32 GPIO；MCU 为事件和 IMU 样本使用同一单调计时基础，再经串口发送给 Radxa 主机。这样可以将**曝光时间**与**惯性采样时间**联系起来，而不是把 Linux 接收 USB 帧的时间误认为曝光时间。

数据流：

双目镜头→UVC MJPEG→Radxa 解码/硬编码 H.264→MP4；同时 SoE→ESP32 timestamp、IMU→ESP32→UART→Radxa CSV/metadata；离线将两路时间对齐，保存 calibration.json 和 session 信息。

## 4. 基于官方源码的复现步骤

### 硬件上电前

核查摄像头确为同步双全局快门模组、SoE/STRB 可访问、输出格式和额定供电；校验 IMU I²C 地址及 ESP32 UART 电平；优先参考官方接线文件，不能根据芯片名称随意猜引脚。

### 获取工程

~~~bash
git clone https://github.com/fpv-labs/ego-oscar.git
cd ego-oscar
find firmware radxa docs -maxdepth 3 -type f | head -80
~~~

### MCU 烧录

打开 firmware/esp32_coprocessor_led_watchdog/esp32_coprocessor_led_watchdog.ino，按配套 README 安装 Arduino 依赖，选择 XIAO ESP32-S3，检查 USE_USB_SERIAL、I²C 与 STRB 引脚，再烧录并验证串口事件。

### Radxa 安装和视频检查

~~~bash
sudo apt update
sudo apt install ffmpeg python3-libgpiod python3-serial python3-smbus rsync v4l-utils
v4l2-ctl --list-devices
ffmpeg -decoders | grep mjpeg_rkmpp
ffmpeg -encoders | grep h264_rkmpp
~~~

**注意**：录制程序依赖 Rockchip 硬件加速编解码。系统自带 FFmpeg 未必含 mjpeg_rkmpp/h264_rkmpp；缺失时需按官方仓库的 Radxa README 构建适配的 FFmpeg，不要直接跳过此项并宣称 30 FPS 已验证。

### 录制

~~~bash
python3 radxa/fpv_recorder.py
~~~

检查录制会话输出中 MP4、IMU CSV、时间偏移、元数据是否齐全。

### 拆左右图像测试

~~~python
import cv2
cap=cv2.VideoCapture("stereo.mp4")
ok,frame=cap.read()
assert ok,"确认替换为实际录像"
h,w=frame.shape[:2]
assert w%2==0
left,right=frame[:,:w//2],frame[:,w//2:]
print(left.shape,right.shape)
~~~

仅是 side-by-side 拆帧，不能代替左右顺序、实际曝光帧 ID、标定和时间同步验证。

## 5. 性能：已报告什么，没有报告什么？

论文/项目报告相机 30 FPS、IMU 120 Hz、约 280 g、约 5–6 小时持续采集、作者标定设置下重投影误差低于 0.03 px；深度在约 0.5–4 m 室内工作范围开展评估。

VINS-Fusion 在 20 段评测中有 **12 段收敛稳定、8 段发散**。这**是轨迹可用性，不是 ATE/RPE 或 60% 位姿精度**。官方未提供这些序列的运动捕捉真值，因此不能宣称毫米级世界定位。

## 6. 单套 BOM 成本

**官方：材料 BOM 低于 USD 200。** 国内非官方预留预算：

| 部件 | 人民币试制估算 |
| --- | ---: |
| Dexcin 类硬同步双目模组 | 300–650 |
| Radxa ROCK 5C + 存储 | 350–700 |
| ESP32-S3、ICM-20948、线束 | 100–280 |
| USB-PD、电池、帽带、3D 打印外壳 | 120–320 |
| 转接、电源与备件 | 80–250 |
| **合计** | **950–2,200** |

不同地区、相机供应情况、存储规格和批次会导致预算与官方低于 200 美元声明不一致；不能当作精确采购报价。

## 7. 必须验收的项目

| 环节 | 验收依据 |
| --- | --- |
| 相机 | 左右视频帧数、标定内外参、极线一致性、USB 掉帧 |
| IMU | 120 Hz 附近实际间隔分布、数据缺失、时间单调 |
| 曝光同步 | SoE 计数与视频帧对应、偏差分布 |
| 长时可靠性 | 30/90 分钟录像、热关机、I/O 错误、线缆松脱 |
| 双目几何 | 标定重投影误差、视差深度与实物测距 |
| VIO | 收敛/发散比例；有独立真值时再报告 ATE/RPE |
| 数据建模价值 | 加入 EGO 数据前后的下游真实机器人成功率 |

## 8. 工业采集选型结论

Ego-OSCAR 适合低成本扩张 EGO 视觉惯性数据生产、为语义与手部重建提供输入，**不能直接替代含双手精确动作标签的 UMI**。对于需要头部姿态+手部操作的系统，可把它作为 EGO 基础设备，再外挂手部估计、定位增强或触觉/手套采集。


## 八、真正的数据接口契约：采集端必须落盘哪些内容

以下是建议的**通用 Canonical Schema**，不是论文官方定义。只有把每一列的物理语义锁死，下游模型才不至于“能训练却学错”。

| 字段 | 建议单位/格式 | 严格要求 |
| --- | --- | --- |
| episode_id | 字符串 / UUID | 一段任务不跨越多个不可解释的地图重置 |
| timestamp_ns | int64，统一单调时钟 | 说明时间原点，禁止混用 unix wall clock 与 MCU clock |
| rgb.* | 视频或压缩图帧 | 相机名、frame_id、曝光起点、有效性 |
| imu.acc | m/s² ×3 | 说明 IMU 坐标轴和重力方向 |
| imu.gyro | rad/s ×3 | 标定陀螺零偏，说明轴向 |
| pose.world_* | 4×4 / SE(3) | 指明 T_AB 的左右下标含义；明确位置单位米 |
| pose_valid | bool | 缺少可靠重建时必须 false |
| pose_source | 枚举 | 直接 marker、VIO、Atlas 回溯或插值 |
| gripper | m、rad 或归一化比例 | 绝不能无单位混用 |
| action | [H,D] | 明确绝对/增量、参考系、未来时刻和有效维度 |
| quality.* | 分项分数/事件 | 数据缺失、重投影、时间偏差、运动异常分别存储 |

### 坐标系的两条不变量

**不变量 A：**相同物理标记固定不动时，变换链求得的世界位姿应保持近似不变；模型不能通过人为平滑掩盖系统性的标定偏差。

**不变量 B：**如果坐标系链为世界 W→相机 C→工具 E，那么全部 T 的定义必须一致。以列向量惯例为例：

$$
T_{WE}=T_{WC}T_{CE}.
$$

对于动作增量，一种常见相对位姿定义为：

$$
\Delta T_t=T_{WE}(t)^{-1}T_{WE}(t+1).
$$

这属于**当前工具局部系**下的相对变换。若需求是世界系增量，应另作定义，二者不能混用。旋转差不能随意用欧拉角逐维相减代替 SE(3) 变换复合。

### 从原始轨迹生成训练样本

推荐先做“独立于训练网络的数据协议验收”：解码传感器→时间插值或选择最近曝光时刻→处理无效帧→统一位姿/工具外参→归一化 action→窗口化未来 H 步→构建观测/动作样本。

最小有效性门槛不只是“没有 NaN”：相机重定位后突然跳变、不同 SLAM 地图被无条件拼接、夹爪状态与视频错位，即使全部数字合法也不应进入真值监督。

## 九、质量验证脚本：时间、速度与异常检查

~~~python
import numpy as np

def verify_timestamps(ts_ns):
    t=np.asarray(ts_ns,dtype=np.int64)
    assert t.ndim==1 and len(t)>2
    dt=np.diff(t)/1e9
    assert np.all(dt>0), "时间戳重复或倒退"
    return {"median_dt_s":float(np.median(dt)),
            "p95_dt_s":float(np.percentile(dt,95)),
            "max_dt_s":float(dt.max())}

def velocity_from_position(positions, ts_ns):
    p=np.asarray(positions,dtype=float)
    assert p.ndim==2 and p.shape[1]==3
    dt=np.diff(np.asarray(ts_ns,dtype=np.int64))/1e9
    assert len(dt)==len(p)-1 and np.all(dt>0)
    return np.linalg.norm(np.diff(p,axis=0)/dt[:,None],axis=1)

t=np.arange(101,dtype=np.int64)*50_000_000
p=np.stack([np.arange(101)*0.01,np.zeros(101),np.zeros(101)],axis=1)
print(verify_timestamps(t))
print("speed m/s", np.median(velocity_from_position(p,t))) # 0.2
~~~

**这段脚本只做基础体检，不是 SLAM 或策略训练代码。** 使用时进一步补充姿态角速度、mask 对应、有效区间、数据格式和独立真值对照。超阈值是否一定是错误，要结合工况，不能把真实快速动作全部过滤掉。

## 十、复现错误分级与失败退出条件

| 级别 | 失败例子 | 处理 |
| --- | --- | --- |
| P0 数据不可用 | 时间反转、视频和位姿完全错位、坐标系未知 | 拒绝进入训练，重新采集或修复 |
| P1 标注不可信 | 锚点不一致、长时间重定位跳跃、缺测被填充 | 保留视频，动作标签置无效 |
| P2 覆盖不足 | 遮挡造成某场景可用数据显著下降 | 加视角或修改采集 SOP |
| P3 部署风险 | 动作定义与机器人控制器不匹配 | 禁止实机直接执行，先做仿真和低速验证 |

## 十一、建议的阶段性交付与研究消融

| 里程碑 | 交付 | 必须记录的证据 |
| --- | --- | --- |
| M1 公开资料复核 | 论文参数逐条引用，源码/数据许可清单 | 原始链接、版本 commit、不可获取的依赖 |
| M2 一段原始数据 | 可重现的解码和同步分析 | 帧率、时间误差、标定数据 |
| M3 定位/动作重建 | 轨迹、地图来源、有效性可视化 | ATE/RPE（若有真值）、不可信区间 |
| M4 标准化训练样本 | schema、数据集、转换单元测试 | 维度、单位、坐标一致性 |
| M5 模型或工程闭环 | 固定数据/网络/评测协议的对照结果 | 成功率、失败归因、推理与执行延迟 |

**实验隔离原则**：每次只改变一种设计（多视角、硬同步、公共地图、被动标记、动作转换或质检），避免因为同时改变多个变量而无法判断算法真实收益。

## 十二、复现级别怎样判定？

- **L0 论文可读**：能解释输入、网络/定位算法、输出与局限。
- **L1 数据可复用**：公开原始数据能解析和转换，重建所需字段真实存在。
- **L2 组件可复现**：标定、时间同步、单帧几何等核心子模块可独立重现与测量。
- **L3 端到端工程可复现**：按固定版本、料号、固件与处理脚本持续采集，真实输出数据。
- **L4 论文指标可复现**：在相同评测协议与独立真值下接近论文性能。
- **L5 策略价值可复现**：证明数据在固定机器人闭环任务上带来可重复的提升。

**本文尚未实际执行完整系统搭建、训练与独立测量，不能标为 L3–L5。** 文件给出的预算、时间表和脚本是供实施的工程方案，不是已经完成的结果。
