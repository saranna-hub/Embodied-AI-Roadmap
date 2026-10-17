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
