# Embodied AI Roadmap

**具身智能学习手册**（中文优先）。整理机器人学习的理论、模型架构、数据工程和跨本体迁移方法，重点讲清楚输入输出、关键公式、训练与推理以及真正可验证的实验。

[阅读网站](https://saranna-hub.github.io/Embodied-AI-Roadmap/) · [内容建设进度](src/content/docs/status.md) · [学习路线](src/content/docs/start.md)

## 六个技术领域

1. **具身基础模型**：模仿学习、强化学习、VLA、世界模型。
2. **具身数据工程**：EGO、UMI、手套、采集、清洗和训练格式。
3. **感知与重建**：SLAM/VIO、深度、手部姿态、手物交互。
4. **Human-to-Robot**：动作转换、跨本体表征和任务迁移。
5. **控制与执行**：运动规划、双臂协调、力控和任务编排。
6. **仿真与评测**：MuJoCo、Isaac Sim、基准测试与真机验证。

当前集中补充前四个领域。**未完成的论文列在进度表中，不用自动生成的空文章充数。**

## 已编写的代表教程

| 分类 | 已发布的部分文章 |
| --- | --- |
| 具身基础模型 | [π₀](src/content/docs/01-foundations/vla/pi0.md)、[π₀-FAST](src/content/docs/01-foundations/vla/pi0-fast.md)、[ACT](src/content/docs/01-foundations/imitation/act.md)、[Diffusion Policy](src/content/docs/01-foundations/imitation/diffusion-policy.md)、[GR00T N1](src/content/docs/01-foundations/vla/gr00t-n1.md)、[RDT-1B](src/content/docs/01-foundations/vla/rdt-1b.md)、[Ego4WAM](src/content/docs/01-foundations/world-models/ego4wam.md) |
| 具身数据工程 | [UMI](src/content/docs/02-data-engineering/umi.md) |
| 感知与重建 | [HaMeR](src/content/docs/03-perception/hamer.md)、[HaWoR](src/content/docs/03-perception/hawor.md)、[MINT](src/content/docs/03-perception/mint.md) |
| Human-to-Robot | [Ego2Robot](src/content/docs/04-human-to-robot/ego2robot.md)、[UMI-Bridge](src/content/docs/04-human-to-robot/umi-bridge.md)、[HumanEgo](src/content/docs/04-human-to-robot/humanego.md) |

这些是**已撰写的文献/源码解读**，并非已在本站完成的训练复现。文章的实际深度仍在逐项完善，不能把有文章等同于已经覆盖所有实验细节。

## 网站如何生成

**唯一内容源：`src/content/docs/**/*.md(x)`。** Astro Starlight 将 Markdown/MDX 编译成网页；技术图存放于 `public/diagrams`；数学公式使用 LaTeX + remark-math + KaTeX，在构建阶段渲染和检查。

```bash
npm install
npm run validate:math
npm run build
npm run dev
```

GitHub Actions 在每次推送 `main` 时执行安装、公式校验、类型检查、构建和 Pages 部署。首次使用 GitHub Pages 需在仓库 Settings → Pages 中把 Source 设置为 GitHub Actions。构建是否成功应以仓库 Actions 结果为准。

早期手写 HTML、旧版 Markdown 仍留有部分仓库历史记录；正式技术正文统一从 `src/content/docs` 维护，不再同时维护第二份 HTML 正文。

## 内容与证据标准

- 论文结果明确写“作者实验”，不等于第三方复现。
- 厂商参数只是厂商声明，不当作独立标定或验收结果。
- 开源状态按论文、代码、权重、数据分开核验。
- 尚未实际执行的训练/推理实验标注为“建议实验”。
- 公开站点不发布私人复习或面试准备资料。

发现内容、公式或链接问题，欢迎通过 Issue / PR 反馈。
