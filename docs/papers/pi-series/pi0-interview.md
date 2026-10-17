# π₀ 架构面试题：从能讲原理到能解释代码

> 这是 [π₀ 深度解读](./pi0-deep-dive.md) 的补充篇，基于 Physical Intelligence 的原始论文和官方 openpi 主线实现。代码会演进，本文引用函数名而不声称所有旧 checkpoint 都与最新主线完全一致。问题按面试追问深度组织；答案区分论文设计、当前代码与工程推导。

## 一、第一轮：五分钟内说明架构

**问题 1：请完整介绍 π₀ 的模型结构，不能只说 VLM + Flow Matching。**

参考回答：π₀ 是使用视觉语言上下文条件的连续机器人动作生成模型。输入多相机 RGB、文本指令和机器人 proprioceptive state；PaliGemma 处理图像和文本形成 prefix tokens，另一组 Gemma action expert 的投影层接收 state、带噪的 action chunk 与 flow 时间。两个专家通过联合注意力共享上下文，动作端输出每个未来动作步、每个维度的速度向量，训练用条件 flow matching 的平方误差；推理时先缓存不变的观察 prefix，然后数值积分，将高斯噪声转成动作块。

关键追问：PaliGemma 内部是视觉编码器 + 语言 Transformer；action expert 不是传统 MoE 的路由专家。跨专家联合注意力不等于简单 cross-attention 一层拼接。

**问题 2：输入输出形状与关键超参是什么？**

官方 openpi Pi0Config 的默认 `action_dim=32`、`action_horizon=50`，action expert 默认 `gemma_300m`，PaliGemma 语言变体默认 `gemma_2b`。训练动作块示意 `[B,50,32]`。这些是当前开源实现默认上限/约定，不表示任意机器人都控制32个真实自由度。要说清楚机器人特定的 action schema、padding、归一化和频率。

追问：如果动作采样迭代数是10，是否代表只有10帧动作？不是；积分10次，每次更新完整50帧动作张量。

## 二、第二轮：逐模块问设计

**问题 3：为什么使用 PaliGemma，而不是完全从头训练视觉语言骨干？**

预训练视觉语言主干提供场景、物体、任务语义的基础表征，机器人专属训练再学习观测到控制的映射。代价是视觉-语言分布与机器人动作控制所需细粒度信息未必一致，模型仍可能需要下游数据和视觉适配。回答时不要宣称论文证明“PaliGemma一定优于所有VLM”。

**问题 4：Action expert 到底输入什么？State 进入哪？**

在 openpi 的 `embed_prefix` 中，图像经过视觉编码器，文本经过语言 embedding；在原始 π₀ 路径的 `embed_suffix` 中，`state_proj` 把状态映射为一个 state token；`action_in_proj` 把带噪动作每一步映射为一个 action token；时间用正余弦嵌入，与 action embedding 拼接，再经 MLP 融合。π₀.₅ 的 state 注入方式及时间 conditioning 不同，不能混写。

**问题 5：Attention Mask 怎么设计？请画出可以看到谁。**

Prefix（图像和语言）内部可双向交互，不读取 suffix；后面的动作 token 可以读取 prefix，并在 action chunk 内交换信息。代码通过 `mask_ar` 的累计段规则生成 mask，不能机械理解成“整个模型都是自回归 causal mask”。

面试追问：为什么不能让 prefix 读取带噪动作？否则观测的 KV cache 依赖每一次流迭代的动作噪声，没法复用；同时破坏前缀作为静态条件的结构设计。注：这是从代码设计出发的工程解释。

**问题 6：Flow 时间如何进入网络？**

原始 π₀ 代码用时间 sinusoidal embedding 与 noisy-action embedding 拼接后过 MLP；π₀.₅ 采用 adaRMSNorm 相关注入方式。要能定位 `embed_suffix` 和 Gemma 专家配置，不能将后续版本特性当成 π₀ 原版。

## 三、第三轮：训练与数学

**问题 7：写出 flow matching 的训练目标，解释网络预测量。**

以 openpi 代码的时间约定为准：\(x_t=t\epsilon+(1-t)A\)，\(u_t=\epsilon-A\)，\(L=\mathbb E\|v_\theta(x_t,t,o)-u_t\|^2\)。(t=1)是噪声、(t=0)是数据；论文可能采用相反的记号。网络预测的是**速度场**，不是直接回归动作本身，也不是图像光流。

问：为什么训练时不需要迭代求解 ODE？训练时先直接构造中间点和对应向量场，做回归；迭代是推理阶段恢复目标样本时才使用。

**问题 8：反向传播经过哪些模块？哪些参数冻结？**

训练 loss 由动作速度头输出，经 action expert、联合 attention、观测侧条件表征回传到所有被设为 trainable 的参数。是否冻结视觉侧、使用 LoRA 或只微调动作侧，要看对应训练配置，不应只凭结构图推断。检查训练脚本参数树和 optimizer mask，不能把网络分支存在与否等同于更新与否。

**问题 9：Flow Matching 与 Diffusion Policy 关系？**

两者都可从随机变量生成条件动作分布，时间连续变换与噪声相关训练目标密切相关，但具体路径、参数化、采样求解器及损失实现不同。不能简单说“Flow Matching 就是光流”，也不能说“π₀ 逐步生成50个动作token”。

## 四、第四轮：推理、效率与部署

**问题 10：π₀ 推理为什么能用 KV cache？**

图像和指令在一次动作采样的积分过程中保持不变，先对 prefix 做一次 forward 缓存 K/V。之后每个 flow step 只更新 action suffix，并读取固定前缀缓存。若每个求解步重复图像编码，延迟将明显增加。需看 `sample_actions` 而不是凭直觉猜缓存方式。

**问题 11：为什么一次输出 action chunk？弊端是什么？**

分块能建模连续时间上相互依赖的动作并减少每个控制步单独规划的开销；但 chunk 开环执行越长，越容易在意外接触、目标移动时积累误差。应讨论重规划频率、执行窗口、时延、动作平滑、稳定性和安全控制。

**问题 12：跨机器人 action_dim=32，真的统一动作空间吗？**

只是可兼容多种动作表示的建模容器，不等于所有关节/末端指令自动具有共同物理语义。具体需要确定坐标系、绝对/相对动作、旋转表示、控制频率、归一化、无效维度 mask、重定向和控制器接口。

## 五、第五轮：系统设计与追问

**问题 13：要微调到轮式双臂，最容易踩的三个坑？**

(1) action mapping 不一致：单位、坐标系、旋转、gripper开合语义及状态动作时延；(2) 视觉数据域差：head/wrist相机语义、标定和遮挡；(3) 闭环接口：模型输出频率、底盘控制和接触安全约束。建议先固定底盘，把双臂取放作为最小任务；之后逐步增加接触策略。

**问题 14：有 EGO + 触觉手套数据，能直接用 π₀ 的动作损失训练吗？**

不能直接假设可以。普通 EGO 没有机器人可执行动作标签；手套关节与接触压力也不等价于目标手/夹爪动作。至少需要设计人类动作表征到机器人动作的映射并验证，或先把手套用于接触阶段、数据选择、辅助监督，再用少量机器人示范对齐。

**问题 15：如何证明某个新教师/触觉标签确实有效？**

固定机器人示范集、主干、训练预算与评测条件，比较：(a) robot-only；(b) robot+EGO 视频辅助；(c) robot+EGO+手套接触阶段标签；分别测 ID/OOD 成功率、接触阶段错误、到位判定、任务耗时以及失败恢复。若只有训练 loss 降、真机成功率不升，不足以支持方法在闭环上有效。

## 面试自测标准

- **基础合格**：能在5分钟内画出prefix/suffix模型结构，正确写出FM目标及积分方向。
- **算法工程师合格**：能解释attention mask、KV cache、state/time注入、训练梯度和 action mapping。
- **高级工程师合格**：能识别 π₀ vs π₀.₅ 版本差异，提出切实可行的下游微调实验和消融，并解释真实失败原因。
- **深入追问**：代码中 `make_attn_mask`、`embed_prefix`、`embed_suffix`、`compute_loss`、`sample_actions` 的输入输出、mask形状、更新参数范围。

## 原始来源

- [π₀ 论文](https://arxiv.org/abs/2410.24164)
- [openpi JAX Pi0](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)
- [openpi PyTorch Pi0](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py)
- [openpi Pi0Config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py)
