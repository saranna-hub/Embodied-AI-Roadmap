---
title: 公式渲染检查页
sidebar:
  hidden: true
description: 用于验证 KaTeX 在桌面和移动浏览器中的渲染
---

此页面用于确认构建链路，不属于正式课程。

## 行内公式

Flow Matching 的时间变量 $t\in[0,1]$，动作噪声 $\epsilon\sim\mathcal N(0,I)$，损失 $\mathcal L_{\mathrm{FM}}$。

## 块级公式

$$
\mathcal L_{\mathrm{FM}}=\mathbb E_{A,\epsilon,t}\left[\left\|v_\theta(x_t,t,o)-(\epsilon-A)\right\|_2^2\right].
$$

## 矩阵与齐次变换

$$
T_{WC}=\begin{bmatrix}R_{WC}&p_{WC}\\0&1\end{bmatrix} \in SE(3).
$$

## 多行对齐

$$
\begin{aligned}
x_t &= (1-t)A+t\epsilon,\\
u_t &= \epsilon-A,\\
\hat A &= x_{0}.
\end{aligned}
$$

通过标准：页面不出现原始 LaTeX 反斜杠串，矩阵行列正确，手机窄屏公式可横向滚动，不遮挡正文。构建过程还会执行 `npm run validate:math` 检查 KaTeX 语法。
