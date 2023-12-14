---
layout: default
title: DDPM
nav_order: 1
parent: Basic
grand_parent: Diffusion
permalink: /docs/computer_vision/diffusion/basic/ddpm_2023_07_10
math: katex
---

# **Denoising Diffusion Probabilistic Models**
{: .no_toc}
[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

Table of contents
{: .text-delta }
1. TOC
{:toc}

Diffusion probabilistic models 과 denoising score matching with Langevin dynamics 사이의 연관성에 따라 설계된 Variational bound로 훈련됨 <br>
Autoregressive decoding의 일반화로 해석되는 progressive lossy decompression으로 볼 수 있음

Target distribution에서 noise를 추가하는 **forward process**와 noise에서 target distribution으로 복원하는 **backward process**의 distribution을 맞추는 것으로 훈련됨

## **Introduction**
DM은 VI를 이용한 parameterized Markov chain<br>
Signal을 파괴(Noise)시키는 Markov chain인 diffusion process의 reverse를 학습<br>
$$\rightarrow$$ Noise가 작으면 transition을 conditional gaussian으로 설정 가능 $$\mathcal{N}(x|\mu, \Sigma)$$
<center><img src="/assets/images/cv/diffusion/basic/ddpm_fig1.jpg" width="80%" alt="Figure 1"></center>

퀄리티는 좋지만, 다른 타입의 모델에 비해, NLL(lossless codelength)가 좋지 않음<br>
이를 lossless codelength를 인지할 수 없는 이미지 세부 정보를 설명하는 데 사용<br>
**$$\rightarrow$$ lossy compression으로 분석하여 AR과 비슷한 progressive decoding이라는 것을 보임**

## **Background**
DM은 latent variable models (Observable variable을 latent variable로 연결하는 statistic model)

$$
p_\theta(x_0):= \int p_\theta(x_{0:T})dx_{1:T}, \quad \mathrm{where\;} x_1,\cdots, x_T \mathrm{\;are\;latent\;of\;data\;} x_0 \sim q(x_0)
$$

$$p_\theta(x_{0:T})$$는 reverse process로 불리고, $$p(x_T)=\mathcal{N}(x_T;0, I)$$부터 시작한 gaussian trasition의 MC

$$
\begin{gather}
p_\theta(x_0):= p(x_T)\sum^T_{t=1}p_\theta(x_{t-1}\mid x_t), \quad p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t), \sum_\theta(x_t,t))
\end{gather}
$$

Probabilistic distribution을 구하기 위한 bayes’s rule의 posterior ()를 구하기 어렵기 때문에, 근사하기 위하여 사용되는 approximate가 noise를 추가해나가는 Markov chaine에 고정되어 있음

$$
\begin{gather}
q(x_{1:T}\mid x_0):= \sum^T_{t=1}q_(x_t\mid x_{t-1}), \quad q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
\end{gather}
$$

훈련은 NLL에 대한 variational bound를 최적화하는 것으로 수행됨

$$
\begin{gather}
    \mathbb{E}\left[-\log p_\theta(x_0)\right] \le \mathbb{E}\left[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right] = \mathbb{E}_q\left[-\log p(x_T)-\sum_{t\ge 1}\log\frac{p_\theta(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})}\right] =: L
\end{gather}
$$

또한 $$\beta_t$$가 충분히 작을 때, forward와 backward는 같은 함수 형식을 가지기 때문에 표현력이 보장되고, $$\beta_t$$는 학습도 가능하고, hyperparam으로 설정할 수도 있음

임의의 t에 대해서 $$x_t$$는 closed form이다 ($$x_0$$가 주어졌을 때)

$$
\begin{gather}
    q(x_t\mid x_0) = \mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
\end{gather}
$$

따라서, 식 (3)에서 random한 t를 적용하여, 효과적으로 학습가능

식 (3)을 다시 적으면,

$$
\begin{gather}
    L := \mathbb{E}_q\left[D_{KL}(q(x_T\mid x_0) \parallel p(x_T)) + \sum_{t>1}D_{KL}(q(x_{t-1}\mid x_t,x_0)\parallel p_\theta(x_(t-1)\mid x_t)) - \log p)\theta(x_0\mid x_1) \right]
\end{gather}
$$

Forward와 reverse를 KL divergence를 이용하여, posterior를 직접 비교 

$$
\begin{align}
    q(x_{t-1}\mid x_t,x_0) &= \mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t\mid x_0),\tilde{\beta}_t I), \\
    \text{where} \quad \tilde{\mu}_t(x_t,x_0) &:= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha}_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \quad \text{and} \quad \tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t 
\end{align}
$$

두 변수 다 gaussian 형식으로 나오기 때문에 결과의 변동성이 큰 MC 대신 Rao-Blackwellized fashion를 사용할 수 있음 (?)

## **Diffusion models and denoising autoencoders**
DM은 자유도가 높음 <br>
DM과 DSM 사이의 단순한 연결(섹션 3.2)과 DM을 위한 weighted variational bound objective (섹션 3.4)를 명시적을 설명

### **Forward process and $$L_T$$**
$$\beta_t$$를 constant로 고정 $$\rightarrow$$ 식 (5)에서 q는 학습가능한 변수가 X <br>
$$\rightarrow$$ $$L_T$$는 무시됨 (너무 작은 constant 값이기 때문)

### **Reverse process and $$L_{1:T-1}$$**
Reverse process의 경우 다음과 같이 표현 가능

$$
p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)), \quad \mathrm{for\;} 1<t \le T
$$

$$\Sigma_\theta(x_t,t)=\sigma^2_tI$$로 표현되는데, $$\sigma^2_tI$$가 $$\beta_t$$일 때와, $$\tilde{\beta_t}$$가 차이는 없음 <br>
$$\beta_t$$는 $$x_0$$가 가우시안을 따를 때, 최적이고, <br>
$$\tilde{\beta_t}$$는 $$x_0$$가 한 점일 때, 최적 <br>

$$\mu_\theta(x_t,t)$$의 경우, $$p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\sigma^2_tI)$$일 때, 아래로 표현 가능 (KLD 확인하기)

