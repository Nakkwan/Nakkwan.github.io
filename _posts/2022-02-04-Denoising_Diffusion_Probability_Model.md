---
title: DDPM(Denoising Diffusion Probability Model)
tags:
    - Deep Learning
    - Vision
    - Generative
    - Paper
---

DDPM은 2020년 NeurIPS에 게재된 논문으로, Diffusion Model(DM)을 기반으로 한 generative model입니다.<br>
Sampling에 반대되는 방향으로 DM(Gaussian noise를 추가하는 것을 data가 파괴(noise)가 될 때까지 하는 Markov chain)의 reverse를 학습하는 diffusion model입니다. <br>

<!--more-->

#### Introduction

Generative 분야에는 GAN, VAE 등이 있고, 최근에는 energy-based model과 score mathing model도 GAN과 비교될만큼 큰 발전을 이루었습니다. <br>
논문에서는 finite time 이후와 맞는 sample을 생성시키기 위해 variational inference를 사용해서 Markov chain을 훈련시킨 Diffusion Model(이하 DM)을 제안합니다. <br>

-  Markov Chain: 미래의 조건부 확률 분포가 과거에 독립적이고 현재에만 의존한다는 Markov property를 따르는 discrete time stochastic process <br>
-  VI(Variational Inference): $$p(x \mid z) = \frac{p(z \mid x)p(z)}{p(x)}$$에서 posterior를 찾기 힘들 때, 더 쉬운 확률 분포 $$q(z)$$로 근사하기 위한 방법 <br>

DM은 input으로 들어온 이미지를 이미지가 파괴(noise)가 될 때까지 noise를 추가하는 difussion process(forward process)의 반대방향을 학습하도록 구성됩니다. Diffusion process에서 각 time step에서 더해지는 noise의 variance가 충분히 작다면, 이 과정은 Neural Network(이하 NN)로 parameterize할 수 있습니다. Diffusion이 진행될수록 distinguish한 특성은 사라지고 isotropic gaussian distribution으로 변합니다. 즉, 마지막 step에서 이미지는 $$\mathcal{N}(0, I)$$를 따르게 됩니다. <br>

<p>
<center><img src="/images/DDPM/DDPM_diffusion_graph.jpg" width="600"></center>
<center><em>Fig n.</em></center>
</p>

#### Background

DM은 $$p_{\theta}(x_{0}) := \int p_{\theta}(x_{0:T})dx_{1:T}$$의 형태로, $$x_{1},\ldots, x_{T} \sim q(x_{0})$$인 모든 latent가 $x_{0}$와 같은 dimension을 가지고 있는 latent variable model입니다. Image에 gaussian noise를 더해가며 diffusion process가 진행되지만, 이 과정에서 latent의 dimension은 변하지 않습니다. <br>
$x_{0}$로부터 $x_{T}$(destroyed image)까지의 diffusion process인 forward process는 다음과 같이 표현됩니다. <br>

$$
\begin{align}
q(x_{1:T} \mid x_{0}) := \prod_{t=1}^T q(x_{t} \mid x_{t-1}), \quad q(x_{t} \mid x_{t-1}) := \mathcal{N}(x_{t}; \sqrt{1-\beta_{t}}x_{t-1, \beta_{t}I}) \\
\end{align}
$$

DM은 posterior인 $$q(x_{1:T} \mid x_{0})$$을 inference 하도록 학습됩니다. <br>

$x_{T}$로부터 $x_{0}$(generate image)까지의 process인 reverse process는 다음과 같이 표현됩니다. <br>

$$
\begin{align}
p_{\theta}(x_{0:T}) := p(x_{T})\prod_{t=1}^T r_{\theta}(x_{t-1} \mid x_{t}), \quad p_{\theta}(x_{t-1} \mid x_{t}) := \mathcal{N}(x_{t-1}; \mu_{\theta}(x_{t},t),\mathsf{\Sigma}_{\theta}(x_{t},t)) \\
\end{align}
$$

Training은 NLL에 대한 variational bound를 optimize함으로써 수행됩니다.
$$
\begin{align}
\mathbb{E}[-\log p_{\theta}(x_{0})] \le D_{KL}(q \parallel p) &= \mathbb{E}_{q}[\log \frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})}] = \mathbb{E}_{q}[-\log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_{0})}] \\
&= \mathbb{E}_{q}[-\log p(x_{T})-\sum_{t \ge 1}\log \frac{p_{\theta}(x_{t-1} \mid x_{t})}{q(x_{t} \mid x_{t-1})}] \\
&:=\mathcal{L}
\end{align}
$$


---


-   [DDPM ppt](https://github.com/Nakkwan/Nakkwan.github.io/blob/main/pdf/DDPM.pdf)<br>

---

##### Reference <br>

-   [Langevin Dynamics](https://towardsdatascience.com/langevin-dynamics-29bbb9407b47)<br>
-   [DDPM Blog](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html)<br>
-   [DDPM paper](https://arxiv.org/abs/2006.11239)<br>
-   [Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)<br>
