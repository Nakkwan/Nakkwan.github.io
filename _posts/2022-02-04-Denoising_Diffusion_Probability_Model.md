---
title: DDPM(Denoising Diffusion Probability Model)
tags:
    - Deep Learning
    - Vision
    - Generative
    - Paper
---

Summary of DDPM <br>

<!--more-->

---

DDPM은 2020년 NeurIPS에 게재된 논문으로, Diffusion Model(DM)을 기반으로 한 generative model이다.<br>
Sampling에 반대되는 방향으로 DM(Gaussian noise를 추가하는 것을 data가 파괴(noise)가 될 때까지 하는 Markov chain)의 reverse를 학습하는 diffusion model이다. <br>

-   [DDPM ppt](https://github.com/Nakkwan/Nakkwan.github.io/blob/main/pdf/DDPM.pdf)<br>

---

##### Reference <br>

-   [Langevin Dynamics](https://towardsdatascience.com/langevin-dynamics-29bbb9407b47)<br>
-   [DDPM Blog](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html)<br>
-   [DDPM paper](https://arxiv.org/abs/2006.11239)<br>
-   [Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)<br>