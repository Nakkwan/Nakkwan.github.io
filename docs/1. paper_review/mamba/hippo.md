---
layout: default
title: HIPPO
nav_order: "2024.02.11"
parent: Mamba
grand_parent: Paper Review
permalink: /docs/paper_review/mamba/hippo_2024_02_11
math: katex
---

# **HIPPO**
{: .no_toc}
[HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://proceedings.neurips.cc/paper_files/paper/2020/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf)

Table of contents
{: .text-delta }
1. TOC
{:toc}


### **TODO**: Update detailed study log of this paper...
{: .no_toc}


Sequence에서 어려운 점은 seq가 길어질수록 누적된 history를 표현하기 힘들다는 점이다. HIPPO는 polynomial base로 continuous, discrete time sequence를 projection하여 online compression을 수행한다. <br>
시간 t에 대해 과거 시간의 중요도가 측정되면, online function approximation을 수행한다. 저자는 LMU (Legendre Memory Unit)의 미분값을 도출하고, GRU와 같은 gate mechanism을 generalization한다. <br>
즉, time scale에서 이전 history를 기억하기 위해 시간에 따라 확장되는 memory update mechanism HIPPO-Legs를 제안한다. <br>
$$\rightarrow$$ time robust, bounded gradient, fast update <br>

## **Intorudction**
Language modeling, speech recognition, video processing와 같은 sequential data의 modeling 및 학습은 long-term에 대한 이전 step의 memory가 가장 중요한 측면이다. (Online update에 대해 bounded storage를 사용하여 전체 history representation을 학습) <br>
RNN과 같은 memory unit을 통해 전체 history에 대한 information을 저장하는 방법은 vanishing gradient 문제가 생길 수 있다. LSTM과 GRU같은 gate나 LMU, Fourier Recurrent Unit과 같은 higher-order frequency도 unified understanding of memory는 도전적인 과제다.
또한 기존에는 전체 길이나 시간대에 대한 사전 제한이 있고 distribution shift가 발생하는 상황에선 문제가 될 수 있었다. <br>
저나는 time scale에 대한 prior와 length에 대한 dependency가 없고 memory mechanism에 대해 이론적이며 기존 방법에 대해 unified method를 설계했다. 



<details markdown="block">
<summary><b>OP (Orthogonal Polynomial)</b></summary>
orthogonal polynomial sequence는 sequence내의 두 polynomial이 inner product 내에서 서로 orthogonal한 것을 의미한다. 많이 사용되는 orthogonal polynomial로는 **Hermite, Laguerre, Jacobi, Gegenbauer, Chebyshev, Legendre polynomial** 등이 있다.  <br>
OP는 특정 measure (weight) $$\mu_t$$에 대해 두 polynomial $$P_m, P_n$$의 적분이 다음과 같은 관계일 때 성립한다. <br>

$$
\begin{align}
\mathrm{If} \; \deg P_n = n, \left\langle P_m, P_n \right\rangle _{\mu(x)} &= \int P_m(x)P_n(x)d\mu(x), \\
\left\langle P_m, P_n \right\rangle _{\mu(x)} &= 0 \;\mathrm{for}\; m \neq n    
\end{align} 
$$

각 basis $$P_n (n=0, 1, 2, \cdots)$$은 Gram-Schmidt로 구할 수 있다.

**Recurrence relation**
polynomials $$P_n$$은 $$A_n$$이 0이 아닐 때, 다음과 같은 recurrence relation을 만족한다. <br>

$$
\begin{align}
P_n(x)=(A_nx+B_n)P_{n-1}(x)+C_nP_{n-2}(x)
\end{align} 
$$

**Reconstruction**

$$
\begin{align}
\sum^{N-1}_{i=0} c_iP_i(x) / \lVert P_i \rVert^2_\mu, \quad \mathrm{where} \; c_i = \left\langle f, P_i \right\rangle _\mu = \int f(x)P_i(x)d\mu(x)
\end{align} 
$$


{: .note-title}
> Legendre Polynomial
>  
> 

</details>

<details markdown="block">
<summary><b>LMU</b></summary>

</details>


## **Reference**
1. [Wikipedia: Orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials)
2. 