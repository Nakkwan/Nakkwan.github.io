---
layout: default
title: Normalize Flow
nav_order: 5
parent: Paper Review
has_children: true
permalink: /docs/paper_review/normalize_flow
---

# Neural ODE
{: .no_toc}
Review of NLP papers

1. TOC
{:toc}


# Normalize Flow 개념
## Introduction
생성 모델에는 대표적으로 GAN, VAE, Normalize Flow, Diffusion이 있다. 일반적으로 생성 모델은 임의의 분포 `z`로부터 데이터 분포 `x`를 생성하도록 학습한다. 즉, `z`에서 `x`로 매핑하는 함수를 학습한다고 볼 수 있다.

`z`(prior)는  Gaussian distribution과 같은 쉬운 함수로 가정하기 때문에 직접적으로 probability density를 계산할 수 있지만, 고차원 데이터 `x`(posterier)의 probability density를 계산하기는 쉽지 않다. 따라서, GAN은 generator와 discriminator를 학습시키며 Jensen-Shannon divergencefmf 최소화하는 방식으로 생성 모델을 학습한다. VAE는 Variance Inference에서 ELBO를 최대화하는 방식으로 implicit하게 probability density를 학습하고, diffusion model은 각 noise sample에 대해 kl-divergence를 최대화하는 방향(VAE랑 비슷함)으로 학습을 implicit하게 진행한다.

반면, Normalize flow는 NLL을 통해 학습을 진행하여, explicit하게 probability density를 계산하지만, 이를 위해 reversable한 trasformation을 사용하기 때문에 구조에 한계가 있다. 

|   -   |      GAN      |   VAE   | Normalize Flow | Diffusion |
| :---: | :-----------: | :-----: | :------------: | :-------: |
| x->z  | Discriminator | Encoder |      $$f$$       |  Forward  |
| z->x  |   Generator   | Decoder |    $$f^{-1}$$    |  Reverse  |
| Train |    MinMax     |  ELBO   |      NLL       |   KL-D    |


## Normalize Flow
생성 모델에서 데이터를 생성한다는 것은 데이터의 확률 밀도가 높은 분포를 학습하고, 해당 분포에서 샘플링을 수행하여 데이터를 생성하는 것이다.
즉, 기본이 되는 Gaussian distribution에서 data distribution으로의 mapping을 neural network가 학습하는 것이다.

하지만, $$p(z)\rightarrow p(x)$$를 수행하려면, $$p(z)$$뿐만 아니라 $$p(x)$$도 알아야 한다. 하지만 prior($$=p(z)$$)만 알고 있고, poterior($$=p(x	\mid z)$$, $$p(z)$$에 대한 $$p(x)$$)에 대해 모르기 때문에 이를 구하기 위한 여러 방법들이 사용된다. 

Normalize flow는 역변환이 가능한 함수들의 sequence($$f_1, f_2, \dots f_N$$)를 이용하여 $$z \leftrightarrow x$$의 분포 변환을 수행한다. 
#### 변수 변환
임의의 random variable $$z\sim \pi(z)$$ 일 때, data distribution은 $$x\sim p(x)$$라고 가정한다.
여기서, $$p(x)$$를 구해야 NLL을 최소화하도록 학습을 수행할 수 있다.

$$
\begin{align}
\int p(x)dx &= \int \pi(z)dx=1 \\ p(x)&=\pi(z)\left\vert \frac{dz}{dx}\right\vert=\pi(f^{-1}(x))\left\vert \frac{df^{-1}(x)}{dx}\right\vert \\ 
&=\pi(f^{-1}(x))\left\vert (f^{-1})'(x)\right\vert \\ &\simeq\pi(f^{-1}(x))\left\vert \det\frac{df^{-1}}{dx} \right\vert \\ 
&=\pi(f^{-1}(x))\left\vert \det J \right\vert \\ &=\pi(z)\left\vert \det J \right\vert 
\end{align}
$$ 

$$\rightarrow \log p(x) = \log\pi(z) + \log{(\det J)}$$

위 수식을 통해 NLL을 계산하여 neural network를 학습할 수 있지만, $$f$$가 invertable 해야하기 때문에 복잡한 함수는 쓸 수 없다는 문제가 있다. 따라서, normalize flow에서는 간단한 변환 $$f$$의 sequence를 통해 점진적으로 복잡한 분포를 학습하여 $$x\rightarrow z$$ mapping을 수행한다. 
$$\rightarrow \log p(x) = \log\pi(z) + \log{(\det J)} $$

위와 같이 기본적인 normalize flow는 explicit하게 probability density를 학습한다. 하지만, Jacobian $$J$$가 계산복잡도가 $$\mathcal{O}(n^3)$$ 라는 문제가 있어, 고차원 데이터에는 적용하기 힘든 문제가 있다. 
따라서, 후속 연구들에서 
- 계산 계산량을 줄이고
- invertable한 linear transform을 적용하며
복잡한 분포에도 적용하기 위한 연구를 수행한다. 



## [NICE](https://arxiv.org/abs/1410.8516)
Normalize flow는 $$f$$가 invertable해야하고, jacobian의 determinant를 구하기 쉽지 않다는 문제가 있었다. 
NICE는 이런 문제를 Triangular Matrix와 Coupling Layer로 해결한 논문이다. 

#### Triangular Matrix



## [RealNVP](https://arxiv.org/abs/1605.08803)


## [Glow](https://arxiv.org/abs/1807.03039)


## [IAF](https://arxiv.org/abs/1606.04934)


## [MAF](https://arxiv.org/abs/1705.07057)


