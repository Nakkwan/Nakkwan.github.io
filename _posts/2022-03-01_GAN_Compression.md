---
title: GAN Compression(Efficient Architectures for Interactive Conditional GANs)
tags:
  - Deep Learning
  - Vision
  - Generative
  - Paper
---

GAN Compression: Efficient Architectures for Interactive Conditional GANs는 2020
년 CVPR에 게재된 논문으로, CGAN의 compression을 위한 method를 제시한 논문입니다
.<br> CGAN은 mobileNet과 같은 image recognition과 비교해서 계산량이 큽니다. 따라
서 CGAN에 대해 inference time과 model size를 줄이기 위한 방식을 제시합니다. <br>

<!--more-->

#### Introduction

GAN의 경우 human interactive한 영역에서 많이 활용되지만, edge device는 hardware
성능에 한계가 있기 때문에 많은 계산량을 필요로하는 model에 대해서는 bottleneck이
생깁니다. CycleGAN과 같은 generative model들은 계산량을 많이 필요로 합니다. 따라
서 GAN compression이 제안됩니다. <br>

<p>
<center><img src="/images/GAN_compression/Compression_computation_magnitude.jpg" width="400"></center>
<center><em>Fig n.</em></center>
</p>

Generative model을 compression 하는데는 2가지 근본적인 어려움이 있습니다. <br>

1. Unstability of GAN training (especially unpaired)<br> Pseudo pair를 만들어 훈
   련시킵니다. <br>

2. Architecture가 recognition CNN과 다르다. <br> 중간 representation만 teacher
   model에서 student model로 transfer합니다. <br>

그리고 fewer cost model을 찾기 위해 NAS(Network Architecture Search)를 사용합니
다.

#### Related work

- Conditional GAN <br> GAN(Generative Adversarial Networks)은 photo-realistic
  image를 합성하는데 좋은 성능을 보입니다. Conditioanl GAN은 image, label, text
  와 같은 다양한 conditional input을 주어 이미지 합성을 제어할 수 있도록 합니다.
  고해상도의 photo-realistic image 생성은 많은 계산량을 필요로 합니다. 이로 인해
  제한된 계산 리소스가 주어진 edge device에 이러한 모델을 배포하기가 어렵습니다.
  따라서 interactive application을 위한 효율적인 CGAN architecture에 중점을 둡니
  다. <br>

- Model Acceleration <br> Network model에서 필요하지 않은(중복된) 부분을 없애기
  위해, network connection이나 weight에 대한 pruning을 할 수 있다. <br>
  AMC(AutoML for Model Compression)은

- Knowledge distillation

- NAS(Neural Architecture Search)

#### Method

CGAN은 source domain $$X$$에서 target domain $$Y$$로의 mapping $$G$$를 훈련시킵
니다. CGAN의 training data는 paired와 unpaired 두가지 방식이 있기 때문에 많은
model에서 paired와 unpaired를 구별하지 않고 objective function을 구성합니다.
General-purpose compression에서는 teacher structure가 어떤 방식으로 training 됐
는지에 관계없이 model compression이 가능하도록 paired와 unpaired를 통합했습니다.
<br>

<p>
<center><img src="/images/GAN_compression/Compression_framework.jpg" width="600"></center>
<center><em>Fig n.</em></center>
</p>

Origin teacher generator를 $$G'$$라고 가정합니다. <br> Unpaired data의 경우
compression과정에서 $$G'$$로 generate된 이미지를 student generator $$G$$의
pseudo GT로 사용합니다. <br>

$$
\begin{align}
\mathcal{L}_{recon} =
\begin{cases}
\mathbb{E}_{X,Y}\parallel G(X)-Y\parallel, & \text{if paired CGAN} \\
\mathbb{E}_{X}\parallel G(X)-G'(X)\parallel, & \text{if unpaired CGAN}
\end{cases}
\end{align}
$$

