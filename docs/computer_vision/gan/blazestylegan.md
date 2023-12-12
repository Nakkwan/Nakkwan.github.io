---
layout: default
title: BlazeStyleGAN
nav_order: "2023.12.10"
parent: GAN
grand_parent: Computer Vision
permalink: /docs/computer_vision/gan/blazestylegan_2023_12_11
---

# **BlazeStyleGAN**
{: .no_toc}
[BlazeStyleGAN: A Real-Time On-Device StyleGAN](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Jia_BlazeStyleGAN_A_Real-Time_On-Device_StyleGAN_CVPRW_2023_paper.pdf)

Table of contents
{: .text-delta }
1. TOC
{:toc}

SmartPhone에서 real-time으로 동작하기 위한 방법 <br>
Generator의 각 level에서 feature를 RGB로 바꾸는 auxiliary head와 inference에서 마지막 하나만 유지하여 효율적인 synthesis network를 디자인 <br
auxiliary head에서 multi-stage perceptual loss를 통해 distillation을 향상시킴 <br
student와 discriminator에 adversarial loss <br
SmartPhone의 GPU에서 real-time inference가 가능했음 <br

## **Introduction**
MobileStyleGAN은 intelCPU에서 큰 속도 향상을 보였지만, real-time이라고 보긴 힘듦 <br>
GAN compression, knowledge distillation, pruning 등은 paired data가 필요한 방법이었고, unpaired에 대해서 sub-optimal solution만 존재함

본 논문에서는 modulated convolution과 feature-to-RGB module이 inference time에 큰 비중을 차지함을 재검토함 <br>
$$\rightarrow$$ Modulated convolution를 simplify하고 auxiliary head를 통한 효율적인 UpToRGB 설계 <br>
$$\rightarrow$$ StyleGAN2가 teacher, BlazeStyleGAN가 student (multi-scale perceptual loss)

Teacher model과 BlazeStyleGAN이 비슷한 quality를 보임

{: .highlight-title}
> Contributions:
> 
> - auxiliary UpToRGB head 제안, 추론 시 마지막 head만 실행하여 mobile-friendly <br>
> - multi-scale perceptual loss와 adversarial loss를 통한 teacher에서의 artifact 억제 <br>
> - 고품질 이미지 생성을 유지하면서 스마트폰에서 real-time을 달성

## **Related Work**
이전의 GAN 압축 연구는 저해상도 이미지에 초점을 맞춤 <br>
$$\rightarrow$$ MSE 손실을 사용하여 student가 동일한 latent가 주어졌을 때 teacher와 유사한 이미지를 생성하도록 훈련

StyleGAN의 등장 이후 고해상도 이미지 합성을 위한 StyleGAN 압축에 많은 관심 <br>
1. Conetnt-aware GAN Compression (CAGAN)은 channel pruning과 distillation을 unconditional GAN에 맞게 조정하고, pruning과 distillation을 위해 content-of-interest mask 사용 <br>
$$\rightarrow$$ 하지만 student가 teacher의 주요 구조를 이어받아야 한다는 제약이 있음

2. Xu 등은 student와 teacher 네트워크가 다를 때 발생하는 output discrepancy가 CAGAN의 성능을 제한한다는 것을 발견 <br>
$$\rightarrow$$ 문제를 해결하기 위한 초기화 전략을 제안

3. MobileStyleGAN은 frequency-based의 이미지 표현을 사용하고 wavelet transform을 student의 예측 대상으로 사용 <br>
$$\rightarrow$$ frequency-based image representation은 생성 이미지에서 디테일을 놓치는 경향이 있음 <br>
$$\rightarrow$$ 모델의 효율성은 실시간 성능을 달성하기 위해 더욱 개선될 수 있음

## **Model Architecture**
StyleGAN에는 크게 **mapping network**와 **synthesis network**로 구성되어 있음

### **Mapping network**
MLP로 설계되어 input latent vector z를 latent space W로 mapping하여, synthesis network의 각 convolution에서 AdaIN에 style input으로 사용

### **Synthesis network**
여러 convolution layer를 포함하고 있으며, style input과 noise로부터 이미지를 생성 <br>
$$\rightarrow$$ 모델 parameter의 대부분을 차지

**Synthesis layer**는 convolution block의 stack으로 구성
- 3×3 convolution
- upsampling
- AdaIN를 포함

각 layer에 multi-stage perceptual loss를 위한 ToRGB block이 추가되어 있음

**MobileStyleGAN**는 MobileNet과 비슷하게, 
- DWModulatedConv를 제안함
- 마지막 layer에서 single frequency domain transformation을 통해 ToRGB
- MobileStyleGAN은 wavelet transform을 통해 high-frequency feature를 향상시킬 수 있다고 주장

저자는 단일 ToRGB 레이어를 사용하는 것이 **fine feature를 잃어버릴 수 있다는 것을 발견**
{: .important-title}
> 
> 따라서 **새로운 Single ToRGB layer(=`UpToRGB`)를 설계**하여 효율적인 Synthesis layer 제안

<center><img src="/assets/images/cv/gan/blazestylegan_fig1.jpg" width="70%" alt="Figure 1"></center>

- 각 Synthesis block은 UpToRGB를 통해 latent feature를 RGB 이미지로 upsampling하고 변환 <br>
- Synthesis block의 복잡성을 줄이기 위해 BlazeStyleGAN의 각 synthesis block에서 latent feature resolution을 teacher의 1/4로 downsampling <br>
- teacher의 output resolution과 일치시키기 위해, 각 feature map은 UpToRGB head에서 목표 resolution로 추가 upsampling <br>
- 각 블록에서 전체 해상도 feature map을 처리하는 것은 UpToRGB head뿐이기 때문에, synthesis block에서 전체 해상도 feature map을 처리하는 것에 비해 복잡성이 크게 감소 <br>
- auxiliary UpToRGB head에서 출력하는 RGB 이미지는 multi-scale perceptual loss를 계산하는데 사용되는 coarse-to-fine 피라미드를 구성

## **Distillation**
Teacher 모델인 StyleGAN으로부터 distillation하여 student 모델인 BlazeStyleGAN을 훈련 <br>
$$\rightarrow$$ multi-scale perceptual loss와 adversarial loss 사용

**Multi-sacel perceptual loss** <br>
RGB 맵 피라미드를 사용하여 계산

**Adversarial loss** <br>
학생 모델 훈련에 모두 사용 <br>
teacher output과의 adversarial이 아닌 real image에 대한 adversarial을 사용

## **Experiments**
FFHQ 데이터셋을 사용하여 모델 훈련과 평가를 진행 <br>
Adam optimizer로 800K training step

<center><img src="/assets/images/cv/gan/blazestylegan_fig2.jpg" width="60%" alt="Figure 2"></center>

그리고 256, 512 resolution에서 real-time 달성 <br>
$$\rightarrow$$ 10 ms runtime 미만이면 real-time이라고 하는 듯

<center><img src="/assets/images/cv/gan/blazestylegan_fig3.jpg" width="90%" alt="Figure 3"></center>
<center><img src="/assets/images/cv/gan/blazestylegan_fig4.jpg" width="50%" alt="Figure 4"></center>

## **Conclusion**
스마트폰에서 실시간으로 고품질의 얼굴 이미지를 생성할 수 있는 첫 StyleGAN 모델인 BlazeStyleGAN을 소개 <br>
StyleGAN의 synthesis network를 위한 효율적인 아키텍처를 설계 <br>
Teacher 모델로부터의 아티팩트를 완화하기 위해 distillation strategy을 최적화 <br>
모델의 복잡성을 크게 줄임으로써 benchmark에서 모바일 장치에서 real-time 성능을 달성

## **Reference**
1. [BlazeStyleGAN: A Real-Time On-Device StyleGAN](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Jia_BlazeStyleGAN_A_Real-Time_On-Device_StyleGAN_CVPRW_2023_paper.pdf)