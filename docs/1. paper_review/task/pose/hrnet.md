---
layout: default
title: HRNet Series
nav_order: "2025.02.22"
parent: Pose Estimation
grand_parent: Task
permalink: /docs/paper_review/task/pose/hrnet_series_2025_02_22
math: katex
---

# **HRNet Series**
{: .no_toc}
[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919) <br>
[Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403) <br>
[HRFormer: High-Resolution Transformer for Dense Prediction](https://arxiv.org/abs/2110.09408)
Table of contents
{: .text-delta }
1. TOC
{:toc}

## **Summary**
일반적으로 deep learning에서는 high dimension data가 low dimension manifold에서 분포할 수 있다고 가정한다. 이미지같은 고차원 데이터도 network layer를 통하며, 효과적으로 압축하고, 그 내재된 구조를 학습하여 좋은 일반화 성능을 보일 수 있다. 하지만, Classification과 같은 task와 다르게, Pose Estimation, Semantic Segmantation과 같은 dense prediction은 pixel 단위의 정밀한 prediction이 필요하기 때문에, high-resolution의 feature 유지가 중요하게 생각된다.<br>
High-resolution의 feature를 유지하기 위해, skip connection이나 upsampling과 같은 방법들이 많이 사용되지만 low-resolution에서는 high-resolution의 feature representation을 학습하기 힘들다. <br> 

따라서, HRNet은 여러 해상도의 **parallel stream**을 구성하고, 각 stream의 feature map을 **fusion** 하여, low-resolution부터 high-resolution의 feature를 연결한다. 이로인해, 각 feature map이 high-resolution의 representation을 유지하면서, low-resolution의 general feature를 효과적으로 학습할 수 있게 설계한다.

<figure>
    <center><img src="/assets/images/papers/task/pose/hrnet-series_fig1.jpg" width="90%" alt="Figure 1"></center>
	<center><figcaption><em>[Figure 1]</em></figcaption></center>
</figure>

HRNet과 같이 고해상도 특징을 지속적으로 유지하는 네트워크들은 pose estimation, object detection, semantic segmentation 등에서 좋은 성능을 보이지만, mobile 및 embedded와 같은 제한된 resource에서는 활용하기 힘들다. 따라서, Lite-HRNet에서는 HRNet을 경량화하여 효율적으로 inference 할 수 있는 model을 설계했다.<br>
일반적으로, 여러 resolution의 feature map을 섞기 위해, 1x1 convolution이 많이 사용된다. 하지만, 이는 channel에 대해 선형적인 시간 복잡도를 가지기 때문에, Lite-HRNet에서는 ShuffleNet의 **Shuffle block**과 **Conditional Channel Weighting**을 통해 경량화를 수행한다. <br>

HRFormer는 Vision Transformer가 여러 Vision Task에서 뛰어난 성능을 보인 것을 참고하여, HRNet에 Transformer를 활용하였다. CNN으로 구성된 HRNet의 Body 부분을 Transformer로 대체하였다. Transformer의 경우 일반적으로, input이 patch로 나뉘어 Attention 및 FFN layer에 들어간다. 즉, token 별로 sequence를 처리하기 때문에 각 token별 representation 학습이 쉽지 않다. 따라서, swin이나 window, shuffle과 같은 방법들이 쓰이고, HRFormer에서는 **FFN layer에서 depth-wise convolution**을 적용하여 여러 feature representation을 연결한다. 

<figure>
    <center><img src="/assets/images/papers/task/pose/hrnet-series_fig2.jpg" width="80%" alt="Figure 2"></center>
	<center><figcaption><em>[Figure 2]</em></figcaption></center>
</figure>


## **[HRNet](https://arxiv.org/abs/1908.07919)**
HRNet의 stem은 2개의 2 stride 3x3 Convolution으로 이뤄져 있다. Stem은 input을 $\frac{1}{4}$ resolution으로 줄어든 후, main body의 입력으로 사용된다. <br>

Body는 <br>
1. 여러 resolution 의 parallel stream으로 분리되는 부분 <br>
2. 각 resolution stream을 fusion하는 부분 <br>
3. Task에 따라 output을 출력하는 head 부분 <br>
으로 나뉜다. 

### **Parallel Multi-Resolution Convolutions**
High-resolution stream에서부터 low-resolution stream을 점진적으로 추가하여, parallel stream을 구성하는 부분이다. 아래와 같은 방식으로 parallel stream이 구성되며, $$\mathcal{N}_{sr}$$에서 $$s$$는 netwokr layer의 depth를 의미하고, $$r$$이 stream index를 의미할 때, resolution은 첫 stream 대비 $$\frac{1}{2^{r-1}}$$의 resolution을 가진다. <br>

$$
\begin{matrix}
\mathcal{N}_{11} & \rightarrow & \mathcal{N}_{21} & \rightarrow & \mathcal{N}_{31} & \rightarrow & \mathcal{N}_{41}   \\
 & \searrow & \mathcal{N}_{22} & \rightarrow & \mathcal{N}_{32} & \rightarrow & \mathcal{N}_{42} \\
 & & & \searrow & \mathcal{N}_{33} & \rightarrow & \mathcal{N}_{43} \\
 & & & & & \searrow & \mathcal{N}_{33} \\
\end{matrix}
$$

### **Repeated Multi-Resolution Fusions**
각 resolution stream의 representation을 학습하기 위해 fusion module이 사용된다. 논문에서는 4개의 residual unit마다 fusion을 수행해준다. Fusion은 각 stream에 aggregation되는데, channel 수와 resolution을 target stream에 맞게 변형한 후 feature map을 더하여 수행된다. <br>
Resolution에 따라, 
- Upsampling: Bilinear upsampling 후, 1x1 convolution
- Downsampling: stride 2의 3x3 convolution
을 통해 수행된다. <br>
그림으로 나타난 과정을 하기 그림과 같다.

<figure>
    <center><img src="/assets/images/papers/task/pose/hrnet-series_fig3.jpg" width="50%" alt="Figure 3"></center>
	<center><figcaption><em>[Figure 3]</em></figcaption></center>
</figure>

### **Representation Head**
HRNet은 task 및 목적에 따라 3개의 head를 제안한다. <br>
1. HRNetV1: High-resolution stream에서만 representation을 출력 <br>
2. HRNetV2: 각 stream의 feature map을 High-resolution stream에 aggregation 하여 representation을 출력 <br>
3. HRNetV2p: HRNetV2의 output을 여러 level로 downsampling하여 representation을 출력 <br>

논문에서는 HRNetV1을 Pose Estimation, HRNetV2를 semantic segmentation, HRNetV2p를 object detection에 적용하였다.

### **Architecture & Human Pose Estimation**
HRNet은 stem에서 $$\frac{1}{4}$$ resolution으로 representation을 출력하기 때문에 main body에서 각 stream의 resolution은 $$\frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}$$로 구성된다. 각 stream의 convolution channel은 $$C, 2C, 4C, 8C$$로 구성되어 있다. 자세한  architecture는 하기 표와 같다. <br>

<figure>
    <center><img src="/assets/images/papers/task/pose/hrnet-series_fig4.jpg" width="90%" alt="Figure 4"></center>
	<center><figcaption><em>[Figure 4]</em></figcaption></center>
</figure>

HENet은 각 해상도에서 representation을 학습하는 반면, fusion 단계에서 representation을 섞어주기 때문에 group convolution과 같이 다양한 feature를 학습하는 효과를 얻을 수 있다. HRNet은 여러 resolution에서 representation을 학습하고 fusion을 통해 representation을 학습하는데 뛰어난 효과를 보인다. <br>

HRNet의 human pose estimation은 K개의 keypoint에 대해 heatmap을 추정한다. Heatmap은 이미지의 각 위치에서 해당 관절이 존재할 확률이나 신뢰도를 나타내는 2차원 맵(heatmap)을 생성하는 것을 의미한다. <br>
일반적으로 실제 관절 위치 주변에는 2D Gaussian 분포와 비슷한 형태의 높은 값이 나타나고, 그 외의 부분은 낮은 값이 나오도록 학습시킨다. 직접 좌표를 regression하는 것보다 heatmap 방식은 spatial representation을 보존하면서 더 안정적으로 학습할 수 있다는 장점이 있다. <br>
$$H_k$$가 $$k$$th keypoint의 heatmap을 의미할 때, 

$$
\begin{gather}
H_k(x,y)=exp(-\frac{(x-x_k)^2 + (y-y_k)^2}{2\sigma^2})
\end{gather}
$$ 

로 나타나고, MSE Loss는

$$
\begin{gather}
\mathcal{L}=\frac{1}{K}\sum^k_{k=1}\sum_{x,y}(\hat{H}_k(x,y)-H_k(x,y))^2
\end{gather}
$$ 

와 같이 계산된다. <br>

추가적으로, HRNet은 bbox prediction은 [SimpleBaseline](https://github.com/Microsoft/human-pose-estimation.pytorch)을 활용하고, 평가 metric으로는 OKS(Object Keypoint Similarity)를 사용한다.

$$
\begin{gather}
OKS: \frac{\sum_i\exp{(-d_i^2 / 2s^2k^2_i)}\delta(v_i > 0)}{\sum_i \delta(v_i > 0)}
\end{gather}
$$ 

추가적인 자세한 내용은 논문의 section 4 참조

## **[Lite-HRNet](https://arxiv.org/abs/2104.06403)**
### **Naive Lite-HRNet**


### **Lite-HRNet**


#### **Conditional channel weighting**


#### **Cross-resolution weight computation**




## **[HRFormer](https://arxiv.org/abs/2110.09408)**
### **High-Resolution Transformer**


