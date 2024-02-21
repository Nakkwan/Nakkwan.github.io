---
layout: default
title: Diffstride
nav_order: "2024.01.20"
parent: Etc
grand_parent: Computer Vision
permalink: /docs/computer_vision/etc/diffstride_2024_01_20
---

# **DiffStride: Learning strides in convolutional neural networks**
{: .no_toc}
[Learning strides in convolutional neural networks](https://arxiv.org/abs/2202.01653)
{: .fs-6 .fw-300 }

Table of contents
{: .text-delta }
1. TOC
{:toc}

Downsampling 등의 크기를 결정하는 것은 큰 hyperparameter <br>
하지만, 이는 NAS로 찾기엔 시간이 오래 걸리고 훈련이 불가능함 <br>
따라서 훈련 가능한 diffstride를 제시

## **Introduction**
Pooling이나 strude convolution은 관련 정보에 집중하게 하고, shift-invariant와 high-receptive field에 연관이 있음 <br>
일반적으로 pooling layer는 convolution (1) 같은 local calculation 후 subsampling (2)으로 진행됨 <br>
보통 (1)을 개선하기 위해 aliasing을 없애는 방식을 제시함 <br>
Stride가 2여도 정보량이 $$\frac{1}{4}$$로 줄어들기 때문에 분수 stride나 maxpooling도 제시됨 <br>
Fractional stride는 layer에 flexible을 주지만, search space를 증가시킴 <br>
또한, 적정한 stride를 찾기 쉽지 않음 <br>

따라서, 저자는 stride를 학습하는 diffstride를 제시 <br>
Diffstride는 spatial domain의 downsampling을 frequency domain의 crop처럼 cast <br>
$$\rightarrow$$ Backpropagation을 통해 split (2D attention window)

## **Methods**



## **Methods**

## **Reference**
1. 