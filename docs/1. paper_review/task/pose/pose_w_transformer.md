---
layout: default
title: Pose w/ Transformer Series
nav_order: "2025.03.03"
parent: Pose Estimation
grand_parent: Task
permalink: /docs/paper_review/task/pose/pose_w_transformer_series_2025_03_03
math: katex
---

# **Pose with Transformer Series**
{: .no_toc}
[TransPose: Keypoint Localization via Transformer](https://arxiv.org/abs/2012.14214) <br>
[TokenPose: Learning Keypoint Tokens for Human Pose Estimation](https://arxiv.org/abs/2104.03516) <br>
[ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation](https://arxiv.org/abs/2204.12484) <br>  <br>

Table of contents
{: .text-delta }
1. TOC
{:toc}

## **Summary**
지난 몇 년간, CNN 기반의 방식들은 vision task에서 큰 성과를 얻었다. Transformer가 NLP 분야에서 큰 성공을 얻은 이후 transformer의 architecture를 vision task에서도 사용하기 위한 연구들이 많이 이뤄졌다. ViT는 Input Image를 일정한 patch로 나눠, NLP의 token 처럼 사용한다. 기존의 Vanilla Transformer는 거의 변형하지 않았음에도 Classification task에서 좋은 성능을 얻었다. <br>
따라서, pose estimation과 같은 dense prediction에서도 transformer를 적용하기 위한 연구가 많이 있었고, 대표적인 몇몇 방법들만 본 글에서 정리한다. <br>

기존의 CNN은 locality에 대한 inductive bias 덕분에 작은 dataset에서도 잘 동작하고, deep layer를 통해 넓은 receptive field도 확보했다. 하지만, layer 내부적으로 학습 과정을 명시적으로 해석하기 어려웠다. <br>

<details markdown='block'>
<summary>XAI</summary>
Deep learning 학습을 통해 얻은 결과를 이해하고, 신뢰할 수 있도록 학습 과정 및 추론 과정을 설명할 수 있도록 분석하는 것을 XAI(eXplainable AI)라고 한다. 

입력 이미지에 대한 결과를 도출했을 때, 이미지의 어떤 부분이 결과에 큰 영향을 미쳤는지 분석하는 것도 모델 분석에 큰 기여를 한다. 

CNN에서는 DeConvNet를 통해 feature activities를 generate하여 feature map을 시각화하거나, CAM, Grad-CAM 등을 통해, output의 gradient를 통해 활성화 영역을 파악한다. 

하지만, 이는 skeleton과 같이 활성화 영역의 관계성을 파악하기는 힘들다.
</details>

**TransPose**에서는 CNN 백본을 통해 image의 feature extraction을 수행하고, 이 feature map을 1D로 flatten한 후, Transformer를 거쳐 각 키포인트의 heatmap을 예측한다. 마지막 attention layer의 attention map이 모델이 keypoints 위치 예측에 대한 시각화에 활용될 수 있어, 각 keypoints의 영향과 output 결정 과정을 이해할 수 있다. <br>

**TokenPose**는 각 keypoints를 하나의 Keypoints Token으로 명시하여 transformer의 input으로 활용한다. Input 이미지는 CNN 백본을 통해 image의 feature extraction을 수행하고, feature map의 linear projection하여 Visual Token을 생성한다. <br>
Visual Token과 Keypoint Token은 merge되어 1D Token Sequence를 구성하고 이를 Transformer의 encoder에 입력으로 사용한다. Transformer 이후, 각 Keypoint Token의 output은 MLP 헤드를 통해 heatmap을 예측하도록 학습한다. <br>
Keypoint Token을 모델링함으로써 인접한 keypoint의 관계를 명확하게 학습할 수 있고 분석 가능한다.  <br>

**ViTPose**는 추가적인 CNN이 필요하지 않은 Transformer를 활용한 pose estimation 구조를 제안한다. Transformer를 MAE와 같은 방법을 통해 pretrain하고, 이를 간단한 decoder와 연결하여 인상적인 성능을 보였다. <br> Transformer만 사용하는 간단한 구조의 모델이고, 많은 연구가 진행된 transformer에 대한 방법론들을 적용해볼 수 있다는 점이 장점이지만, 다른 모델들에 비해 FLOPs가 많다는 점이 경량화 및 자세한 성능 비교가 필요해 보인다. <br>


## **[TransPose](https://arxiv.org/abs/2012.14214)**



## **[TokenPose](https://arxiv.org/abs/2104.03516)**



## **[ViTPose](https://arxiv.org/abs/2204.12484)**



