---
layout: default
title: Survey on Visual Transformer
nav_order: "2023.12.21"
parent: VIT
grand_parent: Computer Vision
permalink: /docs/computer_vision/vit/survey_on_vit_2023_12_20
math: katex
---

# A Survey on Visual Transformer
{: .no_toc}
[A Survey on Visual Transformer](https://arxiv.org/abs/2012.12556)

Table of Contents
{: .text-delta}
1. TOC
{:toc}

Transformer는 self-attention을 기반으로 함

inductive bias가 적게 필요하기 때문에 관심을 받고 있음

장단점과 backbone network, high/mid-level vision, low-level vision, and video processing에서의 transformer를 분석

## **Introduction**
기본이 되는 NN infra가 존재함 <br>
$$\rightarrow$$ MLP + FC, shift-invariant의 CNN, sequential의 RNN <br>
Transformer는 self-attention을 기반으로 특징을 추출하는 새로운 NN 임

Transformer는 NLP에서 처음 시작됨 <br>
Machine translation의 vanilla transformer 이후, BERT는 Bidirectional Encoder를 적용하여 label없는 pretrained를 제안했고, GPT-3는 대규모 LLM을 pretrain함

CV에서도 transformer가 CNN의 대안으로 떠오름 <br>
$$\rightarrow$$ Pixel 예측은 계산이 너무 복잡하기 때문에 VIT에서 Image patch에 transformer 적용

이외에도 object detection, semantic segmentation, image processing, video understanding 등에 활용됨

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig1.jpg" width="60%" alt="Figure 1"></center>

Task의 분류는 Detection과 같은 것은 high-level, 복원, denoise는 low-level로 취급

전체적인 것은 아래의 표와 같음

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig2.jpg" width="60%" alt="Figure 2"></center>

# **Formulation of Transformer**

Transformer는 transformer block이 여러 번 반복되는 encoder와 decoder로 구성됨 <br>
**Encoder**: Input에 대한 encoding <br>
**Decoder**: encoding의 incorporated contextual information를 사용하여, sequential하게 output 생성  <br>
**Transformer block:** MHA(multi-head attention layer), FFNN(feed-forward neural network), shortcut connection, layer normalization으로 구성됨

## Self-Attention

Input으로는 3개의 Vector $$q, k, v$$가 들어감 <br>
$$\rightarrow$$ $$d_q,d_k,d_v,d_{dim}\in\mathbb{R}^{512}$$ <br>
이후 matmul을 통해 $$Q,K,V$$로 유도됨

Attention 계산 과정은

1. Score 계산: $$S=Q\cdot K^T$$
2. Gradient stability를 위한 Normalize: $$S_n=S/\sqrt{d_k}$$
3. Score를 probabilities로 변환(Softmax): $$P=\mathrm{Softmax}(S_n)$$
4. weighted $$V$$ matrix를 얻음: $$Z=V\cdot P$$
    

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig3.jpg" width="60%" alt="Figure 3"></center>

최종적으로 

$$
\mathrm{Attention}(Q,K,V)=\mathrm{Softmax}(\frac{Q\cdot K^T}{\sqrt{d_k}})\cdot V
$$

Encoder에서 $$q,k,v$$가 다 같지만, decoder에서 $$k,v$$는 encoder의 output, $$q$$는 이전 layer에서 유도됨

> value가 decoder에서 오지 않은 이유는 decoder의 next layer의 state가 input layer와 encoder간의 유사성으로 weighted된 encoder라는 것 <br>
Encoder를 통해 aggregation된 input들을 decoder의 input을 참조하는 것이라고 생각 <br>
decoder의 query는 말 그대로 요청 <br>
Encoder의 key가 응답되어, attention map 계산 <br>
aggregated input인 value를 이를 통해 weighted 해나감
> 
- **즉, input sentence에 대해, decoder의 input word를 reference로 삼아, output을 진행** <br>
- **첫 sequence는 무조건 \<sos\>임 (나머진 masking 되어있음)** <br>
- **이후, output들을 reference 삼아, 결과를 출력하는 것이기 때문에 value가 encoder로부터 옴**

> 위에 말한 것은 residual이 없을 때인데, residual이 추가되었으니, 어떻게 동작? <br>
$$\rightarrow$$ Input이 output에 그대로 전달되고, encoder를 ref하는 것은 잔차인데…?
> 

위의 attention process는 word position 정보를 capture하는 능력이 부족함

**Positional Embedding** <br>
d_model 차원의 position embedding이 필요

$$
\mathrm{PE}(pos,2i)=\sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\\mathrm{PE}(pos,2i+1)=\cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

### **Multi-Head Attention**

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig4.jpg" width="60%" alt="Figure 4"></center>

Reference word에 대해 input sentence의 여러 word에 attention을 진행할 수 있음 <br>
$$\rightarrow$$ 동시에 중요한 단어들이 있을 수 있기 때문에 

Multi-head에서 input vector들을 서로 다른 subspace에 projection <br>
head 수 $$h$$에 대해 input vector의 $$d_{model}$$이 나뉨 <br>
각각에 대해 attention을 진행 후 concatenate되고, $$W\in\mathbb{R}^{d_{model}\times d_{model}}$$를 통해 projection 됨  

$$
\mathrm{MultiHead}(Q',K',V')=\mathrm{Concat}(\mathrm{head}_1,\cdots,\mathrm{head}_ℎ)𝐖^0,\\\mathrm{where}\;  \mathrm{head}_i=\mathrm{Attention}(Q_i,K_i,V_i).
$$

## Other Key Concepts in Transformer
### **Feed-Forward Network**
두 개의 Linear layer와 nonlinear activation GELU ($$d_h=2048$$)

$$
FFN(X)=W_2\sigma(W_1X),
$$

### **Residual Connection in the Encoder and Decoder**
각 attention block과 FFN에는 residual이 실행되고, normalize됨 <br>
BN는 feature값이 급격히 변하기 때문에, layer norm을 씀

$$
\mathrm{LayerNorm}(X+\mathrm{Attention}(X)),\\\mathrm{LayerNorm}(X+\mathrm{FFN}(X))
$$

<details markdown="block">
<summary> [Pre_LN](https://arxiv.org/abs/2002.04745) (pre-layer normalization)도 많이 쓰임</summary>
$$\rightarrow$$ Residual 내부에서, Attention이나 FFN 이전에 normalization 수행 <br>

$$
X+\mathrm{Attention}(\mathrm{LayerNorm}(X)),\\
X+\mathrm{FFN}(\mathrm{LayerNorm}(X))
$$

Pre의 경우 norm 후 더하기 때문에, activation의 크기가 크고, Post는 더한 후 norm이기 때문에 scaling되어 있음 <br>
따라서, gradient의 upper bound를 보았을 때, Pre의 scale이 더 작음

$$\rightarrow$$ norm을 하면, activation이 일정 범위 안에 들어오기 때문에, activation func에서 gradient가 큼 <br>
$$\rightarrow$$ Pre는 gradient가 layer의 층수에 반비례 <br>
    $$\rightarrow$$ Skip-connection 때문에, layer가 깊어질수록, 값은 쌓임 (커짐) <br>
    $$\rightarrow$$ 따라서, layer norm에서 input의 크기는 크고, scaling은 작아짐 <br>
        $$\rightarrow$$ output은 작은 값을 내야하는 쪽으로 학습되기 때문에 scaling은 작아짐 (skip-connect) <br>
    $$\rightarrow$$ layer norm에서 scaling의 gradient는 input에 영향을 받고, input의 gradient는 scale의 영향을 받는데, output 쪽의 값이 크기 때문에 scale은 더 작음 <br>
    $$\rightarrow$$ 따라서, input 쪽의 gradient는 더 작아짐 <br>
    $$\rightarrow$$ ~~layer가 쌓일수록, activation이 커지기 때문에, 값이 act_func에서 끝으로 감~~ <br>
    $$\rightarrow$$ 원래, output 쪽에 gradient가 컸으므로, 반비례를 통해 일정하게 맞춰짐 <br>
    $$\rightarrow$$ 따라서, pre는 lr을 warmup없이 크게 줘도 학습이 되고, 수렴 속도도 더 빠름 <br>
        $$\rightarrow$$ warmup은 초기 gradient가 너무 크면, 학습이 불안정하기 때문에, lr을 천천히 늘리는 것 <br>
$$\rightarrow$$ Post의 경우, output layer 쪽의 gradient가 크기 때문에 훈련이 불안정하고, warm up이 필요 <br>
$$\rightarrow$$ representation collapse가 일어날 수 있고, post는 gradient vanishing의 가능성이 있음
</details>

### **Final Layer in the Decoder**

Vector를 다시 word로 바꾸기 위해 linear + softmax <br>
Linear: $$d_{word}$$로 logits를 projection

이후 대부분의 CV Transformer는 encoder의 형태를 취함 <br>
$$\rightarrow$$ Encoder가 feature extractor로 생각될 수 있기 때문 <br>
$$\rightarrow$$ global feature를 capture할 수 있음 <br>
$$\rightarrow$$ 병렬 계산이기 때문에 효율적임

# **Vision Transformer**
CV에서 transformer의 사용을 Review

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig5.jpg" width="60%" alt="Figure 5"></center>

## **Backbone for Representation Learning**

이미지의 경우 text 보다, dim이 크고, noise가 많고, 중복 pixel이 많음 <br>
Transformer는 CNN과 비슷하게, backbone으로 사용될 수 있음 <br>
이런 결과는 아래 그림과 표에 요약됨

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig6.jpg" width="60%" alt="Figure 6"></center>

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig7.jpg" width="60%" alt="Figure 7"></center>
(a) Acc v.s. FLOPs.

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig8.jpg" width="60%" alt="Figure 8"></center>
(b) Acc v.s. throughput.

### Pure Transformer
**VIT.** <br>
Classification task에 대해, image patch를 바로 transformer에 적용 <br>
Transformer의 framework를 거의 그대로 씀

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig9.jpg" width="60%" alt="Figure 9"></center>
Make Patch: $$X\in\mathbb{R}^{h\times w\times c} \rightarrow X_p\in\mathbb{R}^{n\times(p^2\cdot c)},\;\; p: \mathrm{patch \;resolution}$$

따라서, patch의 개수 $$n=hw/p^2$$

Transformer는 모든 layer에서 일정한 width(dim)을 사용하기 때문에 patch vector를 trainable linear projection를 통해 embedding space로 mapping <br>
추가적으로 Positional embedding과 learnable class token (image 표현 학습) 사용

class 개수에 따라, classification head attach <br>
$$\rightarrow$$ 일반적으로, encoder를 large dataset으로 pretrained 후, head에 transfer learning

Vision Transformer는 large dataset에서 강점이 있음 <br>
inductive bias가 없기 때문에, 작은 dataset에서는 약한 모습을 보임 <br>
Pretrained을 한 후에는 좋은 성능을 보임

DeiT(Data-efficient image transformer)에서는 convolution-free transformer를 보임 <br>
CNN teacher, distillation token 사용하여 성능을 높임 <br>
- distillation token은 CNN teacher의 label을 예측함 <br>
추가적으로, Strong data augment를 통해 높은 성능을 달성

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig10.jpg" width="60%" alt="Figure 10"></center>

**Variants of ViT.**

이후, locality, self-attention의 성능 등을 높이려는 연구가 많이 진행됨

$$\rightarrow$$ VIT는 long-range는 잘 capture했지만, local은 간단한 linear로 modeling되어 약했음 <br>
$$\rightarrow$$ 또한 overlap없이, sliding 하는 hard split 방법으로 token을 얻기 때문에 image local structure(ex. edges and lines)를 modeling할 수 없게 됨

local 성능 증가 <br>
- TNT[[29](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib29)]는 patch를 또 subpatch로 나눠, transformer-in-transformer 구조를 씀 <br>
- Twins [[62](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib62)]과 CAT[[63](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib63)]은 layer-by-layer로 local, global attention을 번갈아 수행 <br>
- **Swin Transformer**s [[60](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib60), [64](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib64)]는 window 내에서 local attention을 수행하고, window를 shifting해가며 global에 대한 정보를 얻음 <br>
- Shuffle Transformer [[65](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib65), [66](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib66)]는 shifting 대신 shuffle을 수행 <br>
- RegionViT [[61](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib61)]에선 global, local token을 따로 생성하여, local은 cross-attention을 통해 global의 정보를 얻음 <br>
- **T2T**[[67](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib67)]은 local feature aggregation을 통해 local information를 boosting <br>
    - tokens의 길이를 progressive하게 줄임
    - layer-wise **Token-to-Token module: **이미지의 local structure information을 담음
        - Re-structurization
            - 이전 transformer layer의 tokens이 $$T$$일 때,
            
            $$
            T'=MLP(MSA(T)),\\I=Reshape(T'),\;\;T'\in\mathbb{R}^{l\times c},I\in\mathbb{R}^{h\times w\times c}
            $$
            
        - Soft Split(SS)
            - Re-structurization에서 얻은 $$I$$를 다시 tokenize하는 곳
            - token을 만들 때 information loss가 생길 수 있기 때문에 patch를 overlap하면서 split
            - 각 patch는 surrounding patches와 correlation을 가지게 됨
                - CNN의 locality inductive bias와 비슷한 prior라고 볼 수 있음
            - split patches를 하나의 token으로 concat
                - local information이 surrounding pixels과 patches로부터 aggregate
                
                $$
                T_o=\mathbb{R}^{l_o\times ck^2},\\ \mathrm{where\;} l_o=[\frac{h+2p-k}{k-s}+1]\times [\frac{w+2p-k}{k-s}+1],\\ \mathrm{patch\;size}=k, \mathrm{overlap\;size}=s, \mathrm{padding}=p
                $$
                
    - 따라서, T2T module을 통해 점진적으로 tokens의 길이를 줄일 수 있고 이미지의 spatial structure를 바꿀 수 있음
        - $$k > s$$이기 때문에, h,w는 점점 줄어듦
        - 최종적으로,
        
        $$
        T'_i=MLP(MSA(T_i))\\I_i=Reshape(T'_i)\\T_{i+1}=SS(I_i), i=1,\dots,(n-1)
        $$
        
    - T2T-ViT backbone: **T2T module로부터 tokens의 global attention relation
        - feature richness와 connectivity를 향상시키기 위한 DenseNet의 dense connection
        - channel dimension과 head number를 바꾸기 위한 Wide-ResNets 또는 ResNeXt
        - Squeeze-an-Excitation Networks
        - GhostNet
        
        <center><img src="/assets/images/cv/vit/survey/survey-vit_fig11.jpg" width="60%" alt="Figure 11"></center>
        

self-attention 성능 증가

- DeepViT[[68](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib68)]: Cross-head communication 제안 (re-generate the attention map)
- KVT[[69](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib69)]: Top-k attention 만 계산하는 k-NN attention 제안
- XCiT[[71](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib71)]: self-attention을 channel-wise로 수행하여, high-res에 효과적

Architecture의 경우, pyramid형 (Swin, HVT, PiT, etc…), 2-way형 (UNet 구조), NAS 등이 활용되고 있음

### Transformer with Convolution

Vision Transformer에 locality를 추가하기 위해 CNN을 병합

- **BoTNet**[[100](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib100)]: ResNet의 마지막 3개 block에서 conv를 self-attention으로 대체, latency의 overhead를 최소화하고 baseline을 크게 개선했습니다4

또한, transformer는 optimizer, hyper-parameter, schedule of training에 민감

그 이유가 ViT의 초기 patchfiy에서 사용하는 $$s$$ stride, kernel size의 conv라고 지적 

$$\rightarrow$$ stride:2, kernel size: 3*3인 conv stem을 사용하여 완화시킴 ([Early Convolution](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib102), [CMT](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib94))

$$\rightarrow$$ single transformer block와 비슷한 complexity를 갖는 5 layer 이하의 CNN

### Self-supervised Representation Learning

**Generative Based Approach.**

- iGPT[[14](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib14)]: low resolution으로 reshape 후, flatten. 이후, bert와 같이 학습하여 pretrain
    - 각 pixel은 sequential로 취급되어, train
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig12.jpg" width="60%" alt="Figure 12"></center>
    

**Contrastive Learning Based Approach.**

현재 가장 많이 쓰이는 Self-supervised method

- [MoCo v3](https://arxiv.org/abs/2104.02057)에서 Transformer에 contrastive 사용
    - instability가 ViT pretrain의 낮은 정확도의 문제로 보임
    - 한 이미지에서 2개의 crop을 얻음 (vector $$q,k$$로 encode됨)
    - 위의 두 latent vector를 contrastive loss 최소화하는 것으로 formulate
    
    $$
    L_q=-\log\frac{\exp{(q\cdot k^+/\tau)}}{\exp(q\cdot k^+/\tau)+\sum_{k^-}\exp(q\cdot k^-/\tau)}
    $$
    
    - ViT에서 patch projection layer를 훈련하지 않고, random하게 하면, 안정성이 좋아지지만 완전한 해결책은 아님

### Discussions

MHSA, MLP, shortcut, LN, PE, network 구성은 모두 vision transformer에서 중요한 역할을 함

## High/Mid-level Vision

object detection [[16](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib16), [17](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib17), [113](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib113), [114](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib114), [115](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib115)], lane detection [[116](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib116)], segmentation [[33](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib33), [25](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib25), [18](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib18)], pose estimation [[34](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib34), [35](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib35), [36](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib36), [117](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib117)]에 대해 review

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig13.jpg" width="60%" alt="Figure 13"></center>

### Generic Object Detection

CNN이 basic이지만, transformer도 각광받고 있음

OD에서 transformer는 크게 2가지로 나뉨

1. Transformer-based set prediction methods
2. Transformer-based backbone methods

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig14.jpg" width="60%" alt="Figure 14"></center> 

**Transformer-based Set Prediction for Detection**.

**DETR**[[16](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib16)]에서 Object detection을 intuitive set prediction problem으로 취급하여 NMS같은 post-processing를 제거함

- CNN backbone으로 image의 feature를 추출
- flattened feature에서 position info를 보완하기 위해, flattened feature에 PE 사용
- Encoder 이후, object(image에 있는)의 PE가 같이 decoder의 input으로 들어감
    - PE의 N의 수는 일반적으로 이미지에 있는 객체 수보다 큼
- FFN은 class와 bounding box coord를 predict하는 것에 사용됨
    - N개의 object가 병렬로 예측됨
    - bipartite matching을 사용하여 GT에 대해 predict 할당
    - Hungarian loss로 모든 matching에 대한 loss 계산
    
    $$
    L_{Hungarian}(y,\hat{y})=\sum^N_{i=1}[-\log \hat{p}_{\hat{\sigma}(i)}(c_i)+\mathbb{I}_{\{c_i\neq\varnothing\}}L_{box}(b_i,\hat{b}_{\hat{\sigma}}(i))],
    $$
    
- End-to-End object detection framework 제안
- 긴 training과 작은 object에 대한 문제가 있음

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig15.jpg" width="60%" alt="Figure 15"></center>

Deformable DETR[[17](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib17)]에서 위의 문제를 보완

- image feature map의 spatial location를 보던 기존 DETR에서 reference point 주변의 작은 key position set에 집중
- compute cost를 줄이고, 빠르게 수렴함
- 또한 pyramid형식으로 변형하기 쉬움

[[120](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib120)]: Decoder의 cross-attention에서 발생하는 느린 수렴 개선 (Encoder Only Pyramid)

[[123](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib123)]: SMCA(Spatially Modulated Co-Attention)

[[121](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib121)]: ACT(Adaptive Clustering Transformer) 계산 비용을 줄임

**Transformer-based Backbone for Detection**.

ViT와 비슷하게 transformer 자체를 backbone으로 활용

Patch 단위로 이미지가 split되어 transformer의 input으로 들어감

혹은 general transformer 이후, traditional backbone으로 transfar할 수 있음

**Pre-training for Transformer-based Object Detection.**

**UP-DETR**[[32](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib32)]: unsupervised pretext task(query patch detection)

DETReg[129](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib129)]: Selective Search를 기반으로 임의의 object를 predition하도록 pretrain

YOLOS[[126](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib126)]: ViT에서 class token 대신 detection token을 사용

### Segmentation

**Panoptic Segmentation.**

DETR을 확장하여 사용할 수 있음

Max-DeepLab[[25](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib25)]: Mask Transformer 사용

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig16.jpg" width="60%" alt="Figure 16"></center>

Max-DeepLab

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig17.jpg" width="60%" alt="Figure 17"></center>

Mask Transformer

**Instance Segmentation.**

**VisTR**[[33](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib33)]: Video에서 instance seg를 하기 위해, 3D CNN으로 image 들의 feature를 얻은 후, Transformer

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig18.jpg" width="60%" alt="Figure 18"></center>

**Semantic Segmentation.**

- **[SETR](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Rethinking_Semantic_Segmentation_From_a_Sequence-to-Sequence_Perspective_With_Transformers_CVPR_2021_paper.pdf)**[[18](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib18)]: ViT의 encoder에 Decoder를 design (2개를 design함)
    - Encoder의 output Z는 $$\frac{HW}{256}\times C$$로 reshape 되고,
    - SETR-PUP
        - x2 씩, 4번 upsampling 하여, $$(H, W)$$ resolution output 얻음
    - SETR-MLA
        - M stream에 대해 aggregate해가며 output을 얻음
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig19.jpg" width="60%" alt="Figure 19"></center>
    
- SegFormer[[135](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib135)]: multi layer에서 feature를 뽑고, MLP를 적용하여 decoder와 Transformer를 통합하여 간단하고 좋은 framework 제안
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig20.jpg" width="60%" alt="Figure 120"></center>
    

**Medical Image Segmentation.**

- **Swin-UNet**[[30](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib30)]: Transformer-based U-shaped Encoder-Decoder architecture
    - tokenized image patche를 사용하고,
    - local-global semantic feature를 위한 skip-connection 사용

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig21.jpg" width="60%" alt="Figure 21"></center>

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig22.jpg" width="60%" alt="Figure 22"></center>

- Medical Transformer[[136](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib136)]: self-attention에 control mechanism 적용
    - Gated Axial-Attention model 제안
        
        <center><img src="/assets/images/cv/vit/survey/survey-vit_fig23.jpg" width="60%" alt="Figure 23"></center>
        

### Pose Estimation

RGBD인 input image에서 joint coordinate 또는 mesh vertice를 예측하는 것이 목표

**Hand Pose Estimation.**

- PointNet[[138](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib138)]: Input으로 Point cloud가 들어갈 때에 대한 Classification, Segmentation 모델
    
    기존 point cloud는 3D voxel로 바꾸는 것과 같이, discrete data로 변환 후 NN model에 input
    
    $$\rightarrow$$ 하지만 quantization artifact가 발생할 수 있음
    
    $$\rightarrow$$ Point cloud는 mesh와 같이 이웃 점들을 고려하지 않아도 되기 때문에 쉬움
    
    $$\rightarrow$$ 점들의 조합이므로 permutation invariant하고 rigid motion에도 invariant
    
    $$\rightarrow$$ 점의 순서에 따라 달라지지 않고, 전체를 회전, 이동해도 3D 형체가 달라지지 않음
    
    $$\rightarrow$$ 하지만 local point간에는 연관성이 있을 수 있음
    
    - Point Cloud는 x,y,z corrd와 feature channel (color, normal) 등의 정보가 있음
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig24.jpg" width="60%" alt="Figure 24"></center>
    
    - input permutaton에 invariant하려면 3가지의 방법이 있음
        1. input을 canonical 하게 정렬
            1. data perturbation에 stable하지 않음
        2. input을 seq처럼 다룸. 단, 모든 permutation에 대해 augment
            1. Point 들에 순서가 생겨버림
        3. **symmetric fucntion**을 사용함. ex) +와 *는 symmetric binary function
            1. MLP와 max pooling 사용
    - MLP는 input node에 대해 weight를 sharing
    - Point 간 정보는 max pooling에서 얻음
    - Segmentation은 global 정보도 필요하기 때문에, 각 point에 appendix
    - T-Net의 경우 permutation invariant를 위해 orthogonal regularization 추가
    
    $$
    L_{reg}=\lVert I-AA^T \rVert^2_F
    $$
    
- PointNet++[[139](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib139)]: PointNet에서 더 계층적으로 feature를 뽑음
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig25.jpg" width="60%" alt="Figure 25"></center>
    
- Hand-transformer[[34](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib34)]: PointNet으로 feature를 뽑은 후, MHSA로 embedding을 뽑
    - Reference Extractor를 뽑은 후, positional encoding으로 decoder에 input
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig26.jpg" width="60%" alt="Figure 26"></center>
    
- HOT-Net[[35](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib35)]: ResNet으로 2D hand-object pose를 추정 후, transformer의 input으로 3D hand-object pose predict
- Handsformer[[140](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib140)]: 손에 대한 color image가 input으로 들어가면 2D location set과 Spatial PE가 MHSA로 들어감
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig27.jpg" width="60%" alt="Figure 27"></center>
    

**Human Pose Estimation.**

- METRO[[36](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib36)]: 2D 이미지에서 사람 pose 추정
    - CNN으로 feature 추출 후, template human mesh에 concate하여 PE
    - 이후, multi-layer transformer encoder
    - Human pose에서 non-local relationship에 대한 학습을 위해 일부 masking하며 진행
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig28.jpg" width="60%" alt="Figure 28"></center>
    
- Transpose[[117](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib117)]: 각 pose의 keypoints가 어떤 spatial point를 refer했는지, heatmap 추정
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig29.jpg" width="60%" alt="Figure 29"></center>
    
- TokenPose[[141](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib141)]: Pose Estim에 Keypoint token을 사용하는 새로운 방법 제시
    - 이미지를 CNN을 통해 vector를 뽑고, transformer에 PE와 같이 input으로 들어감
    - 각 관절에 대해 keypoint vector를 Transformer에 input으로 같이 넣고, output을 뽑음
    - 이미지와 관절에 대한 contraint를 동시에 학습할 수 있음
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig30.jpg" width="60%" alt="Figure 30"></center>
    
- Test-Time Personalization[[144](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib144)]: Human labeling 없이, test-time에서 pose personalize
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig31.jpg" width="60%" alt="Figure 31"></center>
    

### Other Tasks

**Pedestrian Detection.**

PED[[145](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib145)]: DETR, Deformable DETR은 조밀한 pedestrian에 열화가 있으므로, dense query와 강한 attention을 가지는 DQRF(Rectified Attention field) decoder 제안

**Lane Detection.**

LSTR[116]: Transformer를 통해, global context 학습

**Scene Graph.**

Graph R-CNN[149]: self-attention을 통해 인접 노드의 상황 정보를 그래프에 통합

Texema[151]: T5(Text-to-Text Transfer Transformer)를 사용하여, text input에서 structured graph를 생성하고 이를 활용하여 relational reasoning module 개선

**Tracking.**

- TMT[153]: Template-based discriminative tracker에서 Transformer encoder-decoder
- TransT[155]: Template-based discriminative tracker에서 Transformer encoder-decoder
- TransTrack[156]: online joint-detection-and-tracking
    - Query, Key 메커니즘을 활용하여 기존 개체를 추적
    - learned object querie set을 pipeline에 도입하여 새로운 object를 검색

**Re-Identification.**

- TransReID[[157](https://arxiv.org/abs/2102.04378)]: Image의 id를 식별할 수 있는 detail feature를 보존하면서 re-identification
    - VIT 구조를 기반으로 함
    - JPM (jigsaw patch module): shift 및 shuffle을 통해 patch embedding을 재배열
    - SIE (side information embeddings): 카메라의 view, variations에 대한 편향 완화
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig32.jpg" width="60%" alt="Figure 32"></center>
    
- Re-ID[[158](https://arxiv.org/abs/2104.01745)]: Video에 대한 Re-ID
    - spatial, temporal feature를 refine 후 cross view transformer로 multi-view aggregate
- Spatiotemporal Transformer[[159](https://arxiv.org/abs/2103.16469)]: Video에 대한 Re-ID
    - spatial, temporal feature를 refine 후 cross view transformer로 multi-view aggregate

**Point Cloud Learning.**

- Point Transformer[[162](https://arxiv.org/abs/2012.09164)]: Point Transformer라는 새로운 transformer framework 제안
    - self-attention layer가 point set의 permutation에 invariant
    - 3D point cloud의 segment에 강함
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig33.jpg" width="60%" alt="Figure 33"></center>
    
- PCT[[161](https://arxiv.org/abs/2012.09688)]
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig34.jpg" width="60%" alt="Figure 34"></center>
    

## Low-level Vision

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig35.jpg" width="60%" alt="Figure 35"></center>

### Image Generation

- TransGAN[[38](https://arxiv.org/abs/2102.07074)]: 단계별로 feature map resolution를 점진적으로 높여, memory-friendly generator
    - Discriminator는 다양한 input을 받을 수 있도록 설계됨
    - 성능과 안정성 증가를 위해 grid self-attention, data augmentation, relative position encoding, modified normalization 도입
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig36.jpg" width="60%" alt="Figure 36"></center>
    
- ViTGAN[[163](https://arxiv.org/abs/2107.04589)]: INR 및 Self-modulated layernorm과 Euclidean distance 사용
    - Disc의 Lipschitzness를 위해 self-attention에 Euclidean distance 도입
    
    $$
    \mathrm{Attention}_h(X)=\mathrm{softmax}(\frac{d(XW_q,XW_k)}{\sqrt{d_h}})XW_v,\;\;\mathrm{where\;} W_q=W_k
    $$
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig37.jpg" width="60%" alt="Figure 37"></center>

    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig38.jpg" width="60%" alt="Figure 38"></center>
    
    
- VQGAN[[37](https://arxiv.org/abs/2012.09841)]: 직접적으로 tranformer에서 high res를 뽑기 어렵기 때문에, VQVAE의 latent에서 pixelCNN을 이용하여, autoregressive로 이미지를 생성하던 방식에 transformer 도입
- DALL-E[41]

### Image Processing

일반적으로, pixel을 그대로 활용하는 것보다 patch 단위로 transformer에 사용함

- **TTSR**[[39](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Learning_Texture_Transformer_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf)]: Reference-base SR (Refer image에서 low-resolution으로 relevant texture transfer)
    - Low-res input과 low-res refer에서 각각 $$q,k$$를 얻어, 연관성 $$r_{i,j}$$ 계산
        
        $$
        r_{i,j}=\left\langle \frac{q_i}{\lVert q_i\rVert},\frac{k_i}{\lVert k_i\rVert} \right\rangle
        $$
        
    - 연관성이 높은 feature $$h_i = \arg\max_j r_{i,j}$$ 만 뽑아, $$V$$ patch에 hard-attention
    - 이로 인해, high-res의 refer image의 feature가 low-res image에 transfer됨

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig39.jpg" width="60%" alt="Figure 39"></center>

- **IPT**[[19](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.pdf)]: Large dataset으로 pre-train된 Transformer 활용 (Multi image processing task 가능)
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig40.jpg" width="60%" alt="Figure 40"></center>
    
- SceneFormer[[165](https://ar5iv.labs.arxiv.org/html/2012.12556#bib.bib165)]: Indoor 장면 변환

## Video Processing

frame synthesis, action recognition, video retrieval, etc…

### High-level Video Processing

**Video Action Recognition.**

Video에서 human action을 identifying, localizing

Video에서 target이 아닌 다른 사람이나 사물은 critical하게 작용할 수 있음

- **Action Transformer**[[172](https://arxiv.org/abs/1812.02707)]: I3D를 이용하여, action에 대한 feature를 뽑고, intermidiate feature의 객체에 대한 ROI pool를 query, feature를 K, V로 하여 ROI의 객체에 대한 classification
- **Temporal Transformer Network**[[175](https://arxiv.org/abs/1906.05947)]: Class 내 variance를 줄이고, class 간 variance를 넓힘
- **Actor-transformer**[[178](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gavrilyuk_Actor-Transformers_for_Group_Activity_Recognition_CVPR_2020_paper.pdf)]: Group에서의 행동인식. 각 input의 object 들을 입력으로, embedding 후 Transformer. Output은 예측된 행동 (=classification)

**Video Retrieval.**

Video에서 비슷한 부분을 찾는 것이 관건임

- **Temporal context aggregation**[[179](https://openaccess.thecvf.com/content/WACV2021/papers/Shao_Temporal_Context_Aggregation_for_Video_Retrieval_With_Contrastive_Learning_WACV_2021_paper.pdf)]: 각 frame에서 feature를 뽑은 후, temporal transformer. 이후, contrastive learning으로 feature에 대해 pair의 pos, neg를 학습
- **Multi-modal Transformer for Video Retrieval**[[180](https://arxiv.org/abs/2007.10639)]

**Video Object Detection.**

영상에서 객체 탐지

일반적으로 전역에서 object를 detect하고, local에서 classification

- **MEGA**[181]
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig41.jpg" width="60%" alt="Figure 41"></center>
    

**Multi-task Learning.**

Video에는 target과 관계없는 frame이 많이 포함되어 있는 경우가 있음

- **Video Multitask Transformer Network**[[183](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CoView/Seong_Video_Multitask_Transformer_Network_ICCVW_2019_paper.pdf)]

### Low-level Video Processing

**Frame/Video Synthesis.**

Frame에 대한 extrapolation

- **ConvTransformer**[[171](https://arxiv.org/abs/2011.10185)]: feature embedding, position encoding, encoder, query decoder, synthesis feed forward network
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig42.jpg" width="60%" alt="Figure 42"></center>
    
    - Decoder의 Query set $$\mathcal{Q}$$는 $$t-0.5$$의 frame을 만들 때, $$\mathcal{J}$$(feature embedding한 값)가 $$t-1$$일 때와 $$t$$일 때의 average
    - SSFN은 UNet 형식의 network가 계단식으로 구성됨

**Video Inpainting.**

- **Learning Joint Spatial-Temporal Transformations for Video Inpainting**[[28](https://arxiv.org/abs/2007.10247)]
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig43.jpg" width="60%" alt="Figure 43"></center>
    

## Multi-Modal Tasks

Video2Text, Image2Text, Audio2Text, etc…

Transformer-based Model은 다양한 작업을 통합하기에 효율적인 아키텍쳐를 가지고, 방대한 데이터를 구축할 수 있는 capability를 가지고 있어, 확장성이 좋음

### **CLIP**[[41](https://arxiv.org/abs/2103.00020)]

Text-Encoder와 Image-Encoder에 대해 두 latent가 pair한지, contrastive learning으로 학습

- Text encoder: Standard Transformer with Masked self-attention
- Image encoder: ResNet 또는 Vision Transformer

N개의 image-text pair가 주어졌을 때, $$N$$개의 positive pair에 대해서는 cosine similarity가 크게, $$N^2-N$$개의 negative pair에 대해서는 cosine similarity가 작게 학습

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig44.jpg" width="60%" alt="Figure 44"></center>

- **Unified Transformer**[[189](https://arxiv.org/abs/2105.13290)]: Multi-Modal Multi-Task Model
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig45.jpg" width="60%" alt="Figure 45"></center>
    
- **VideoBERT**[[185](https://arxiv.org/abs/1904.01766)]: CNN으로 video feature 추출 후, Transformer로 caption 학습
    - Bert에 input만 CNN의 feature token으로 바뀐 느낌

**VQA**(Visual Question Answering)

- **VisualBERT**[[186](https://arxiv.org/abs/1908.03557)]
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig46.jpg" width="60%" alt="Figure 46"></center>
    

**VCR**(Visual Commonsense Reasoning)

- **VL-BERT**[[187](https://arxiv.org/abs/1908.08530)]
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig47.jpg" width="60%" alt="Figure 47"></center>
    

**SQA**(Speech Question Answering)

- **SpeechBERT**[[188](https://arxiv.org/abs/1910.11559)]
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig48.jpg" width="60%" alt="Figure 48"></center>
    

## Efficient Transformer

Transformer에서 높은 계산량과 메모리는 범용성을 저하시킴

효율성을 위해 compressing, accelerating transformer model 소개

- Pruning
- Low-rank decomposition
- Knowledge distillation
- Network quantization
- Compact architecture design

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig49.jpg" width="60%" alt="Figure 49"></center>

### Pruning and Decomposition

- **Michel**[[45](https://arxiv.org/abs/1905.10650)]: Attention에서 task와 layer에 따라, 필요한 head의 수가 다를 수 있음
    - 중복된 relation을 볼 수 있기 때문
    - 따라서, 각 head가 최종 output에 미치는 영향을 estim 후 pruning
    - Head를 하나씩 제거해가며, BLEU score를 측정
- **BERT LTH**[[190](https://arxiv.org/abs/2005.00561)]: Bert의 LTH를 general redundancy, task-specific redundancy에서 분석
    - BERT에서도 LT에 해당하는 good sub-network가 존재하고, attention과 FFN layer에서 모두 높은 compression이 가능
- **Patch Slimming**[[192](https://arxiv.org/abs/2106.02852)]: Layer에서 효과가 없는 patch들을 삭제해나가며 계산
    - attention의 경우, patch간의 연관성을 학습
    - 연관성이 없는 patch가 있을 수 있음
    - output layer부터 이전 layer에서의 연관성 영향을 보고, patch를 선택
- **Vision Transformer Pruning**[[193](https://arxiv.org/abs/2104.08500)]:
    1. Sparsity Regularization ($$L_1$$)
    2. Pruning dimension of Linear Projection
    3. Fine-tuning
- **LayerDrop**[[204](https://arxiv.org/abs/1909.11556)]: 훈련 중엔 regularization이 가능하고, inference에선 prune이 됨
    - 즉, training 시에, layer를 random delete 하며 훈련하기 때문에 model이 pruning에 robust 함
    - Inference 시에, 원하는 subnetwork를 사용할 수 있음
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig50.jpg" width="60%" alt="Figure 50"></center>
    
- **Structured pruning**[[206](https://arxiv.org/abs/1910.04732)]: Low-rank로 parameterizing하며 훈련을 진행하고, rank가 낮은 component들은 삭제해가며 pruning
- **[LLM-Pruner](https://arxiv.org/abs/2305.11627):** gradient information을 기반으로 non-critical coupled structure를 prune

### Knowledge Distillation

Large teacher network $$\rightarrow$$ shallow architecture를 가진 student network로 distillation

- **XtremeDistil**[[210](https://arxiv.org/abs/2004.05686)]: Bert $$\rightarrow$$ small model로 distillation
- **Minilm**[[211](https://arxiv.org/abs/2002.10957)]: self-attention layers의 output을 student가 mimic
    - teacher와 student에서 $$Q\cdot K, V$$에 대해 scaled dot-product를 이용하여 distillation
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig51.jpg" width="60%" alt="Figure 51"></center>)
    

- **Jia**[[213](https://arxiv.org/abs/2107.01378)]: patch 단에서 manifold distillation을 통해 fine-grained distillation 달성
    
    $$
    \mathcal{M}(\psi(F_S))=\psi(F_S)\psi(F_S)^T, \mathrm{\;\;where\;} \psi = \mathrm{reshape} \\
    \mathcal{L}_{mf}=\lVert\mathcal{M}(\psi(F_S)) - \mathcal{M}(\psi(F_T))\rVert^2_F
    $$
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig52.jpg" width="60%" alt="Figure 52"></center>
    

### Quantization

기존 양자화 방법들

1. [Searching for low-bit weights in quantized neural networks](https://arxiv.org/abs/2009.08695)
2. [Profit: A novel training method for sub-4-bit mobilenet models](https://arxiv.org/abs/2008.04693)
3. [Riptide: Fast end-to-end binarized neural networks](https://cowanmeg.github.io/docs/riptide-mlsys-2020.pdf)
4. [Proxquant: Quantized neural networks via proximal operators](https://arxiv.org/abs/1810.00861)

- **Compressing transformers with pruning and quantization**[[222](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00413/107387/Compressing-Large-Scale-Transformer-Based-Models-A)]
- **Post-Training Quantization for Vision Transformer**[[224](https://arxiv.org/abs/2106.14156)]

### Compact Architecture Design

Distillation 이외에 직접적으로 compact한 model을 설계

- **Lite Transformer**[[225](https://arxiv.org/abs/2004.11886)]: LSRA(Long-Short Range Attention)
    - Attention의 각 head 마다 long, short에 대한 relation modeling
- **hamburger**[[226](https://arxiv.org/abs/2109.04553)]: self-attention layer 대신 matrix decomposition 사용
    - token 간의 종속성을 decomposition을 통해 더 명확히 알 수 있고, compact 함
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig53.jpg" width="60%" alt="Figure 53"></center>
    
- NAS 방법 사용: [[ViTAS](https://arxiv.org/abs/2106.13700)], [[227](https://arxiv.org/abs/1910.14488)], [[228](https://arxiv.org/abs/1901.11117)], [[Bossnas](https://arxiv.org/abs/2103.12424)]

시간 복잡도를 $$O(N)$$으로 줄이는 방법

- **Transformers are RNN**[[230](https://arxiv.org/abs/2006.16236)]: Linearized Attention
    - Softmax로 각 token의 relation을 계산하기 때문에 $$O(N^2)$$임
    - Softmax 전에 $$Q, K$$의 similarity function을 바꿀 수 있음
    
    $$
    V'_i=\frac{\sum^N_{j=1}\phi(Q_i)\phi(K_i)^TV_j}{\sum^N_{j=1}\phi(Q_i)\phi(K_i)^T} \rightarrow
    V'_i=\frac{\phi(Q_i)\sum^N_{j=1}\phi(K_i)^TV_j}{\phi(Q_i)\sum^N_{j=1}\phi(K_i)^T}
    $$
    
    - Sum 부분은 Query와 관계가 없기 때문에 시간 복잡도 $$O(N)$$으로 계산 가능
    - 공간 복잡도 또한 분모의 값을 미리 저장할 수 있기 때문에 $$O(N)$$으로 볼 수 있음
    - Softmax와 dot-product를 대체할 함수는 similarity function 값이 양수가 되어야 함
    - 이에 대한 feature representation $$\phi$$는 다음과 같이 설정됨
        
        $$
        \phi(x)=\mathrm{elu}(x)+1=
        \begin{cases}
        x+1, & x>0 \\
        \exp(x), & x\le0
        \end{cases}
        $$
        
    - Causal Masking도 triangle matrix일 필요가 없고, sum의 범위를 $$N$$에서 $$i$$로 바꾸면 됨
- **Big Bird**[[232](https://arxiv.org/abs/2007.14062)]: 각 token을 graph의 vertex로 간주하여 inner-product를 edge로 정의
    - graph 이론의 sparse graph를 통해 transformer의 dense graph를 근사하여 $$O(N)$$달성
    
    <center><img src="/assets/images/cv/vit/survey/survey-vit_fig54.jpg" width="60%" alt="Figure 54"></center>
    

### 전체 Efficient 순서

<center><img src="/assets/images/cv/vit/survey/survey-vit_fig55.jpg" width="60%" alt="Figure 55"></center>

# CONCLUSIONS AND DISCUSSIONS

Transformer는 CNN에 비해 경쟁력 있는 성능과 potential을 가지고 있음

Transformer는 광범위한 Vision Task에서 좋은 성능을 달성했음

## Challenges

Transformer의 Robustness와 generalization은 challenge point 임

CNN에 비하여, transformer는 inductive bias가 낮기 때문에 많은 dataset이 필요함

즉, dataset의 quality가 성능에 영향을 미침

CNN 같이 locality iductive bias가 없는 Transformer가 잘 동작하는 이유는 보통 직관적으로 분석함

1. Large scale Dataset이 inductive bias를 능가함
2. PE를 통해, 이미지에서 중요한 position 정보를 유지함
3. Over-parameterization 도 하나의 이유가 될 수 있음
    1. 일반적으로 256x256 resolution의 경우 input의 경우의 수가 5천만개 정도임

Transformer는 계산량이 높음

ex) ViT는 inference에 180억 FLOPs. 하지만, GhostNet 등은 6억 FLOPs

그리고 NLP에 맞게 디자인되어 있는 경우가 많아서 개선이 필요함

## Future Prospects

1. Low Cost, High performance의 Transformer
    1. Real-world에 application 가능성을 높여줌
2. NLP model들과 비슷하게 multiple task에 대해 동작하는 model 개발

# Appendix

## Self-attention for Computer Vision

Self-attention이 Transformer에서 중요한 역할을 함

CNN의 low-resolution, long-range를 보는 output 단의 layer와 비슷한 역할로 간주 가능

따라서, self-attention에 대해 조사

### Image Classification.

Classification에서 trainable attention은 2개로 분류될 수 있음

1. Hard attention
    1. local 또는 전체 이미지에 대한 attention
2. Soft attention
    1. SENet과 같이, channel-wise conv를 통해, feature들을 reweight

### Semantic Segmentation.

### Object Detection.

### Other Vision Tasks.