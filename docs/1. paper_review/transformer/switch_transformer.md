---
layout: default
title: Switch Transformer
nav_order: "2025.04.06"
parent: Transformer
grand_parent: Paper Review
permalink: /docs/paper_review/transformer/switch_transformer_2025_04_06
---

# **Switch Transformer**
{: .no_toc}
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)


Table of contents
{: .text-delta }
1. TOC
{:toc}


## **Summary**
일반적으로, ML model은 입력에 대해 동일한 parameter를 사용하여 추론을 수행한다. 하지만 대규모 데이터셋에 대해 더 유연하고 효율적인 계산을 위해 MoE(Mixture of Experts)가 쓰인다. MoE는 입력에 대해 layer의 하위 parameter를 sparsely-activated하여 연산을 수행한다. 많은 parameter 수에 비해 일정한 활성화 parameter를 가져, 효율적인 추론을 수행한다. 하지만, 어떤 parameter를 활성화시킬지에 대한 router 설계가 복잡하다. Switch Transformer는 이런 routing algorithm을 단순화하여 직관적이고 효율적인 모델을 설계한다. <br>
Switch Transformer의 contribution은 <br>
- MoE의 단순화<br>
- 제한된 resource에서 성능 향상 및 T5 모델의 학습 속도를 7배 향상<br>
- 작은 모델로 좋은 성능의 distillation <br>
- Finetuning technique 향상<br>
  - bloat16으로 학습<br>
  - 더 많은 export를 위한 initialize scheme<br>
  - Finetune 및 multi-task 학습을 위한 expert regularization<br>
- parallelism을 통한 data, model, export의 효율적인 결합<br>
결과적으로, switch transformer는 scalable하고 effective한 MoE Transformer 모델로, 기존 모델보다 직관적이고 stable하며 같은 size의 dense model에 비해 efficient하다.


## **Switch Transformer**
Switch Transformer는 parameter를 늘리면서, 효율적인 계산을 수행하는 것을 목표로 한다. 수행 연산량과 무관하게 parameter의 수가 model scaling의 중요한 축이라고 생각하며 sparsely activated model을 설계한다. 

<center>
<div class="img-with-text">
<img src="/assets/images/papers/transformer/switch_fig1.jpg" width="75%" alt="Figure 1">
<p><b>Figure 1.</b> Illustration of a Switch Transformer encoder block.</p>
</div>
</center>

### **Simplifying Sparse Routing**
기존 [TopK Routing MoE](https://arxiv.org/abs/1701.06538)는 입력에 대해 $$W_g$$를 곱하고 Noisy Top-K를 적용하여 Softmax를 통해 gating probability를 얻는다. 학습 과정에서 $$W_g$$는 activated export의 결정을 학습하고, Top-k는 효율성은 유지하면서도 전체 모델 용량은 크게 확장하도록 한다. 학습 초기에 export imbalance를 해결하기 위해 noise를 추가하여 exploration을 유도한다. 논문의 저자는 각 token이 2개 이상의 export를 선택하여 학습하여야 효과적이라는 직관을 제시하고, 후속 연구에서 export의 수에 따라 높은 k가 중요하다는 것을 보인다. <br>

Switch Transformer에서는 이와 반대로, 1개의 export만 활성화하여 router를 단순화하며 성능을 유지한다. (Switching Layer) <br>
Switching Layer는 아래와 같은 이점이 있다. <br>
- router computation 감소 <br>
- expert batch size 감소 <br>
- communication cost 감소 <br>

### **Efficient Sparse Routing**
##### **Distributed Switch Implementation**
일반적으로 tensor의 shape나 연산은 compile 시, 결정되지만 switch transformer는 router 때문에 동적이다. 따라서 균등한 연산 할당과 마진을 위해 아래와 같이, export capacity를 설정한다. 

$$
\begin{gather}
  \text{expert capacity} = \frac{\text{tokens per batch}}{\text{number of experts}} \times \text{capacity factor}
\end{gather}
$$

<center>
<div class="img-with-text">
<img src="/assets/images/papers/transformer/switch_fig2.jpg" width="75%" alt="Figure 2">
<p style="text-align:justify"><b>Figure 2.</b> Illustration of token routing dynamics. Expert Capacity는 각 export 당 처리할 수 있는 batch size (token 수)를 의미하며, Capacity Factor는 Capacity를 계산할 때 사용되고 어느정도 overflow를 허용하기 위한 buffer와 같으며, 보통 1.0 ~ 1.25가 사용된다. Capacity를 넘은 token들의 경우, drop 되며 다음 layer에 residual 방식으로 전달된다.</p>
</div>
</center>

너무 큰 expert capacity는 높은 계산량과 메모리를 사용하게 된다. 따라서, dropped token을 낮추며 expert capacity도 낮추기 위해 switch transformer는 auxiliary load balancing loss를 사용한다. 

##### **A Differentiable Load Balancing Loss**
각 switch layer에서 export에 대한 load-balancing을 위해 auxiliary loss를 사용하고, 전체 loss에 더하여 학습을 진행한다. <br>
$$N$$개의 expert와 batch $$B$$, token $$T$$에 대해, $$f_i, P_i$$는 각각 expert $$i$$에 할당된 token과 router probability의 비율일 때,

$$
\begin{gather}
  \mathcal{L} = \alpha\cdot N \cdot\sum^N_{i=1}f_i\cdot P_i
\end{gather}
$$

### **Putting It All Together: The Switch Transformer**
Switch Transformer에서 설계한 model의 구조는 및 다른 MoE Transformer와의 비교는 아래와 같다.

<center>
<div class="img-with-text">
<img src="/assets/images/papers/transformer/switch_fig3.jpg" width="90%" alt="Table 1">
<p style="text-align:justify"><b>Table 1.</b> Switchmodel design and pre-training performance.</p>
</div>
</center>

<center>
<div class="img-with-text">
<img src="/assets/images/papers/transformer/switch_fig4.jpg" width="90%" alt="Table 2">
<p style="text-align:justify"><b>Table 2.</b> Benchmarking Switch Transformer & MoE Transformer. Switch Transformer는 다른 MoE보다 speed-quality가 좋고, Memory 사용량이 작다. 또한 낮은 capacity factors에서도 좋은 성능을 보인다.</p>
</div>
</center>

### **Improved Training and Fine-Tuning Techniques**


## **Scaling Properties**


## **Downstream Results**


## **Parallelism**


