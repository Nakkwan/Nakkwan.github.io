---
layout: default
title: Switch Transformer
nav_order: "2025.04.06"
parent: Transformer
grand_parent: Paper Review
permalink: /docs/paper_review/transformer/switch_transformer_2025_04_06
---

# Switch Transformer
{: .no_toc}
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)


Table of contents
{: .text-delta }
1. TOC
{:toc}


## Summary
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


## Switch Transformer


## Scaling Properties


## Downstream Results


## Parallelism


## 
