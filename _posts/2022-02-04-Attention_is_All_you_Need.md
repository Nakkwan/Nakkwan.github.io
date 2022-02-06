---
title: Transformer(Attention is all you need)
tags:
    - Deep Learning
    - NLP
    - Transformer
---

Summary of Transformer paper(Attention is all you need) <br>

<!--more-->

#### Introduction
Transformer은 2017년 NIPS에 게재된 논문으로, RNN기반 모델이 SOTA를 이루던 기존 NLP에 새롭게 SOTA를 달성한 모델입니다. 최근 NLP분야에서 좋은 성능을 내는 모델들은 transformer를 기반으로 한 모델들이 많습니다. 대표적으로는 2018년 google에서 공개한 BERT와 2020년 OpenAI에서 공개한 GPT-3가 있습니다. <br>
기존 NLP분야에서 많이 쓰이던 RNN을 이용한 방식에는 한계점이 있었습니다. <br> 
우선 Seq-to-Seq의 경우 vanishing gradient와 fixed-size context vector의 문제가 나타납니다. RNN 방식을 사용하기 때문에 squence의 길이가 길어지면, gradient가 소실되어 훈련이 잘 안되는 문제가 있습니다. 예를 들어, "I am a good student"라는 문장에서 student라는 단어를 비워놓고 예측한다고 가정할 때, "I"에 관한 정보는 "good"에 비해 student에서 멀리 떨어져 있기 때문에 관계를 학습하기 더 힘듭니다. <br>
이에 decoder에서 매번 encoder를 참조(가중치 계산)하여 결과를 예측하는 attention 방식이 등장하였습니다. 모든 decoder step에서 encoder를 참조하기 때문에 앞서 말한 문제점은 완화되지만, 여전히 RNN 방식을 사용한다는 한계점이 존재했습니다. Time step으로 인해 병렬처리에 한계가 있기 때문에 연산 속도가 느린 단점이 남아있고, 또한 encoder에서 decoder로 context를 전달하는 context vector의 크기가 정해져 있기 때문에 정보의 손실이 일어날 수 있는 문제점이 있습니다. <br>
따라서 기존 RNN 방식이 아닌 새로운 architecture, transformer가 등장했습니다. Transformer는 attention을 Seq-to-Seq에서와 같이 추가 구조로써 사용하는 것이 아니라 attention을 중심으로 model architecture를 구성했습니다.<br>

#### Attention
Attention은 decoder에서 예측할 output과 encoder의 sequence들과의 관계가 어느정도 있는지 판단하여 output을 예측하기 위한 방법입니다. 앞서 예를 들었던 "I am a good student"의 경우 'am'보다 'I'가 'student'라는 단어를 예측하는 것에 더욱 연관성이 있기 때문에 'I'라는 단어에 더욱 집중(attention)하여 'student'를 예측하는 것에 사용합니다. 이에 관한 내용은 아래의 그림에 나타나 있습니다. <br>

Attention은 기본적으로 Query, Key, Value를 사용하여 attention value를 구합니다. <br>
- Query: t시점에서의 decoder의 hidden state <br>
- Key: 모든 시점의 encoder의 hidden state <br>
- Value: 모든 시점의 encoder의 hidden state <br>

Attention의 machanism 중 하나인 dot-product machanism의 경우, Query와 Key에 dot-product를 취합니다(Attention Score). Attention score에 softmax를 적용하여 score의 합이 1이 되도록 해줍니다(Attention Weight). Attention weight는 각 encoder의 step이 decoder의 hidden state와 얼마나 연관성이 있는지를 나타냅니다. Attention value는 attention weight에 Key를 weight sum을 적용하여 구합니다. 가중치가 높은 step의 hidden state일수록 attention value에 기여하는 바가 높고, attention value는 decoder의 hidden state에 concatenate되어 사용되기 때문에 해당 step에 더 집중하여 output을 얻을 수 있습니다. <br>
Attention의 전체적인 진행 과정은 Fig. 와 같습니다.<br>

###### Self-Attention
Transformer에서는  self-attention이라는 method가 사용됩니다. Attention과 전체적인 진행 과정은 동일하지만 self-attention은 query, key, value가 모두 encoder의 hidden state라는 점이 다릅니다. Attention은 예측할 decoder의 output과 encoder input의 관계(변역이라면 언어 사이의 관계, 빈칸 예측이라면 단어 사이의 관계)를 얻기 위해 수행하지만, self-attention은 input으로 들어가는 sequence의 이해를 위해서 사용이 됩니다. <br>

### Transformer
Transformer(Attention is all you need)는 제목 그대로 RNN이 아닌 attention machanism만 사용하여 architecture를 구성합니다. Architecture는 크게 <br>
- Positional Encoding <br>
- Encoder <br>
- Decoder <br>
의 범주로 나눌 수 있습니다.

###### Positional Encoding

##### Reference <br>