---
title: Transformer(Attention is all you need)
tags:
  - Deep Learning
  - NLP
  - Transformer
  - Paper
---

Summary of Transformer paper(Attention is all you need) <br>

<!--more-->

#### Introduction

Transformer은 2017년 NIPS에 게재된 논문으로, RNN기반 모델이 SOTA를 이루던 기존
NLP에 새롭게 SOTA를 달성한 모델입니다. 최근 NLP분야에서 좋은 성능을 내는 모델들
은 transformer를 기반으로 한 모델들이 많습니다. 대표적으로는 2018년 google에서공
개한 BERT와 2020년 OpenAI에서 공개한 GPT-3가 있습니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_Research_trend.png" width="600"></center>
<center><em>Fig 1. Research trend</em></center>
</p>

기존 NLP분야에서 많이 쓰이던 RNN을 이용한 방식에는 한계점이 있었습니다. <br>
우선 Seq-to-Seq의 경우 vanishing gradient와 fixed-size context vector의 문제가 나타납니다. RNN 방식을 사용하기 때문에 squence의 길이가 길어지면, gradient가 소실되어 훈련이 잘 안되는 문제가 있습니다. 예를 들어, "I am a good student"라는 문장에서 student라는 단어를 비워놓고 예측한다고 가정할 때, "I"에 관한 정보는 "good"에 비해 student에서 멀리 떨어져 있기 때문에 관계를 학습하기 더 힘듭니다. <br>
이에 decoder에서 매번 encoder를 참조(가중치 계산)하여 결과를 예측하는 attention 방식이 등장하였습니다. 모든 decoder step에서 encoder를 참조하기 때문에 앞서 말한 문제점은 완화되지만, 여전히 RNN 방식을 사용한다는 한계점이 존재했습니다. Time step으로 인해 병렬처리에 한계가 있기 때문에 연산 속도가 느린 단점이 남아있고, 또한 encoder에서 decoder로 context를 전달하는 context vector의 크기가 정해져 있기 때문에 정보의 손실이 일어날 수 있는 문제점이 있습니다. <br>
따라서 기존 RNN 방식이 아닌 새로운 architecture, transformer가 등장했습니다. Transformer는 attention을 Seq-to-Seq에서와 같이 추가 구조로써 사용하는 것이 아니라 attention을 중심으로 model architecture를 구성했습니다.<br>

#### Attention

Attention은 decoder에서 예측할 output과 encoder의 sequence들과의 관계가 어느정도
있는지 판단하여 output을 예측하기 위한 방법입니다. 앞서 예를 들었던 "I am a good
student"의 경우 'am'보다 'I'가 'student'라는 단어를 예측하는 것에 더욱 연관성이
있기 때문에 'I'라는 단어에 더욱 집중(attention)하여 'student'를 예측하는 것에 사
용합니다. 이에 관한 내용은 아래의 그림에 나타나 있습니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_attention.jpg" width="600"></center>
<center><em>Fig 2. Attention</em></center>
</p>
Attention은 기본적으로 Query, Key, Value를 사용하여 attention value를 구합니다. <br>

- Query: t시점에서의 decoder의 hidden state <br>
- Key: 모든 시점의 encoder의 hidden state <br>
- Value: 모든 시점의 encoder의 hidden state <br>

Attention의 machanism 중 하나인 dot-product machanism의 경우, Query와 Key에
dot-product를 취합니다(Attention Score). Attention score에 softmax를 적용하여
score의 합이 1이 되도록 해줍니다(Attention Weight). Attention weight는 각
encoder의 step이 decoder의 hidden state와 얼마나 연관성이 있는지를 나타냅니다.
Attention value는 attention weight에 Key를 weight sum을 적용하여 구합니다. 가중
치가 높은 step의 hidden state일수록 attention value에 기여하는 바가 높고,
attention value는 decoder의 hidden state에 concatenate되어 사용되기 때문에 해당
step에 더 집중하여 output을 얻을 수 있습니다. <br>

Translation을 수행할 때 계산된 두 언어 사이의 attention score를 나타내면 다음과
같습니다. Translate과정에서 연관된 단어들 사이의 attention이 높은 것을 확인할 수
있고, attention이 training에서 제대로 동작하고 있다는 것을 알 수 있습니다.<br>

<p>
<center><img src="/images/Transformer/Transformer_Result_Attention.png" width="400"></center>
<center><em>Fig 3. Attention result weight</em></center>
</p>
##### Self-Attention

Transformer에서는 self-attention이라는 method가 사용됩니다. Attention과 전체적인
진행 과정은 동일하지만 self-attention은 query, key, value가 모두 encoder의
hidden state라는 점이 다릅니다. Attention은 예측할 decoder의 output과 encoder
input의 관계(변역이라면 언어 사이의 관계, 빈칸 예측이라면 단어 사이의 관계)를 얻
기 위해 수행하지만, self-attention은 input으로 들어가는 sequence의 이해를 위해서
사용이 됩니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_Compare_Architecture.png" width="500"></center>
<center><em>Fig 4. Compare architecture</em></center>
</p>

### Transformer

Transformer(Attention is all you need)는 제목 그대로 RNN이 아닌 attention
machanism만 사용하여 architecture를 구성합니다. Architecture는 크게 <br>

- Positional Encoding <br>
- Encoder <br>
- Decoder <br> 의 범주로 나눌 수 있습니다. <br>
  <p>
  <center><img src="/images/Transformer/Transformer_architecture.jpg" width="300"></center>
  <center><em>Fig 5. Architecture</em></center>
  </p>

##### Positional Encoding

자연어 처리에서 단어를 vectorization할 때 embedding을 진행하게 됩니다. One-hot vector의 형식으로 corpus의 단어들을 모두 표현하려면 vector의 공간이 너무 커지게
됩니다. (10만개의 단어가 있다면, dimension 또한 10만) 따라서 corpus의 크기에 상관없이 일정한 차원의 벡터에 단어값을 설정하여 dense representation으로 만들어주는 것을 embedding이라고 합니다. 기존의 RNN에선 훈련에 사용될 vector는 embedding만 사용했지만 transformer에서는 positional encoding(이하 PE)이라는 추가적인 작업이 필요합니다. <br>

PE은 단어의 위치 정보를 model에 알리는 방식입니다. 단어의 위치를 알 수 없다면 문
장의 의미가 완적히 달라질 수 있기 때문입니다. 예를 들어, "귤은 과일이다"와 "과일
은 귤이다"의 의미는 완전히 다릅니다. RNN은 위치에 따라 입력된 단어를 순차적으로
받아 처리하기 때문에 각 단어가 위치 정보를 자동적으로 모델이 학습하기 때문에 자
연어 처리에서 강력합니다.(Sequentiality inductive bias가 있다고 말합니다.) 그러
나 Transformer는 입력된 단어를 순차적으로 받지 않기 때문에 다른 방식(PE)으로 위
치 정보를 알립니다. 그림 n과 같이 PE는 일부 값을 입력 임베딩 매트릭스와 합산하여
수행됩니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_PE_Add.jpg" width="500"> </center>
<center><em>Fig 6. Add positional encoding</em></center>
</p>
PE를 구현하기 위해서 다음과 같은 조건은 지켜져야 합니다. <br>

1. 각 position 별로 vector값이 다 달라야한다. <br>
2. 문장의 길이가 다르더라도 적용할 수 있는 method여야 한다. <br>
3. 문장의 길이가 달라도각 position 사이의 길이가 일정해야한다. <br>
   - 예를 들어, 단어가 5개인 문장의 포지션 간의 간격이 5라면, 단어가 20개인 문장
     의 포지션 간의 간격도 5여야한다. <br>
   - 즉, 다음 단어로 이동할 때 거리가 일정한 method여야 한다.
4. 모델의 일반화가 가능해야한다. <br>
   - 단어가 10개인 문장이 나와도, 1000개인 문장이 나와도 적용될 수 있어야한다.
     <br>
   - 예를 들어, position 값이 일정 범위 이내에 있어야 한다. <br>
   - transformer에서는 PE값을 입력에 더해주기 때문에 너무 큰 값이 들어가면 안됨
     <br>
5. 같은 문장이라면 매번 같은 값이 나와야한다.<br>

Transformer에서는 sin, cos 함수를 사용하여 PE를 구현합니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_PE.jpg" width="600"></center>
<center><em>Fig 7. Positional Encoding</em></center>
</p>

기본적으로 PE는 embedding된 input과 더해져야하기 떄문에 사이즈가 같습니다. 따라
서 (sequence length, embedding dimension(=$$d_{model}$$))차원의 matrix가 됩니다.
행은 sequence에서 단어의 위치를 나타내고, 열은 position vector의 index를 의미합
니다. PE의 각 elements는 <br>

$$
\begin{align}
\sin(\pi i/2^{j}) \rightarrow \sin(x_{i}w_{0}^{j/d_{model}}),\;\;i=0 \sim d_{model}-1,\;\;j=0\sim seq_{len}-1 \\
\mathbb{PE} = [v^{(0)},\ldots,v^{seq_{len}-1}]^{T},\quad v^{(i)}=[\cos(w_{0}x_{i}),\; \sin(w_{0}x_{i}),\; \ldots,\; \cos(w_{n-1}x_{i}),\; \sin(w_{n-1}x_{i}) ] \\
\end{align}
$$

로 나타납니다. <br><br> 삼각함수를 사용했기 때문에 기본적으로 값의 범위가 -1 ~ 1
로 제한되어 있고, 주기함수이기 때문에 sequence의 길이가 달라져도 적용할 수 있습
니다. 또한 같은 position에서는 같은 value가 나오기 때문에 1,2,4,5 조건을 다 만족
할 수 있습니다. 조건 3을 만족시키기 위해서 $$PE(x + \Delta x)=PE(x) \cdot T(\Delta x)$$ 를 만족해야하는데 이는 회전 변환으로 만족시킬 수 있습니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_T_matrix.jpg" width="450"></center>
<center><em>Fig 8. Distance T </em></center>
</p>
PE가 기본적으로 cos, sin의 조합으로 이루어져 있기 때문에 T를 회전변환 블록으로 나타낼 수 있습니다. 회전변환 블록은 <br>

$$
\begin{align}
    \begin{pmatrix} \cos (w_0 + \Delta x) \\ \sin (w_0 + \Delta x) \end{pmatrix}
    = \begin{pmatrix} \cos \Delta x & -\sin \Delta x \\ \sin \Delta x & \cos \Delta x \end{pmatrix}
    \begin{pmatrix} \cos w_0  \\ \sin w_0 \end{pmatrix}
\end{align}
$$

와 같이 나타낼 수 있고, 전체 회전변환 $$T$$는 <br>

$$
\begin{align}
    T(\Delta x) =
    \begin{bmatrix}
        \begin{pmatrix} \cos (w_0 \Delta x) & -\sin (w_0 \Delta x) \\ \sin (w_0 \Delta x) & \cos (w_0 \Delta x) \end{pmatrix} & \cdots & 0 \\
        \vdots & \ddots & \vdots \\
        0 & \cdots & \begin{pmatrix} \cos (w_{n-1} \Delta x) & -\sin (w_{n-1} \Delta x) \\ \sin (w_{n-1} \Delta x) & \cos (w_{n-1} \Delta x) \end{pmatrix}
    \end{bmatrix}
\end{align}
$$

로 표현할 수 있습니다. 따라서 각 position 사이의 간격이 동일하다고 말할 수 있으며 PE의 모든 조건을 만족할 수 있습니다.<br> Sequence length가 50,$$d_{model}$$이 128인 Positional Encoding을 나타내면 Fig n.과 같습니다. <br>

<p>
<center><img src="/images/Transformer/Transforemr_PE_heatmap.jpg" width="400"></center>
<center><em>Fig 9. Positional Encoding heatmap</em></center>
</p>

##### Encoder

우선 인코더는 위치 인코딩과 입력 임베딩을 입력으로 받습니다. 따라서 인코더는 한번에 하나의 문장을 입력으로 받을 수 있으며 입력은 ($$Seq_{len},\;d_{model}$$) 크기의 행렬입니다. 인코더는 두 개의 중요한 하위 계층인 1) Multi-head Attention, 2) Feedforward Network로 구성됩니다. 또한, normalization layer는 각 하위 layer 뒤에 쌓입니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_encoder.png" width="400"></center>
<center><em>Fig 10. Encoder</em></center>
</p>

1.  Seq2seq의 attention의 경우 Query는 decoder의 hidden state고 Key와 Value는
    encoder의 hidden state입니다. 그러나 Transformer는 서로 다른 언어를 서로 번
    역하는 모델이기 때문에 입력된 문장을 이해하는 것이 중요합니다. 따라서
    Transformer의 self-Attention 하위 계층은 모두 Query, Key 및 Value와 동일한값
    을 갖습니다. Self-Attention은 input으로 입력된 sequence에서 각 단어의 연관정
    도를 측정할 수 있습니다. 예를 들어, "나는 손흥민이 훌륭한 축구선수라는 것을
    안다"라는 문장에서 '축구 선수'가 '나'보다 '손흥민'과 더 관련이 있음을 측정할
    수 있습니다. Transformer의 self-attention도 multi-head attention을 사용합니
    다. Multi-head Attention은 여러 output을 가지며 output을 concatenate하여최종
    output을 도출합니다. 상세한 동작 과정은 Fig. n과 같습니다.<br>

<p>
<center><img src="/images/Transformer/Transformer_Multi-head.jpg" width="650"></center>
<center><em>Fig 11. Multi-head Attention</em></center>
</p>

Self-Attention이 수행되기 전에 Query, Key, Value에 각각 다른 weight
matrix($$W^{Q},\;W^{K},\;W^{V}$$)가 곱해지고 self-attention의 input에 제공됩니다
. Multi-head Attention이기 때문에 각 Query, Key, Value에는 head 수만큼 weight
matrix가 있습니다. 논문에서는 head의 수를 8로 설정했기 때문에 8개의 weight
matrix가 있지만 각 weight matrix의 column size가 ($$d_{model},\;d_{model} /head_{num}$$)이므로 최종 parameter 수는 변경되지 않습니다. 각 weight의 학습이 독립적으로 수행되기 때문에 input sequence의 다양한 연관성을 학습할 수 있습니다. 예를 들어, "그는 수학자이자 철학자이다."라는 문장에서 '그'는 '수학자'와 '철학자' 모두와 관련이 있지만 multi-head가 아닌 경우 둘 중 하나의
연관성만 잡아낼 가능성이 있습니다. 그러나 multi-head는 여러 측면(weight)의 input을 보기 때문에 여러 연관성을 잡아낼 가능성이 커지게 됩니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_QKV.jpg" width="600"></center>
<center><em>Fig 12. Query, Key, Value</em></center>
</p>

그리고 Transformer는 Scaled Dot-Product Attention을 사용합니다. Dot-Product Attention을 사용하는 이유는 이 외에도 많은 Attention 방법이 있지만 가장 계산하기 쉬운 방법이기 때문입니다. 그리고 계산 과정에서 activation value가 너무 커지면 softmax 과정에서 gradient가 너무 작아져 학습에 방해가 되기 때문에 scaling을 진행해줍니다. 즉, scaling은 activation value를 줄이는 데 사용됩니다. Input으로
sequence을 받을 때 corpus에 없는 단어나 의미 없는 단어는 attention에서 학습되면 안됩니다. 따라서 attention에서 mask technique이 사용됩니다. 이러한 단어는 tokenization할 때 \<pad\>라는 token으로 대체됩니다. \<pad> token은 attention과정 에서 masking이 적용됩니다. Masking은 softmax를 통과하기 전에 attention score에아 주 작은 수(−∞)를 더합니다. (Fig n) 매우 작은 수는 softmax를 거치면서 0으로 수렴하므로 어떤 단어와도 연관되지 않도록 학습이 진행됩니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_paddingmask.jpg" width="250"></center>
<center><em>Fig 13. Padding mask</em></center>
</p>

이렇게 계산된 각 head의 attention은 concatenate되고 다음 sublayer로 들어가기 위
해 weight matrix $$W^{O}$$가 곱해집니다. <br>

2.  Feedforward layer는 Transformer의 두 번째 sublayer입니다. Feedforward
    network는 일반 MLP와 동일합니다. Input은 attention의 $$d_{model}$$
    diemension이고 중간 hidden layer의 dimension은 2048입니다. <br> 각 sublayer
    의 output size는 연결 용이성을 위해 input size와 동일하며 각 layer는
    residual 및 normalization layer를 사용하여 연결됩니다. Residual connection은
    다음과 같이 layer의 input과 ouptut을 더합니다. Residual은 학습을 더 효과적으
    로 수행할 수 있게 합니다. 기본적으로 한 번에 많은 정보를 학습하던 기존 방식
    과 다르게 residual은 이전 layer output을 추가하여 학습되므로 이전에 학습한정
    보에 따라 모델은 추가적인 정보만 학습하면 되기때문에 network training이 조금
    더 쉬워집니다. 또한, 적은 양의 학습량으로 인해 네트워크가 깊어져도 훈련이용
    이합니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_Residual.jpg" width="250"></center>
<center><em>Fig 14. Residual connection</em></center>
</p>

그리고 normalization layer를 사용하는데 normalization도 마찬가지로 학습을 용이하
게 하는 데 도움이 됩니다. Residual method의 경우 input과 output을 더하기 때문에
variance가 2배가 되어 activation value가 매우 커질 수 있지만 normalization는 이
를 방지해줍니다. Normalization training이 좋은 이유는 여전히 논쟁의 여지가 있지
만 기본적으로 ICS(Internal Covariance Shift)와 loss landscape smoothing 때문이라
는 연구가 있습니다. <br>

##### Decoder

변압기의 디코더는 인코더와 유사한 구조를 가지고 있습니다. 디코더에는 1) Masked
Multi-head Attention, 2) Multi-head Attention, 3) Feed Forward layer의 세 가지
sublayer가 있습니다.<br>

<p>
<center><img src="/images/Transformer/Transformer_decodermask.jpg" width="500"></center>
<center><em>Fig 15. Decoder Mask</em></center>
</p>

1.  Masked Multi-head Attention은 미래 정보에 대한 내용을 가린다는 점을 제외하고
    는 Multi-head Attention과 동일합니다. 변환기가 변환하는 방식은 토큰으로 시작
    하여 토큰이 나타날 때까지 디코더를 반복합니다. time step으로 입력을 받는 RNN
    과 달리 Transformer는 정답 라벨을 한번에 받기 때문에 self-Attention을 할 때
    미래의 정보가 포함될 수 있습니다. 따라서 아직 예측하지 못한 정답 라벨을 마스
    킹할 필요가 있다. 마스킹 방법은 그림 n과 유사하며 패딩 마스크와 동시에 사용
    할 수 있습니다. <br>

    <p>
    <center><img src="/images/Transformer/Transformer_bothmask.jpg" width="250"></center>
    <center><em>Fig 16. Mask</em></center>
    </p>

2.  Multi-head Attention의 key와 value를 encoder에서 가져오는 것을 제외하고 나머지
    Multi-head Attention 및 Feed Forward 계층은 encoder에서의 multi-head attention과 동일하며 각 sublayer도 Residual 및 Normalization layer과 연결됩니다. Encoder와 decoder는 그림 n과 같
    이 연결되고 decoder는 예측한 단어에 대한 token을 output으로 내보냅니다. <br>

    <p>
    <center><img src="/images/Transformer/Transformer_EncDec.jpg" width="500"></center>
    <center><em>Fig 17. Encoder Decoder connection</em></center>
    </p>

#### Why self-attention?

Transformer에서 translate을 위한 각 단어와 연관된 매핑에 self-Attention을 사용하는 이유는 세가지 정도가 언급되어 있습니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_table.jpg" width="600"></center>
<center><em>Fig 18. advantage</em></center>
</p>

1. Complexity가 낮습니다. 그림 n과 같이 self-attention은 sequence 길이가 embedding 차원보다 작을 때 layer당 덜 복잡합니다. 현대 사회에서 대부분의 문장은 embedding dimension의 길이(transformer에서는 512)보다 적은 단어로 구성되어 대부분의 경우 self-attention이 recurrent 및  다른 method보다 complexity가 낮습니다. <br>

2. 병렬 처리할 수 있는 연산의 양입니다. RNN의 경우 network에 sequence로 단어를 입력하고 recurrent 형태로 계산을 하기 때문에 모든 과정을 한 번에 parallel하게 계산할 수 없고, 단어의 길이에 해당하는 sequencial 연산이 필요합니다. 반면에 self-attention layer는 sequence를 한 번에 input으로 입력하기 때문에 output에서 input으로 한 번에 backpropagation를 parallel히게 연산할 수 있습니다. <br>

3. 마지막으로 self-Attention은 long term dependency에 효과적입니다. 논문에서 서로 다른 계층 유형으로 구성된 네트워크에서 두 input 및 output position 간의 maximum path length를 비교합니다. maximum path length가 길어질수록 더 긴 거리에 걸쳐 gradient loss가 발생하기 때문에 dependency를 학습하기가 쉽지 않기 때문입니다. Self-Attention도 sequence length가 너무 길어지면 한 번에 sequence을 학습할 수 없고 RNN과 마찬가지로 일정한 간격으로 sequence를 잘라 학습해야 하지만 그럼에도 불구하고 maximum path lengt는 recurrent보다 짧습니다. <br>

눈문에는 또한 FFN의 hidden layer의 dimension을 2048로 설정한 이유에 대해서 잠깐 언급이 있는데 이에 대해서는 잘 이해하지 못했습니다. (Computational complexity를 맞춰주기 위함이라고만 이해했습니다.) <br>

#### Disadvantage of Transformer

Transformer는 우수한 성능으로 인해 최근 연구에 활발히 사용되고 있지만 단점도 있습니다. 먼저 앞에서 언급했듯이 sequence length가 너무 길면 학습을 용이하게 하기 어렵기 때문에 transformer가 RNN과 비슷하게 sequential하게 연산이 진행됩니다. 그림 n의 마지막 행 self-attention(restricted)에 나타나있습니다. <br>
둘째로, transformer는 JFT-300M과 같이 일반적으로 ImageNet보다 큰 dataset이 필요하기 때문에 vision에서 활용하기 어렵습니다. 몇몇 논문은 transformer에 대해 "Transformers lack some of the inductive biases inherent to CNNs"고 언급하고 있습니다. Machine learning은 기본적으로 train dataset에 대해 원하는 성능의 data를 생성하는 것뿐만 아니라 untrained data에 대해서도 data를 생성하기 위해 구성됩니다. Inductive bias는 training에서 데이터 내부적으로 이해하기 위해 얻어지는 bias라고 할 수 있습니다. 즉, inductive bias는 학습 모델이 이전에 만난 적이 없는 상황에서 정확한 예측을 하기 위해 사용하는 추가적인 assumption을 의미합니다. <br>

<p>
<center><img src="/images/Transformer/Transformer_inductive_bias.jpg" width="600"></center>
<center><em>Fig 19. inductive bias</em></center>
</p>

예를 들어, RNN은 순차적으로 단어가 입력이 되기 때문에 sequenciality inductive bias를 갖습니다. 그래서 가까운 단어에 대해서는 우수한 성능을 보여주지만, 주어를 언급하는 것과 같이 먼 단어에 대해서는 학습이 어렵습니다. <br>
그리고 CNN은 kernel과 convolution을 사용합니다. 따라서 output의 한 point에 영향을 끼치는 범위인 reception fleid가 있고, locality inductive bias를 갖습니다. 따라서 CNN은 local image에서 feature을 추출하는 데 탁월한 성능을 보여줍니다. <br>
이런 면에서 RNN과 CNN은 global region에서 feature을 볼 수 있도록 많은 연구를 진행하고 있습니다.<br>  
Transformer에서는 NLP에서 inductive bias가 없기 때문에 입력에 대해 위치 인코딩을 적용합니다. Vision 분야에서는 image에 대한 많은 정보를 활용하지만 CNN과 같은 locality가 없기 때문에 transformer는 훈련을 위해 많은 dataset가 필요합니다. <br>

### Reference
