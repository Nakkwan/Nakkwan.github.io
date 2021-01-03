---
title: Building Deep NN
tags:
  - Machine Learning
  - Supervised Learning
  - Neural Network
---

[이전 포스팅](https://nakkwan.github.io/2020/12/06/Neural_Network.html)에선 Neural Network에 기본적인 부분에 대해서 포스팅을 했었습니다. 
<!--more-->

### Forward & Backward propagation

간략하게 NN에 대해 살펴보면, forward를 통해, 출력을 얻고, 출력으로부터 backward를 통해, weight를 업데이트합니다.

##### Forward Propagation

- Input: $$a^{[l-1]}$$ 
- Output: $$a^{[l]}, cache W^{[l]}, b^{[l]}$$<br>
$$    Z^{[l]} = W^{[l]}a^{[l]} + b^{[l]}$$<br>
$$    a^{[l]} = g^{[l]}(Z^{[l]})$$<br>


##### Backward Propagation

- Input: $$da^{[l]}$$ 
- Output: $$da^{[l-1]}, cache dW^{[l]}, db^{[l]}$$
$$    dz^{[l]} = da^{[l]} * {g^{[l]}}'(z^{[l]})$$<br>
$$    dW^{[l]} = dz^{[l]}a^{[l-1}$$<br>
$$    db^{[l]} = dz^{[l]}$$<br>
$$    da^{[l-1]} = W^{[l]T}dz^{[l]}$$<br>
$$    dz^{[l]} = W^{[l+1]T}dz^{[l+1]} * {g^{[l]}}'(z^{[l]})$$<br>

위의 식들을 그림으로 나타내면 다음과 같습니다. <br>
<img src="https://user-images.githubusercontent.com/48177363/103478885-104daf00-4e0d-11eb-9865-94d72253f5cc.PNG" width="700" height="450">
