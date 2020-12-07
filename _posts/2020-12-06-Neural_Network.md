---
title: Neural Network
tags:
  - Machine Learning
  - Logistic Regression
  - Supervised Learning
  - Neural Network
---

Neural Network는 __특정한 직무나 기능을 수행하는 방법을 모델링 하도록 설계된 것__ 입니다. Neurun(신경)으로 불리는 cell의 상호 연결을 통해 만들어집니다. 
<!--more-->

다음 그림은 간단한 Neural Network입니다.
<img src="https://user-images.githubusercontent.com/48177363/101283812-3721c100-3820-11eb-818c-58ac683fb901.PNG" width="500" height="350"> <br>
Logistic Regression을 여러개 연결한 것 같은 모양으로, Input($$x_{i}$$), Output($$\hat{y}$$), Hidden Layer가 존재합니다. Hidden Layer는 입력, 출력과는 달리 보이지 않는 층으로, 그림에서는 2개의 Hidden Layer가 존재합니다. Layer를 말할 땐 입력층은 제외하기 때문에 그림은 2 Layer NN입니다. <br>
NN에서 기호의 표현은 다음과 같습니다.
- Layer는 윗 첨자의 대괄호로 나타낸다. 
  - ex) i번째 Layer : $$a^{[i]}$$
- 각 Layer의 Unit은 아래첨자로 표시한다. 
  - ex) 2 Layer의 3번째 unit : $$a^{[2]}_{3}$$
- i번째 training data는 윗 첨자의 괄호로 나타낸다. 
  - ex) 2 Layer의 3번째 unit의 9번째 training data : $$a^{[2](9)}_{3}$$<br>
($$i$$는 1부터 시작하지만 입력 층을 $$a^{[0]}$$으로 표현할 수 있습니다.)

각 Layer는 Logistic Regression과 마찬가지로, w, b로 이루어집니다. 1 Layer의 경우, $$W^{[1]} = (4, 3), b^{[1]} = (4, 1)$$차원의 matrix입니다. 1 Layer의 Output인 $$a^{[1]}$$으로 나타나고 2 Layer의 입력으로 들어갑니다. 따라서 각 varible의 matrix는 <br>
- $$W^{[i]}$$ = (i Layer의 Unit 수, i Layer의 Input 수(= i - 1 Layer의 Output 수))
- $$b^{[i]}$$ = (i Layer의 Unit 수, 1)
- $$a^{[i]}$$ = (i Layer의 Unit 수, 1)

Layer에서 계산 과정을 그림으로 보면 다음과 같습니다. <br>
<img src="https://user-images.githubusercontent.com/48177363/101284949-0f355c00-3826-11eb-87aa-07c34d6606ae.PNG" width="550" height="350"><br>

### Represntation

$$Z = W^{T}x + b$$로 나타나는 NN의 1 Layer에서의 식은 Matrix로 보면 다음과 같습니다. <br>
$$Z = \begin{bmatrix}
-- & W^{[1]T}_{1} & --\\ 
-- & W^{[1]T}_{2} & --\\ 
-- & W^{[1]T}_{3} & --\\ 
-- & W^{[1]T}_{4} & --
\end{bmatrix}\begin{bmatrix}
X_{1}\\ 
X_{2}\\ 
X_{3}
\end{bmatrix} + \begin{bmatrix}
b^{[1]}_{1}\\ 
b^{[1]}_{2}\\ 
b^{[1]}_{3}\\ 
b^{[1]}_{4}
\end{bmatrix} = \begin{bmatrix}
W^{[1]T}_{1}X + b^{[1]}_{1}\\ 
W^{[1]T}_{2}X + b^{[1]}_{2}\\ 
W^{[1]T}_{3}X + b^{[1]}_{3}\\ 
W^{[1]T}_{4}X + b^{[1]}_{4}
\end{bmatrix} = \begin{bmatrix}
Z^{[1]}_{1}\\ 
Z^{[1]}_{2}\\ 
Z^{[1]}_{3}\\ 
Z^{[1]}_{4}
\end{bmatrix}$$ <br>
$$a^{[1]} = \sigma(Z^{[1]})$$ <br>
위의 $$a^{[1]}$$은 2 Layer의 입력이 되어 들어갑니다.

이제까지 NN의 간단한 모델에 대해서 알아봤습니다. NN은 Deep한 NN과 Shallow한 NN이 있습니다. 그 기준이 명확한 것은 아니라, 상대적으로 Layer가 깊으면 Deep하다고 말하고, 얕으면 Shallow하다고 말하는 것 같습니다.<br>

### Forward Propagation


