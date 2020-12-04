---
title: Logistic Regression
tags:
  - Machine Learning
  - Logistic Regression
  - Supervised Learning
---

Machine Learning(기계학습)이란 __컴퓨터의 어떤 작업에 프로그래밍하지 않아도 스스로 배우는 능력을 가지는 것__ 을 의미합니다.<br>
<!--more-->

Supervised Learning(지도 학습)은 기계학습의 데이터에 정답(label) 존재하여 입력과 답 사이의 규칙을 컴퓨터가 학습하는 것을 의미합니다. <br>
Unsupervised Learning(비지도 학습)은 지도 학습과 반대로 데이터에 정답(label)이 존재하지 않습니다. 입력과 답 사이의 규칙을 찾는 지도학습과 달리, 
입력의 특성, 특징들을 발견해내는 것이 비지도 학습의 목표라고 할 수 있습니다. 

이번 글에선 지도 학습과 기초적인 학습방식인 Logistic Regression에 대해 간략히 적겠습니다.<br>
우선 지도학습은 Regression과 Classification으로 나눌 수 있습니다. 
Regression은 입력에 대한 연속된 출력으로부터 이산적인 결과를 예측하는 것이고, Classification은 입력을 discrete category로 mapping하는 것을 의미합니다.

학습에 들어가는 입력 또한 Structured Data, Unstructured Data로 나뉩니다. <br>
- Structured Data : 이미 의미가 정의된 Data
  - ex) 가격, 나이, etc.
- Unstructured Data : 의미가 정의되지않은 원시적인 Data
  - ex) text, Image Pixel, Audio, etc.
  
이제 우선적으로 Binary Classification에 대해 알아보겠습니다. Binary Classification은 결과에 따라 discrete하게 분류하는 것을 의미하는데 결과는 0 또는 1로 나타나게 됩니다.<br>
예를 들어, 입력된 Image가 고양이인지 아닌지에 대한 Classification이라면, 

<img src="https://user-images.githubusercontent.com/48177363/100990248-05e19080-3595-11eb-9d52-3ea194c886ef.PNG" width="900" height="300">

위의 그림과 같이 training data(위의 예제에서는 (64*64*3, 1) vector) 하나를 $x^{i}$라 할 때 training data의 개수를 $n_{x}$라 정하면 전체 입력 data는 $X = \begin{bmatrix}
x^{0} & . & . & x^{n_{x} - 1}
\end{bmatrix}$, 출력 $y = \begin{bmatrix}
y^{0} & . & . & y^{n_{x} - 1}
\end{bmatrix}$가 된다.

$$f(x)= if x < x_{min} : (x/x_{min})^a$$  
$$otherwise : 0$$  
$$P(w)=U(x/2)(7/5)/Z$$  
$$p_{\theta}(x) = \int p_{\theta}(2z)p_{\theta}(y\mid k)dz$$  
$$x = argmax_k((x_t-x_u+x_v)^T*x_m)/(||x_b-x_k+x_l||)$$  


Logistic Regression은 위와 같은 입력에 대한 출력을 학습시키는 방식입니다. 출력은 0 ~ 1사이의 값으로 나오고, 그 값은 확률을 의미합니다. y(label)의 값이 0,1만 존재할 때 사용할 수 있습니다.<br>
Logistic Regression에서 쓰는 변수, 용어로는<br>
- Input: $x\in R^{n_{x}} $ 
- label: $y\in 0, 1 $
- weight(가중치): $w\in R^{n_{x}}$
- threshold(임계점): $b\in R$
- Output: $\hat{y} = \sigma (w^{T}x + b)$
- z: $w^{T}x + b$
- sigmoid : $\frac{1}{1+e^{-x}}$
  - 0,1 사이의 확률로 나타내기 위해 쓰임

Logistic Regression은 w와 b를 학습시키는 것 입니다. 이를 위해선 cost function(J)를 최소화하는 방향으로 학습이 진행되어야 합니다. Cost function은 Regression Loss(L)의 평균합으로 나타내어 집니다.
$$J(w,b) = \frac{1}{m}\sum L(\hat{y}^{(i)}, y^{(i)})$$
$$L(\hat{y}^{(i)}, y^{(i)}) = -(y^{{(i)}}log(\hat{y}^{(i)})) - (1 - y^{(i)})log(1-\hat{(i)})$$

Logistic Loss로부터 <br>
- $y^{(i)} = 0$일 때 : $L(\hat{y}^{(i)}, y^{(i)}) = - (1 - y^{(i)})log(1-\hat{(i)})$
- $y^{(i)} = 1$일 때 : $L(\hat{y}^{(i)}, y^{(i)}) = -(y^{{(i)}}log(\hat{y}^{(i)}))$

