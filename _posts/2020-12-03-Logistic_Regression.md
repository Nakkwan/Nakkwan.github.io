---
title: Logistic Regression
tags:
  - Machine Learning
  - Logistic Regression
  - Supervised Learning
html header: <script type="text/javascript"  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
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

\\( x(t)=\frac{-b\pm \sqrt{{b}^{2}-4ac}}{2a} \\)

Logistic Regression은 위와 같은 입력에 대한 출력을 학습시키는 방식입니다. 출력은 0 ~ 1사이의 값으로 나오고, 그 값은 확률을 의미합니다. y(label)의 값이 0,1만 존재할 때 사용할 수 있습니다.<br>
Logistic Regression에서 쓰는 변수, 용어로는<br>
- Input: x 
- label: y
- weight(가중치): w
- threshold(임계점): b
- Output: y
- z:
- sigmoid = 
