---
title: Pytorch Terms
tags:
  - Machine Learning
  - Code
  - Neural Network
  - Pytorch
---
Code term for Pytorch
<!--more-->

#### Dataset

#### Model

#### Training
- grad <br>
grad가 붙은 함수들은 보통 미분 계산에 사용된다
> zero_grad()
>> Backpropagation을 사용하기 전 변화도를 0으로 만들어주는 함수
>> torch에서 backward시, autograd를 사용하게 되는데, autograd에선 grad를 합쳐주기 때문에 그 전에 gradient를 0으로 만들어 주어야한다.
>> <https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch>

#### Test
