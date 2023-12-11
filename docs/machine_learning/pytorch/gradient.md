---
layout: default
title: Gradient
parent: PyTorch
grand_parent: Machine Learning
has_children: False
permalink: /docs/machine_learning/pytorch/gradient_2023_07_10
math: katex
---

# Gradient
{: .no_toc}

[Documents](https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html)
{: .fs-5 .fw-400 }

Table of contents
{: .text-delta }
1. TOC
{:toc}

- Pytorch는 NN 학습을 위한 자동 미분을 지원
- Tensor에서 실행된 연산들을 Function (example) 객체 형식으로 DAG(Directed Acyclic Graph)를 이용하여 저장함
- DAG를 forward에서 tensor.grad_fn에 저장하고, 이를 통해 backward에서 grad 계산과 propagate 수행

## Forward
1. 요청된 연산에 대한 output 계산
2. DAG에 gradient function을 유지 
  - Function 객체는 forward 결과와 output_grad에 대한 미분 연산을 저장

## Backward
1. 각 .grad_fn 으로부터 변화도를 계산
2. 각 텐서의 .grad 속성에 계산 결과 accumulate
3. Chain Rule을 사용하여, 모든 leaf 텐서들까지 propagate
- **.grad_fn**의 경우 requires_grad=True일 때만 저장
- 입력 tensor중 하나라도 True라면, output tensor로 True가 됨
- Backward가 수행될 tensor는 scalar 값을 가져야 함
  
- 이후 optimizer의 step에서 계산 및 저장된 gradient에 대한 실질적인 parameter 조정을 수행
  
- Pytorch의 DAG는 Dynamic 형식으로 생성됨
- 즉, 모델을 정의할 때 생성되는 것이 아니라, 매 iter 마다 새로 생성되기 때문에 forward 과정에서 control flow를 사용할 수 있음
  
- Tensorflow의 경우 모델 정의 시에 생성되기 때문에 static 하고, control flow가 불가능
  - 2.0 부턴 dynamic도 추가됨

## Example
```python
# 연산에 대해 input tensors(input 및 weight)는 leaf, output tensor는 root
# grad는 leaf에 대해서만 연산되고, grad_fn은 root에 생성됨
# 따라서 weight에 대한 grad는 input과 수행한 grad_fn에 의해 연산됨
# grad_fn에는 입력에 대한 forward의 연산을 수행
# setup_context에서 save_for_backward로 backward시
# 연산에 필요한 tensor(inputs)를 ctx에저장
# backward에서는 needs_input_grad로 불러오고, 
# 이를 이용하여 grad_output로 들어온 input에 대해 gradient 연산 수행
import torch

inp = torch.Tensor([1, 2])
w_1 = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
w_2 = torch.tensor([[2., 3.], [4., 2.]], requires_grad=True)

print(inp, w_1, w_2)
"""
tensor([1., 2.])
tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
tensor([[2., 3.],
        [4., 2.]], requires_grad=True)
"""
y_1 = w_1 @ inp
y_2 = w_2 @ y_1
output = y_2.sum()

print(y_1, y_2, output)
"""
tensor([ 5., 11.], grad_fn=<MvBackward0>)
tensor([ 43., 42.], grad_fn=<MvBackward0>)
tensor(85., grad_fn=<SumBackward0>)
"""
# 아직 backward가 수행되기 전이기 때문에 grad_fn에 대한 grad accumulate가 되지 않음
print(w_1.grad, w_2.grad)
"""
None, None
"""
# scalar 값을 가진 output이 아닌, y_2에서 backward를 수행하면
# RuntimeError: grad can be implicitly created only for scalar outputs
output.backward()
print(w_1.grad, w_2.grad)
"""
tensor([[ 6., 12.],
        [ 5., 10.]])
tensor([[ 5., 11.],
        [ 5., 11.]])
"""
```