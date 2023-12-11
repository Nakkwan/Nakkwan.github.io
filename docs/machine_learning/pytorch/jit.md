---
layout: default
title: JIT
parent: PyTorch
grand_parent: Machine Learning
has_children: False
permalink: /docs/machine_learning/pytorch/jit_2023_07_10
math: katex
---

# JIT
{: .no_toc}


Table of contents
{: .text-delta }
1. TOC
{:toc}

Just-In-Time으로, 프로그램을 실행하는 시점에서 코드를 기계어로 변역함 <br>
기존엔 interpreter 방식과 compiler 방식이 존재함 <br>
Interpreter의 경우 코드를 한줄씩 읽어가며 동적으로 기계어 코드를 생성 및 실행하고, Compiler는 코드를 우선 기계어로 번역 후 프로그램을 생성함 <br>
JIT은 혼합된 방식으로, interpreter와 같이 코드를 한 줄씩 실행하여 기계어로 번역하는데, 기계어 코드를 캐싱하여 같은 함수가 불릴 때 여러 번 기계어로 생성되는 낭비를 줄임 <br>
이외에도 Ahead-of-Time Compiler라는 중간에 다른 언어로 번역 후 기계어로 번역하는 방식도 있음 <br>

## Trace & Script
pytorch는 기존 모델을 load할 때, model architecture에 대한 코드가 필요함 (Eager Mode) <br>
- (Dict 형태로 weight를 저장하기 때문) <br>
TorchScript를 이용하면, script mode로 모델을 저장하여 직렬화하거나 inference에 최적화할 수 있음 <br>
즉, 실행 파일처럼 취급되어, model architecture에 대한 코드 없이 inference를 할 수 있음 <br>
추가적으로, graph와 code attribute도 제공 <br>
sciprt를 생성하는 방식은 trace와 script 방식이 있음 <br>

### Trace
Trace는 input에 따라 코드의 동작 흐름을 추적하여 모델을 기록함 <br>
하지만 input에 대한 흐름을 기록하기 때문에, forward 안에 조건문이 포함된 경우 동작이 안될 수 있음 <br>
또한 마찬가지의 이유로  input이 scalable 하지 않고,  batch의 크기  또한 고정되어야 함 <br>
```python
my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)
print(traced_cell.graph)
print(traced_cell.code)
```

### Script
Trace 방식과 다르게 python 코드를 직접 해석하여 모델을 생성하는 script 방식도 존재 <br>
하지만 python의 상수 값이 포함된 경우 script가 해석하지 못한다는 단점이 있음 <br>
```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.jit.trace produces a ScriptModule's conv1 and conv2
        self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
        self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

    def forward(self, input):
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        return input

scripted_module = torch.jit.script(MyModule()
```
두 방법을 혼합해서 사용할 수도 있음 <br>
```python
class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)
```

### Model Load
```python
traced.save('wrapped_rnn.pt')

loaded = torch.jit.load('wrapped_rnn.pt')

print(loaded)
print(loaded.code)
```