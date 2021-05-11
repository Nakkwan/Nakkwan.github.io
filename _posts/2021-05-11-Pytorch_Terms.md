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

### Dataset
---

### Model
---

### Training
---
##### Loss
Loss는 torch.nn에서 사용가능 <br>
> ex) torch.nn.L1Loss(), torch.nn.MSELoss()
torch.nn.MSELoss()는 Default가 mean이지만 reduction option을 sum으로 설정 가능
##### Optimizer
torch.optim에서 사용가능
> ex) torch.optim.Adam(), torch.optim.SGD(), torch.optim.RMSprop(), etc...
learning rate와 옵션들을 function에 따라서 lr= ... 등으로 설정 가능 <br>
> ex) torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
> 
##### lr_scheduler
각 parameter나 시간에 따라 leraning rate를 조절해준다. Optimizer.step()을 호출한 후 Scheduler.step()을 호출한다. <br>
(반대로하면 첫번째 learning rate를 optimizer가 건너뛰게 된다.)
- 공통
  - last_epoch: default는 -1. 처음 시작 epoch를 설정하는 option
  - verbose: default는 False, True로 설정하면 update될 때 메세지를 출력한다.
- torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
  - 설정된 lr_lambda 함수에 따라 learning rate를 조절해준다. (초기 lr에 곱할 factor를 결정하는 함수)
- torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
  - LambdaLR과 비슷하지만 초기 lr이 아니라 누적곱이라는 점이 다르다.
- torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)
  - step size마다 gamma 비율로 lr을 감소시킨다.
- torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)
  - step size가 아닌 lr을 감소시킬 epoch를 지정
##### grad
grad가 붙은 함수들은 보통 미분 계산에 사용된다
- zero_grad()
  - Backpropagation을 사용하기 전 변화도를 0으로 만들어주는 함수
  - torch에서 backward시, autograd를 사용하게 되는데, autograd에선 grad를 합쳐주기 때문에 그 전에 gradient를 0으로 만들어 주어야한다.
  - <https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch>
- required_grad
  - required_grad = True 라면, 이 Tensor에 대한 미분값을 계산하여 Tensor에 저장
  - required_grad = False 라면, Tensor의 미분값을 계산할 필요가 없다는 것
  - default는 True다. 

### Test
---
