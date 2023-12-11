---
layout: default
title: DP, DDP
parent: Machine Learning
permalink: /docs/machine_learning/DPDDP_2023_07_20
---

# DataParallel and DataDistributed
{: .no_toc}

Table of contents
{: .text-delta }
1. TOC
{:toc}

## DataParallel
Multi Thread로 동작함
1. 배치를 32씩 4개의 GPU에 할당
2. resnet101 모델 전체를 각 GPU에 복사 (replicate)
3. 각 GPU에서 32개의 배치를 Forward
4. 나온 output을 기준 GPU3에 모음 (gather)
5. GPU3에서 각 GPU에서 나온 output을 이용하여 각 batch 32개에 대한 loss를 계산
6. loss를 다시 각 GPU에 scatter (scatter)
7. 각 GPU가 gradient를 계산,
8. GPU3에 모아서 모델 업데이트 (reduce, update)
![Figure 1](/assets/images/ml/DPDDP_fig1.jpg) <br>
편하지만, 느리고, 메모리를 많이 잡아먹어서 터질 수 있음
```python
# 아래와 같이 모델에 Dataparallel만 추가해주면 됨
# 하지만 저장할 때, Dataparallel로 저장하게 되면, module이 앞에 붙어서 저장되므로, 저장할 땐
# torch.save(model.module.state_dict(), PATH)
model = torch.nn.DataParallel(model).cuda()
```

## DataDistributed
Multi Process로 동작함 <br>
1. 기존 DataParallel은 하나의 GPU로 gradient를 reduce한 다음, step 후 모든 GPU에 scatter 해줌 (모델 복사) <br>
- 하지만 위의 방식은 너무 overhead가 심함 <br>
2. 따라서, 각  GPU에 모두 reduce를 한다면, 모델을 복사할 필요가 없음 <br>
- 하지만 마찬가지로, ALL-reduce는 비싼 연산임. (n일 때, n^2 만큼의 reduce가 필요) <br>
3. 따라서 Ring-All-Reduce라는 방식을 사용 <br>
![Figure 2: Ring-All-Reduce](/assets/images/ml/DPDDP_fig2.gif) <br>
- node : 컴퓨터의 개수. (GPU 개수가 아님)
- world_size : 여러 컴퓨터에 같은 GPU 갯수가 달려 있다고 가정 할 때, (각 컴퓨터에 달린 GPU 개수) * (컴퓨터 개수). 즉, 모델을 훈련하는 데에 필요한 총 GPU 개수
- nproc_per_node : torch.cuda.device_count()와 반드시 일치. os.environ[“CUDA_VISIBLE_DEVICES”]에 딸린 숫자 개수와 일치해야 함
![Figure 3](/assets/images/ml/DPDDP_fig3.jpg)

## FULLY SHARDED DATA PARALLEL(FSDP)
- DDP에서 모델의 가중치와 옵티마이저는 각 rank에 복제되어 있음 <br>
- FSDP는 rank에 걸쳐, model parameter, optimizer, gradient를 쪼갠(shard) 일종의 data parallel <br>
- 내부 최적화가 되어있기 때문에, DDP 보다 모든 worker에서 메모리 usage가 적음 <br>

![Figure 4](/assets/images/ml/DPDDP_fig4.png)
### In Constructor
shard model (쪼개진 모델)과 각 rank에선 오직 각자의 것만 유지

### In forward path
1. all_gather를 실행하여 모든  rank에서 모든 shard를 수집, FSDP 단위의 전체 매개변수를 복구
2. Forward 계산 수행
3. gather한 shard를 폐기

### In backward path
1. all_gather를 실행하여 모든  rank에서 모든 shard를 수집, FSDP 단위의 전체 매개변수를 복구
2. Backward 계산 수행
3. Gradient Sync를 위해 reduce_scatter 수행
4. Parameter 폐기

## Reference
1. [https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)