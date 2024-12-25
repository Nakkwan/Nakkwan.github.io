---
layout: default
title: einops
nav_order: "2023.08.10"
parent: Python
grand_parent: Development
permalink: /docs/development/python/einops_2023_08_10
---

# einops
{: .no_toc}
Tensor의 연산을 flexible하고 readable하게 할 수 있는 tool


Table of contents
{: .text-delta }
1. TOC
{:toc}


## Installation
```shell
pip install einops
```
Rearrange, repeat, reduce, stacking, reshape, transposition, squeeze/unsqueeze, tile, concatenate, view 등 기본적인 tensor 연산을 제공하고, Torch나 keras 등 ML framework의 모델에 대한 API도 제공함 <br>
또한 기본적인 tensor 연산에서 dimension의 assert도 제공
```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)', c=256, h=19, w=19)
```
예를 들어, from einops.layers.torch import Rearrange 를 통해, Rearrange 등의 연산을 nn.Module 처럼 사용할 수 있음 (forward 등에서 tensor 연산 작성 불필요)
```python
model = Sequential(
    ...,
    Conv2d(6, 16, kernel_size=5),
    MaxPool2d(kernel_size=2),
    # flattening without need to write forward
    Rearrange('b c h w -> b (c h w)'),  
    Linear(16*5*5, 120), 
    ReLU(),
    Linear(120, 10), 
)
```
추가적으로 tensor들에 대해 packing, unpacking도 가능

```python
from einops import einsum, pack, unpack
# einsum is like ... einsum, generic and flexible dot-product 
# but 1) axes can be multi-lettered  2) pattern goes last 3) works with multiple frameworks
C = einsum(A, B, 'b t1 head c, b t2 head c -> b head t1 t2')

# pack and unpack allow reversibly 'packing' multiple tensors into one.
# Packed tensors may be of different dimensionality:
packed,  ps = pack([class_token_bc, image_tokens_bhwc, text_tokens_btc], 'b * c')
class_emb_bc, image_emb_bhwc, text_emb_btc = unpack(transformer(packed), ps, 'b * c')
# Pack/Unpack are more convenient than concat and split, see tutorial
```

## Syntex
기본적으로 einops는 str로 입력 tensor shape와 output tensor shape를 입력 받음 <br>
각 dimension의 이름을 사용자가 지정하기 때문에, 역할과 동작을 더 명확히 알 수 있음 <br>
```python
y = rearrange(x, 'b c h w -> b h w c')
```

### Rearrange
```python
# (b, w)와 같이 괄호로 묶인 부분은 1개의 차원으로 취급
# results shape: [h, b*w, c]
# 괄호 내 변수의 순서대로 rearrange 되기 때문에 유의
rearrange(ims, 'b h w c -> h (b w) c')
# dim을 늘리고 싶을 때, input 부분에서 늘릴 dim을 괄호로 묶고, 값을 설정
rearrange(ims, '(b1 b2) h w c -> b1 b2 h w c ', b1=2)

# 또한 다음과 같이 squeeze, unsqueeze를 할 수 있음
x = rearrange(ims, 'b h w c -> b 1 h w 1 c')
x = rearrange(ims, 'b 1 h 1 w c -> b h w c')

# reduce를 이용하여 tensor에서 max, min, max 값에 대한 연산도 가능
# 아래의 예시는 밑의 그림과 같이 h, w에 대하여 max값을 계산하여 뺌
# 즉, 이미지 반전 (배경이 검은색이었기 때문에, 글씨 색(max값)에서 이미지를 빼서 이미지 반전
x = reduce(ims, 'b h w c -> b () () c', 'max') - ims
rearrange(x, 'b h w c -> h (b w) c')
```

### Reduce
특정 차원에 대한 reduce도 수행 가능
```python
# reduce시 수행될 연산은 argment로 넣어줌
# mean, min, max, sum 등이 가능
reduce(ims, 'b h w c -> h w c', 'mean')

# 다음과 같이 reduce를 이용하여, mean-pooling, max-pooling 등을 수행 가능
reduce(ims, 'b (h h2) (w w2) c -> h (b w) c', 'mean', h2=2, w2=2)
reduce(ims, 'b (h h2) (w w2) c -> h (b w) c', 'max', h2=2, w2=2)
```

### Repeat
Repeat을 이용하여 numpy의 repeat과 tile을 둘 다 구현가능<br>
numpy에서 repeat은 말그대로 tensor의 크기를 늘리면서 구조는 유지하는 방식<br>
    ex) repeat([1, 2, 3], 3) -> [1, 1, 1, 2, 2, 2, 3, 3, 3] <br>
tile은 tensor의 크기를 늘릴때, tile 처럼 통째로 복사하는 방식<br>
    ex) tile([1, 2, 3], 3) -> [1, 2, 3, 1, 2, 3, 1, 2, 3] <br>
```python
# repeat될 부분을 괄호에서 어디에 놓느냐에 따라 방식이 결정됨
# (repeat w)는 w자체를 반복한다는 느낌 (=np.tile)
repeat(ims[0], 'h w c -> h (repeat w) c', repeat=3)

# (w repeat)는 w의 요소들을 반복한다는 느낌 (=np.repeat)
repeat(ims[0], 'h w c -> h (w repeat) c', repeat=3)
```
```python
# reduce와 repeat은 상대적인 관계
repeated = repeat(ims, 'b h w c -> b h new_axis w c', new_axis=2)
reduced = reduce(repeated, 'b h new_axis w c -> b h w c', 'min')
assert numpy.array_equal(ims, reduced)
```

### Importance of order
앞에서 언급한 것과 같이 괄호로 dim을 묶어줄 때, 순서가 중요함
```python
# batch에 있는 이미지가 정상적으로 재배치됨
# repeat에서와 같이 h, w가 tile과 같이 배치된다는 느낌
rearrange(ims, '(b1 b2) h w c -> (b1 h) (b2 w) c ', b1=2)
```

```python
# batch가 h와 w에 repeat처럼 배치되어, 여러 글자가 겹쳐서 보임
rearrange(ims, '(b1 b2) h w c -> (h b1) (w b2) c ', b1=2)
```

```python
# h부분만 repeat처럼 rearrance되어, 세로축으로만 글자가 겹치게 보임
rearrange(ims, '(b1 b2) h w c -> (h b1) (b2 w) c', b1=2)
```

```python
# w 방향으로 rearrange 되는 것과 같지만, 사라진 b1 dim에 대해 max로 rearrange
# 따라서 h w에서 두 글자가 겹쳐서 보임
reduce(ims, '(b1 b2) h w c -> h (b2 w) c', 'max', b1=2)
```

```python
# 이미지가 downsample (repeat과 같이 2개의 pixel 중 mean값을 남김)
# channel의 경우 h방향으로 rearrange되어 rgb값이 따로 나타남
reduce(ims, 'b (h 2) (w 2) c -> (c h) (b w)', 'mean')
```
즉, <br>
rearrange로 **transpose, reshape, stack, concatenate, squeeze and expand_dims,** <br>
reduce로 **mean, min, max, sum, prod,** <br>
repeat으로 **repeat, tile** <br>
을 수행 가능 <br>

## In ML
einops에서도 일반 pytorch module과 같이 backpropagation을 수행할 수 있음

### Layer
```python
from einops.layers.torch import Rearrange, Reduce

model = Sequential(
    ...,
    Conv2d(6, 16, kernel_size=5),
    MaxPool2d(kernel_size=2),
    # flattening without need to write forward
    Rearrange('b c h w -> b (c h w)'),  
    Linear(16*5*5, 120), 
    ReLU(),
    Linear(120, 10), 
)
```
```python
import torch
x = torch.randn((10, 3, 100, 200), requires_grad=True)
y0 = x
y1 = reduce(y0, 'b c h w -> b c', 'max')
y2 = rearrange(y1, 'b c -> c b')
y3 = reduce(y2, 'c b -> ', 'sum')

y3.backward()
print(reduce(x.grad, 'b c h w -> ', 'sum'))
"""
tensor(320., dtype=torch.float64)
"""
```
einsum을 이용하여, tensor 곱도 가능
```python
result = torch.einsum('tbh,hd->tbd', inputs, weight)
```

### einops.asnumpy
tensor를 numpy로 가져옴 (GPU에서도 가져올 수 있음)
```python
from einops import asnumpy
y3_numpy = asnumpy(y3)

print(type(y3_numpy))
"""
<class 'numpy.ndarray'>
"""
```

### Tensor manipulation
```python
# x.shape: [10, 3, 16, 32]
# Flatten
y = rearrange(x, 'b c h w -> b (c h w)')
""" [10, 288] """

# space-to-depth
y = rearrange(x, 'b c (h h1) (w w1) -> b (h1 w1 c) h w', h1=2, w1=2)
""" [10, 12, 8, 16] """

# depth-to-space
y = rearrange(x, 'b (h1 w1 c) h w -> b c (h h1) (w w1)', h1=2, w1=2)
""" [10, 3, 16, 32] """

# GAP
y = reduce(x, 'b c h w -> b c', reduction='mean')
""" [10, 3] """

# max-pool
y = reduce(x, 'b c (h h1) (w w1) -> b c h w', reduction='max', h1=2, w1=2)
y = reduce(x, 'b c (h 2) (w 2) -> b c h w', reduction='max')
""" [10, 3, 8, 16] """

# keep-dims
y = x - reduce(x, 'b c h w -> b c 1 1', 'mean')
""" [10, 3, 1, 1] """

# channel shuffle
y = rearrange(x, 'b (c1 c2) h w-> b (c2 c1) h w', c1=2, c2=1)
```

### Split
einops를 이용하여, dim split도 가능함 <br>
하지만 order에 따라 동작이 전혀 달라지기 때문에 주의해서 사용해야함 <br>
```python
# numpy에서 보았을 때, y1 = x[:, :x.shape[1] // 2, :, :]와 같음
y1, y2 = rearrange(x, 'b (split c) h w -> split b c h w', split=2)
# numpy에서 보았을 때, y1 = x[:, 0::2, :, :]와 같음
y1, y2 = rearrange(x, 'b (c split) h w -> split b c h w', split=2)
```
따라서 Group Conv나 Multi-head attentio 등의 연산에서 전혀 다르게 동작할 수 있음<br>
또한 shape에 대한 parsing을 제공
```python
from einops import parse_shape

parse_shape(x, 'b c x y')
""" {'b': 10, 'c': 3, 'h': 16, 'w': 32} """
parse_shape(x, 'b c _ _')
""" {'b': 10, 'c': 3} """
```

## Advanced
### Pack & Unpack
einops는 tensor를 concatenation 하거나 다시 split 할 수 있는 function을 제공 <br>
일반적인 concate과 다르게, 어떻게 packing 되었는지에 대한 shape를 제공하여 unpack 가능 <br>
```python
from einops import pack, unpack

h, w = 100, 200
# RGB 이미지와 그에 대한 Depth를 concate해서 RGBD로 만든다고 가정
image_rgb = np.random.random([h, w, 3])
image_depth = np.random.random([h, w])
# 다음과 같이 RGBD로 packing 가능
image_rgbd, ps = pack([image_rgb, image_depth], 'h w *')
print(image_rgb.shape, image_depth.shape, image_rgbd.shape)
""" ((100, 200, 3), (100, 200), (100, 200, 4)) """
# input에 사용된 tensor 중 첫번째 tensor는 h, w 이후에 3의 크기를 가진 dimension을 가짐
# 두번째 tensor의 경우 h, w가 전부임
print(ps)
""" [(3,), ()] """
```
```python
# ps를 이용하여 그대로 unpack 가능
unpacked_rgb, unpacked_depth = unpack(image_rgbd, ps, 'h w *')
print(unpacked_rgb.shape, unpacked_depth.shape)
""" ((100, 200, 3), (100, 200)) """

# ps를 사용하지 않고, manually unpack도 가능
# 아래의 경우 depth에 dimension이 추가됨 (100, 200, 1)
rgb, depth = unpack(image_rgbd, [[3], [1]], 'h w *')
print(rgb.shape, depth.shape)
""" ((100, 200, 3), (100, 200, 1)) """

# 다른 크기로도 unpack 가능
rg, bd = unpack(image_rgbd, [[2], [2]], 'h w *')
print(rgb.shape, bd.shape)
""" ((100, 200, 3), (100, 200, 1)) """

# 빈 array []는 ps에서와 같이 1을 의미하고 dimension expansion을 하지 않음
[r, g, b, d] = unpack(image_rgbd, [[], [], [], []], 'h w *')
print(r.shape, g.shape, b.shape, d.shape)
""" ((100, 200), (100, 200), (100, 200), (100, 200)) """
```

### Example
일반적인 model의 inference 시, multi input에 대해 단일 inference를 위한 packing 후 출력을 unpack 하여 return 가능
```python
# 단일 이미지에 대해서만 classifier 후 unpack
# x의 모양이 달라질 수 있으므로 ps가 필요했음
def universal_predict(x):
    x_packed, ps = pack([x], '* h w c')
    predictions_packed = image_classifier(x_packed)
    [predictions] = unpack(predictions_packed, ps, '* cls')
    return predictions
    
# 1개의 이미지
print(universal_predict(np.zeros([h, w, 3])).shape)
""" (3,) """
# batch 이미지
batch = 5
print(universal_predict(np.zeros([batch, h, w, 3])).shape)
""" (5, 3) """
# batch 및 동영상에 대한 frame
# '* h w c'이기 때문에 35개의 batch로 취급됨
n_frames = 7
print(universal_predict(np.zeros([batch, n_frames, h, w, 3])).shape)
""" (7, 5, 3) """
```
multimodal에서 여러 domain의 tensor를 하나로 packing 할 때도 사용 가능 <br>
(GAN의 fake, real images에도 가능) <br>
추가적으로 object detecting의 경우 output이 복잡하게 얽혀있음 <br>
(처음은 class, 다음 4개는 좌표, 그 다음 n개는 class) <br>
이에 대해 einops  unpack으로 가독성 있게 나눌 수 있음 <br>
```python
def loss_detection_einops(output, coord: int, n_cls: int):
    confidence, xy, class_logits = unpack(output, [[], [coord], [n_cls]], 'b h w *')

    confidence = confidence.sigmoid()

    # downstream computations
    return confidence, class_logits
```
### EinMix
einsum과 비슷한 방식으로 동작 <br>
MLP에서 많이 쓰임
```python
from einops.layers.torch import EinMix as Mix
```

## Reference
[Documents](https://github.com/arogozhnikov/einops) <br>
[In ML](https://github.com/arogozhnikov/einops/blob/master/docs/2-einops-for-deep-learning.ipynb) <br>
[Pack, Unpack](https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb) <br>
[Examples](http://einops.rocks/pytorch-examples.html) <br>