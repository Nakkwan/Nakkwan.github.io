---
layout: default
title: Fold and UnFold
parent: PyTorch
grand_parent: Machine Learning
has_children: False
permalink: /docs/machine_learning/pytorch/foldunFold_2023_07_10
math: katex
---

# Fold and UnFold
{: .no_toc}

Table of contents
{: .text-delta }
1. TOC
{:toc}

Fold와 unfold는 tensor를 어떤 kernel_size에 대해서 sliding 형식으로 뽑아내거나 결합할 때 사용 <br>
예를 들어, unfold는 (N, C, S1, S2, ...)의 tensor를 Sn에 대해서 -1 dim으로 펼침 <br>
kernel_size가 -1 dim에 들어가고, stride만큼 sliding되면서 생긴 kernel의 개수가 원래 dim에 남음 <br>
fold는 unfold의 동작을 정확히 반대로 함 <br>
큰 이미지를 작은 사이즈의 이미지들로 나누어, batch로 쌓을 때나, (torch.tensor.unfold & F.fold)
convolution 연산을 matrix로 계산하고 싶을 때 사용 (nn.unfold, nn.fold)  <br>

## Fold
### nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)
nn.fold와 같음

### nn.fold(output_size, kernel_size, dilation=1, padding=0, stride=1)
matrix (sliding local block=sliding 연산을 할수 있는 block(matrix))를 일반 tensor로 다시 돌림
($$N, C, \times\prod(kernel_size), L$$) -> ($$N, C, out_size[0], out_size[1], \cdots $$)

## Unfold
### torch.Tensor.unfold(dimension, size, step)
원래 tensor의 dimension에 대해 size를 -1 dim에 추가하고 step 씩 이동하여 얻은 batch를 dimenstion에 추가함 <br>
즉, size 크기의 window를 step만큼 이동하며 얻은 patch들을 반환하는 것  <br>
- dimension엔 patch의 개수, -1 dim엔 size <br>
큰 이미지를 batch로 만들어 쌓을 때 좋음 <br>
```python
images = torch.randn(1, 3, 256, 256)
# img_shape: [1, 3, 8, 16, 32, 16]
# 2 dim에는 8개 patch의 32 window, 3 dim에는 16개 patch에 16 window
img_ = images.unfold(2, 32, 32).unfold(3, 16, 16)
# 최종 img는 [32, 16] 크기의 이미지 patch를  8x16개 얻음
# img.shape: [8x16, 3, 32, 16]
img = img_.reshape(128, 3, 32, 16)
```

### nn.unfold(kernel_size, dilation=1, padding=0, stride=1)
unfold는 sliding 연산을 하는 batch 형식을 가진 tensor를 matrix (sliding local block=sliding 연산을 할수 있는 block(matrix))로  만듦 <br>
($$N, C, ∗$$)일 때, ($$N, C\times\prod(kernel_size), L$$)이 됨 <br>
$$L$$은 연산의 output의 총 size를 의미함 <br>
Convolution 연산은 w * x 일때, x를 flatten(x_f)하고, w를 unfold 후 output size로 reshape (fold) 한 것과 같음 <br>

$$
\begin{aligned}
    \begin{bmatrix} 
    x_1 & x_2 & x_3 \\ 
    x_4 & x_5 & x_6 \\ 
    x_7 & x_8 & x_9 
    \end{bmatrix} 
    * 
    \begin{bmatrix} 
    w_1 & w_2 \\ 
    w_3 & w_4 
    \end{bmatrix} = 
    \begin{bmatrix} 
    w_1 & w_2 & 0 & w_3 & w_4 & 0 & 0 & 0 & 0 \\  
    0 & w_1 & w_2 & 0 & w_3 & w_4 & 0 & 0 & 0 \\  
    0 & 0 & w_1 & w_2 & 0 & w_3 & w_4 & 0 & 0 \\ 
    0 &  0 & 0 & w_1 & w_2 & 0 & w_3 & w_4 & 0\\   
    \end{bmatrix} \cdot 
    \begin{bmatrix}  x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \\ x_6 \\ x_7 \\ x_8 \\ x_9  
    \end{bmatrix}
\end{aligned}
$$ 
<br>
convolution은 Commutative하기 때문에 반대도 성립하는데, unfold는 input에 대해 동작함 <br>

```python
x * w = unfold(x).(t) @ w.flatten() <br>
```

$$
\begin{bmatrix} x_1 & x_2 & x_3 \\ x_4 & x_5 & x_6 \\ x_7 & x_8 & x_9 \end{bmatrix} * \begin{bmatrix} w_1 & w_2 \\ w_3 & w_4 \end{bmatrix} = \begin{bmatrix}   x_1 & x_2 & x_4 & x_5 \\    x_2 & x_3 & x_5 & x_6 \\   x_4 & x_5 & x_7 & x_8 \\  x_5 & x_6 & x_8 & x_9  \\    \end{bmatrix}^t  \cdot \begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \\   \end{bmatrix}
$$
<br>

```python
inp = torch.randn(1, 3, 3, 3)
w = torch.randn(2, 3, 2, 2)
# inp_unf.shape = torch.Size([1, 3 * 2 * 2, ((3 - 2) + 1) * ((3 - 2) + 1)]) 
# [batch, in_ch * kernel * kernel, out_size]
# [1, 3 * 2 * 2, ((3-2)+1) * ((3-2)+1)]
inp_unf = torch.nn.functional.unfold(inp, (2, 2))
# w.view(w.size(0), -1).t() = [27, 2] (3 * 3 * 3, out_ch)
# out_unf = [1, 2, 4] --> transpose하기 전엔, matmul로 [1, 4, 2]
out_unf = w_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
# out.shape = [1, 2, 2, 2]
# fold의 output shape, kernel_size
out = torch.nn.functional.fold(out_unf, (2, 2), (1, 1))
```

{: .no_toc}