---
layout: default
title: Neural Tangent Kernel
parent: Theorem
permalink: /docs/theorem/ntk_2023_10_12
math: katex
---

# Neural Tangent Kernel & Fourier Features
{: .no_toc}
[Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/pdf/1806.07572.pdf)

Table of contents
{: .text-delta }
1. TOC
{:toc}

## Kernel이란?
### Kernel의 등장 이유
기본적으로 linear regression 같은 방법들은, input x에 대해 linear한 W 연산을 통해, 원하는 output y가 나오도록 훈련하는 것 (x $$\rightarrow$$ mapping $$\rightarrow$$ y) <br>

$$
\begin{gather}
f(W,x)=W^Tx, \mathrm{where\;} y_i \in\mathbb{R},\; x_i \in\mathbb{R^l},\; \{(x_i, y_i)\}^n_{i=1}  \\
\min L(W) = \frac{1}{2}\sum^n_{i=1}(y_i-f(W,x_i))^2 \\
\mathrm{Gradient\;Descent}: W(t+1) = W(t)-\eta_t\nabla L(W_t)
\end{gather}
$$

$$\nabla L(W_t) = \sum(y_i-f(W,x_i))\nabla_wf(W_t,x_i)$$로 나타나는데, 사실상 $$\nabla_wf(W_t,x_i)$$는 $$W$$에 indepedent함 <br>
<br>
linear regression이 간단하고 convex optimization을 풀기에 좋지만, 성능이 좋지 못함 (non-linear한 data에 대해 취약) <br>
따라서, linear의 성질을 유지하면서 non-linearity를 추가하고 싶었음 <br>
$$\rightarrow$$ input space  자체를 high-dimension으로 lifting 시킨 후 linear regression을 적용 <br>
이 때, input space $$\rightarrow$$ feature space를 수행하는 함수를 basis function ($$\Phi$$) 이라고 함

<center><img src="/assets/images/theorem/ntk_fig1.jpg" width="90%" alt="Figure 1"></center>

$$
\begin{gather}
f(W,x)=W^T\Phi(x), \mathrm{where\;} y_i \in\mathbb{R},\; x_i \in\mathbb{R}^l,\; \Phi(x_i)\in\mathbb{R}^k,\;\{(x_i, y_i)\}^n_{i=1},\;l < k  \\
\min L(W) = \frac{1}{2}\sum^n_{i=1}(y_i-f(W,\Phi(x_i)))^2 \\
\mathrm{Gradient\;Descent}: W(t+1) = W(t)-\eta_t\nabla L(W_t)
\end{gather}
$$

$$ ex) \quad x:\begin{bmatrix}x_1\\x_2\\x_3
\end{bmatrix} \rightarrow \Phi(x):\begin{bmatrix}x_1\\x_2\\x_3\\x_1x_2\\x_1x_3\\x_2x_3
\end{bmatrix}
$$

<br>
따라서, 위의 $$f(W,x)$$는 $$W$$나 $$x$$에 대해 linear한가? <br>
$$\rightarrow$$ $$W$$에 대해서는 linear하고, $$x$$에 대해서는 linear하지 않음 <br>
여전히, $$W$$에 대해서는 linear하기 때문에 convex한 loss function을 사용하여, 쉬운 optimization이 가능하면서, $$x$$의 non-linearity로 인해, flexibility를 가짐 <br>

하지만 이 방법은 문제점이 있음 <br>
1. 대부분의 문제에서, basis function을 찾는 것이 매우 어려움
2. basis function에 대한 연산량이 너무 많음 (feature space로 lifting 후, inner matrix까지)
$$\rightarrow$$ 예를 들어, 256 res의 경우 $$10^5$$ dim인데, polynomial 3만 해도 $$10^{15}$$로 dim이 급증  <br>    
$$\rightarrow$$ 여기서 내적까지 하면, 계산량과 메모리에 문제가 있음  <br>

위의 고차원 변환 + 연산량 문제를 해결하기 위해 kernel을 사용함 <br>

$$
\begin{gather}
K(x_i, x_j)=\Phi(x_i)^T\Phi(x_j)=
\left\langle \Phi(x_i)\Phi(x_j) \right\rangle
\end{gather}
$$

### Kernel Trick (method)
일반적으로, ML algorithm에서는 inner-Matrix만 알아도 되는 경우가 많음 <br>
따라서, 위의 basis function에서의 내적에 대한 연산량을 극도로 줄일 수 있는 방법이 kernel trick <br>
(lifting 후 내적이 아니라, input space에서의 연산만으로 basis function과 같은 값을 냄) <br>

$$
\begin{gather}
K(x_i, x_j)\triangleq\Phi(x_i)^T\Phi(x_j)=
\left\langle \Phi(x_i),\Phi(x_j) \right\rangle, \quad K\in\mathbb{R}^{l\times l}
\end{gather}
$$

basis function의 내적과 동치인 값을 얻을 수 있는 함수 $$K$$를 정의 <br>
$$\rightarrow$$ 내적에 대한 값이므로 $$K$$는 당연히, symetric하고, positive semi-definite <br>
kernel function 들은 **Mercer's Theorem**를 만족해야 함 <br>

**자주쓰이는 kernel들**
Linear: $$K(x_i, x_j)=x^T_ix_j$$  <br>
Polynomial: $$K(x_i, x_j)=(x_i^Tx_j+c)^d$$ <br>
Sigmoid: $$K(x_i, x_j)=\tanh\{a(x^T_ix_j)+b\}, \quad a,b \ge0$$ <br>
Gaussian: $$K(x_i, x_j)=\exp\{-\frac{\lVert x_i-x_j \rVert^2_2}{2\sigma^w}\},\quad \sigma \ne 0$$ <br>

Kernel SVM, Kernel regression, Kernel PCA 등에 활용될 수 있음

### Neural Network
이전의 linear regression과 같이, shallow한 1 level NN의 경우 다음과 같이 표기할 수 있음 <br>

$$
\begin{gather}
f(W,x)=\sigma(W^Tx), \mathrm{where\;} y_i \in\mathbb{R},\; x_i \in\mathbb{R^l},\; \{(x_i, y_i)\}^n_{i=1}  \\
\min L(W) = \frac{1}{2}\sum^n_{i=1}(y_i-f(W,x_i))^2 \\
\mathrm{Gradient\;Descent}: W(t+1) = W(t)-\eta_t\nabla L(W_t)
\end{gather}
$$

하지만, linear regression과 다르게 activation function 때문에 $$\nabla_wf(W_t,x_i)$$는 $$W$$에 indepedent하지 않음 <br>
이 부분에서의 차이로부터 NTK의 논문이 시작 <br>

## Neural Tangent Kernel
이전과 같이, 1 level의 Neural Network는 m의 hidden unit을 가질 때, 다음과 같이 표현 가능 <br>

$$
\begin{gather}
f(W,x)=\sigma(W^Tx), \mathrm{where\;} y_i \in\mathbb{R},\; x_i \in\mathbb{R^l},\; \{(x_i, y_i)\}^n_{i=1},\; W\in\mathbb{R}^{l\times m}  \\
\min L(W) = \frac{1}{2}\sum^n_{i=1}(y_i-f(W,x_i))^2 \\
\mathrm{Gradient\;Descent}: W(t+1) = W(t)-\eta_t\nabla L(W_t)
\end{gather}
$$

여기서 저자들은 $$m \rightarrow \infty$$일 때, weight의 값들이 almost static이라는 것을 발견 <br>
$$\rightarrow$$ m이 매우 크면, weight들이 거의 바뀌지 않아도, output이 크게 바뀔 수 있기 때문 <br>
$$\rightarrow$$ Lazy training <br>

따라서, initial weight에서 거의 바뀌지 않으므로, $$W(0)$$에서 1-order taylor approximate를 해도, $$W$$를 잘 approximate할 수 있지 않을까?

$$
\begin{gather}
f(W,x)\simeq f(W_0,x)+\nabla_Wf(W_0,x)^t(W-W_0)
\end{gather}
$$

$$\rightarrow$$ approximate된 함수는 $$W$$에 대해서 linear하고, $$x$$에 대해서는 linear하지 않음 <br>

여기서, 앞의 kernel method를 떠올릴 수 있음 (linaer regression에서 $$f(W,x)=W^T\Phi(x)$$) <br>
그러면, 위의 aprroximate도 $$W$$앞의 값을 basis function (lifting)으로 볼 수 있음 <br>
**Basis function:** $$\nabla_Wf(W_0,x)$$
따라서 kernel은 다음과 같이 표현할 수 있고,

$$
\begin{gather}
K(x_i, x_j)=
\left\langle \Phi(x_i),\Phi(x_j) \right\rangle=\left\langle \nabla_Wf(W_0,x_i),\nabla_Wf(W_0,x_j) \right\rangle
\end{gather}
$$

이를 **Neural Tangent Kernel (NTK)**라고 함 <br>
$$\rightarrow$$ NN에 관한 것이기 때문에 Neural <br>
$$\rightarrow$$ Tangent Space (접공간)이기 때문에 Tangent <br>
$$\rightarrow$$ Kernel처럼 생각되기 때문에 Kernel <br>

## NTK의 활용
<center><img src="/assets/images/theorem/ntk_fig2.jpg" width="70%" alt="Figure 2"></center>
**위의 NN의 NTK**

$$
\begin{gather}    
\begin{cases}
\nabla_{a_i}f_m(W,x)&=\frac{1}{\sqrt{m}}b_i\sigma(a_i^Tx)x\\
\nabla_{b_i}f_m(W,x)&=\frac{1}{\sqrt{m}}\sigma(a_i^Tx)
\end{cases}
\end{gather}
$$

따라서, kernel을 구해서 계산할 수 있음

$$
\begin{gather}
K_m(x,x')=K_m^{(a)}(x,x')+K_m^{(b)}(x,x')
\end{gather}
$$

$$\rightarrow$$ kernel + kernel = kernel (symetric, PSD이기 때문) <br>
계산해보면,

$$
\begin{align}
K_m^{(a)}(x,x')&=\frac{1}{m}\sum^m_{i=1}b_i^2\sigma(a_i^Tx)\sigma(a_i^Tx')xx'\\
K_m^{(b)}(x,x')&=\frac{1}{m}\sum^m_{i=1}\sigma(a_i^Tx)\sigma(a_i^Tx')
\end{align}
$$

여기서, $$a_i,b_i$$는 서로, independent sample이고, $$m\rightarrow\infty$$일 때, kernel 값을 $$\mathbb{E}$$로 볼 수 있음

$$
\begin{align}
K_m^{(a)}(x,x')&=\mathbb{E}[b_i^2\sigma(a_i^Tx)\sigma(a_i^Tx')xx'] \\
K_m^{(b)}(x,x')&=\mathbb{E}[\sigma(a_i^Tx)\sigma(a_i^Tx')]
\end{align}
$$

$$\rightarrow$$ 즉, $$\infty$$의 width를 가진, NN의 behavior를 expectation을 계산함으로써 분석할 수 있음 <br>
추가적으로, input sample이 gaussian과 같이, rotation invariant하다면, <br>

$$
\begin{align}
K_m^{(a)}(x,x')&=\frac{(xx')\mathbb{E}[b^2]}{2\pi}(\pi-\theta(x,x'))\\
K_m^{(b)}(x,x')&=\frac{\lVert x \rVert\lVert x' \rVert\mathbb{E}[\lVert a \rVert^2]}{2\pi d}(\pi-\theta(x,x'))(\cos\theta-\sin\theta)
\end{align}
$$

<br>
**NN을 분석한다는 것은 gradient가 수행되는 과정에서, W와 output y의 동작을 안다는 것** <br>
Gradient Descent의 경우 $$W(t+1) = W(t)-\eta_t\nabla_Wf(W_t,x_i)$$로 생각할 수 있음 <br>
$$\eta \rightarrow 0$$라고 할 때, $$\frac{W(t+1)-W(t)}{\eta_t}=\nabla_Wf(W_t,x_i)$$로 생각될 수 있고 <br>

$$
\begin{align}
\frac{dW(t)}{dt}=-\nabla\hat{y}(W)(\hat{y}(W)-y)
\end{align}
$$

로, W가 어떻게 동작하는지에 대해 알 수 있고, 마찬가지로, y에 대해서

$$
\begin{align}
\frac{\hat{y}(W(t))}{dt}=-\nabla_W\hat{y}(W(t))^T\frac{W(t)}{dt}=\nabla_W\hat{y}(W(t))^T\nabla_W\hat{y}(W)(\hat{y}(W)-y)
\end{align}
$$

로 나타낼 수 있고, 위의 식에서 결국 kernel의 형태를 뽑아낼 수 있음

$$
\begin{align}
\frac{\hat{y}(W(t))}{dt}=\nabla_W\hat{y}(W(t))^T\nabla_W\hat{y}(W)(\hat{y}(W)-y)=-K(W(0))(\hat{y}(W)-y)
\end{align}
$$

<br>

### NN이 low-frequency에 비해, high-frequency를 잘 fitting하지 못하는 이유
Define. $$u=\hat{y}-y$$

$$
\begin{align}
\frac{du}{dt}=-K(W_0)\cdot u
\end{align}
$$

$$\rightarrow$$ ODE solve <br>

$$
\begin{gather}
u(t)=u(0)e^{-K(W_0)t}
\end{gather}
$$

여기서 알 수 있는건, $$t \rightarrow \infty$$일 때, $$\hat{y}\rightarrow y$$ <br>
$$K$$는 Kernel이기 때문에 PSD이기 때문에

$$
\begin{align}
K(W_0)&=\sum^n_{i=1}\lambda_iv_iv_i^T, \quad 0<\lambda_1<\lambda_2<\cdots<\lambda_n \\
u(t)&=u(0)\sum^n_{i=1}e^{-\lambda_iv_iv_i^Tt}
\end{align}
$$

$$u$$라는 것은 error값이 어떻게 바뀌느냐에 대한 것. <br>
위의 식에서 보면, eigenvalue 값이 rate of change로 보여질 수 있음 <br>
**따라서, eigenvalue가 0에 가까우면, t가 아무리 커져도(훈련이 진행되어도) $$u$$가 수렴하지 않음** <br>
**반면, eigenvalue가 크면, 빠르게 수렴** <br>

**eigenvalue는 사실상, fourier coefficient로 볼 수 있고, low-frequency에 대응되는 쪽이, eigenvalue가 큰 부분이고, high-frequency는 eignevalue가 작은 쪽에 대응됨** <br>

**즉, NN은 high-frequency에 대해 fitting을 잘 하지 못함** <br>

따라서, 아래의 fourier feature에서 high-frequency의 값들을 lifting 해줄 수 있는 kernel을 사용하면, fitting이 잘 됨 <br>
<center><img src="/assets/images/theorem/ntk_fig3.jpg" width="70%" alt="Figure 3"></center>
즉, fourier feature의 경우, 고주파수 부분을 입력으로 넣어준다는 것으로 생각할 수 있음
<center><img src="/assets/images/theorem/ntk_fig4.jpg" width="50%" alt="Figure 4"></center>

## Fourier Features


## Reference
1. [Son's Notation Blog](https://sonsnotation.blogspot.com/2020/11/11-1-kernel.html)
2. [PR-374](https://www.youtube.com/watch?v=9Jx2ulS1sAk)
3. [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://proceedings.neurips.cc/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf)