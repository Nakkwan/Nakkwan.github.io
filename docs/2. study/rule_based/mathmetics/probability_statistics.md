---
layout: default
title: Probability & Statistics
nav_order: "2024-06-12"
parent: Mathmetics
grand_parent: Rule-based
permalink: /docs/study/rule_based/mathmetics/probability_statistics_2024_06_12
math: katex
---


# Central Limit Theorem
[그런데 중심극한정리란 무엇인가? (youtube.com)](https://www.youtube.com/watch?v=zeJD6dqJ5lo)

$$
\text{Normal Distribution}: \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
$$

### 3 Assumptions of CLT
1. 모든 변수는 서로 독립적이다
2. 모든 변수는 동일한 분포에서 추출된다
3. 분산은 유한하다

Mean $$\mu=E[X]$$: 해당 분포의 center of mass

Variance $$Var(X)=E[(X-\mu)^2]$$: 평균과 떨어진 정도 (모두 양수) (하지만 제곱이기 때문에 거리로 보긴 힘듦)

StdDev $$\sigma=\sqrt{Var(X)}$$ : 분산보다 거리처럼 취급하기 좋음

- Normal Distribution에서, n-Sigma 만큼 떨어진 값 안에 전체의 $$x\%$$가 포함된다.
    - 1-Sigma: 68.3%
    - 2-Sigma: 95.4%
    - 3-Sigma: 99.7%

변수가 n번 더해지면,

$$
\begin{gather}
  \mu=nE[X] \\
  Var(X)=Var(X_1)+\cdots + Var(X_n)=nE[(X-\mu)^2] \\
  \sigma=\sqrt{n}\sqrt{Var(X)} 
\end{gather}
$$


즉, 분포는 제곱근에 비례하게 커진다. (주사위를 n개 던질 때 숫자의 합의 분포를 생각하면 됨)

### Gaussian Formula

일반적으로 다음 식은 종 모양의 그래프를 띔

$$
y=e^{-cx^2}
$$

상수 $$c$$의 값에 따라, 종 모양의 퍼짐 정도가 달라짐 (증가할수록 좁아짐)

자연 상수 $$e$$가 아니어도 같은 형태를 띔


Normal distribution식을 따라가보면,

확룰 분포가 되기 위해, 분포의 면적이 1이 되어야 한다. 면적을 $$\phi$$라고 할 때,

$$
\begin{gather}
  \phi(e^{-x^2})=\sqrt{\pi} \\
  \phi(\frac{1}{\sqrt{\pi}}e^{-x^2})=1
\end{gather}
$$

$$\sigma$$가 StdDev가 되도록 식을 재구성하면,

$$
\begin{gather}
  \phi(\frac{1}{\sqrt{\pi}}e^{-\frac{1}{2}(x/\sigma)^2})=\sigma\sqrt{2} \\
  \phi(\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(x/\sigma)^2})=1
\end{gather}
$$

로, $$\sigma$$가 StdDev인 확률 분포가 만들어진다.

$$\sigma=1$$인 경우, standard normal distribution $$(y=\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2})$$ 라고 하고,

$$
\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
$$

로 매개변수화하면, normal distribution의 평균과 분산을 규정할 수 있다.

즉, 위의 식은 해당 distribution의 **평균과 분산에 대한 정보**를 담고 있다.

# Normal Distribution
$$
\text{Normal Distribution}: \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
$$

Gaussian distribution에서 계수에 $$\pi$$가 붙는 이유는, bell curve의 면적에 $$\pi$$가 들어가기 때문이다.

확률 분포로 해석하기 위해, 면적을 1로 만들 필요가 있었기 때문이다. 

그럼 gaussian distribution에서 면적은 어떻게 해석되어야 하는가?

왜 bell curve 중에서도 $$y=e^{-x^2}$$이 특별하고 중요한가?