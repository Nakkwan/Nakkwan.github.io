---
layout: default
title: Linear Algebra
nav_order: "2024-06-12"
parent: Mathmetics
grand_parent: Rule-based
permalink: /docs/study/rule_based/mathmetics/linear_algebra_2024_06_12
math: katex
---

# **Linear Algebra Basic**
{: .no_toc}

Table of Contents
{: .text-delta }
1. TOC
{:toc}

### **Inverse Matrix**



### **Determinant**



### **Symmetric matrix**
Square Matrix 중에서 대각원소를 중심으로 원소값들이 대칭되는 행렬을 Symmetric matrix라고 한다.
$$A^T=A$$

Symmetric matrix는 Eigen Decomposition에서 2가지 성질을 가진다.
1. Symmetric matrix는 항상 Eigen Decomposition이 가능하다.
2. Eigen Value는 항상 실수값을 가진다.
3. 직교행렬(orthogonal matrix)로 대각화된다.
	1. 수치적으로 안정적이며, 역행렬이 전치행렬과 같기 때문에 계산이 효율적이다.

### **Orthogonal Matrix**
일반적으로 두 벡터가 직교하면 orthogonal하다고 한다. 그리고 벡터를 unit vector로 만드는 것을 normalization이라고 한다.  따라서, orthonormal은 임의의 두 벡터 $$v_1, v_2$$가 unit vector면서 orthogonal할 때 **orthonormal**이라고 한다.

**행렬**에서 orthogonal matrix는 자기 자신의 transpose를 inverse matrix로 갖는 square matrix를 의미한다. 
$$A^{-1}=A^T,\quad AA^T=E$$
위와 같이, inverse matrix를 transpose로 쉽게 구할 수 있기 때문에 계산에서 용이하다.
또한 orthogonal matrix의 column vector들은 서로 orthonomal한 성질을 가지고 있다. 
-> rank도 n이라고 볼 수 있음


### **Unitary Matrix**
복소수의 square matrix에서 conjugate transpose가 inverse matrix와 같으면 unitary matrix
$$
U^*U=UU^*=I
$$

Unitary Matrix는 복소수 영역에서 orthogoonal matrix와 같다고 생각할 수 있다.
이외의 표현법으로 $$A^H, A^\dagger$$도 쓰인다.


## **고유값 및 고유 벡터**
행렬 $$n\times n$$의 $$A$$에 의해 선형 변환을 수행할 때, 변형 결과가 자기 자신의 상수 배가 되는 벡터를 EigenVector, 상수배 값을 Eigen Value라고 한다 

Square Matrix에서만 정의가 가능하고 다음과 같이 표시된다.
$$
Av=\lambda v
$$
기하학적으로는 변환의 축이라고 볼 수 있다.

[고유벡터와 고유값 (3Blur1Brown)](https://www.youtube.com/watch?v=PFDu9oVAE-g)

## **Positive definite**
임의의 열벡터 $$z$$에 대해 symetric 행렬 $$M$$의 $$z^TMz$$ 값이 양수면 positive definite,
0을 포함한 양수면 positive-semi definite $$M$$이 역행렬이 존재하면, positive definite임

## **Eigen Decomposition**
Eigen Diagonalization은 $$n\times n$$ matrix $$A$$가 $$n$$개의 linearly independent eigenvalue를 가지고 있을 때 가능하다.
예를 들어, matrix $$A$$의 eigenvector를 column으로 가지는 matrix $$P$$가 있을 때, 
$$
A=PDP^{-1}
$$
가 되며 이 때, $$D$$는 eigenvalue를 element로 가지는 diagonal matrix다. 
Diagonalization을 수행하면 다음과 같이, matrix의 exponent를 쉽게 계산할 수 있다.
$$
A^k=PD^kP^{-1}
$$
하지만, Eigen Decomposition은 square matrix만 대각화할 수 있다.  


## **특이값 분해 (SVD)**
특이값 분해는 임의의 행렬과 전치 행렬의 곱에 대한 고유값 분해로, row와 column에 대한 basis와 데이터 변형의 스케일을 나타낸다.
Eigen Decomposition은 square matrix에 대해서만 적용이 가능하지만, SVD는 모든 행렬에 적용이 가능하다.
Matrix $$A$$가 $$m\times n$$ 행렬일 때, SVD는 다음과 같이 정의된다.

$$
\begin{gather}
A=U\Sigma V^T \\
\Sigma: \;m\times n \;\text{(diagonal matrix)} \\
U: \;m\times m \;\text{(orthogonal matrix)} \\
V: \;n\times n \;\text{(orthogonal matrix)} \\
\end{gather}
$$

$$U$$(Left sigular vector) $$AA^T$$를 Eigen Decomposition을 하여 얻은 orthogonal matrix \
$$V$$(Right sigular vector) $$A^TA$$를 Eigen Decomposition을 하여 얻은 orthogonal matrix \
$$\Sigma$$는 $$AA^T$$ 혹은 $$A^TA$$의 eigen value의 root(singular value)를 원소로 하는 diagonal matrix

$$AA^T$$, $$A^TA$$은 모두  symmetric matrix이기 때문에 항상 eigen decomposition이 가능하다.

기하학적으로, orthonormal matrix는 회전변환으로 볼 수 있기 때문에, SVD는
Rotation Matrix -> Scaling -> Rotation Matrix
로 볼 수 있다.

즉, $$U,V$$는 모두 회전 변환이므로, 모양의 변화에 관여하지 않기 때문에 singualr matrix만  scaling 및 모양을 결정한다. 

또한, SVD는 matrix $$A$$를 각 column layer별로 나눠서, 정보량 (sigular vector)에 따라 생각할 수 있게 한다.


## **Truncated SVD**
위와 같은 full SVD가 아닌, singular value가 0에 가까운 부분을 없애는 것을 Truncated SVD라 한다. 
Truncated SVD는 원래 행렬 $$A$$를 완벽히 보존하진 않지만, 제거한 singular value에 따라 근사한 행렬 $$A'$$를 가진다.

영상 처리 분야에서는 Truncated SVD와 같이 차원 축소 혹은 데이터 압축에 사용될 수 있다.
Transformer에도 계산량 및 param 수를 줄이기 위해 attention에서 활용할 수 있다.


## **Pseudo Inverse**
$$Ax=b$$ 와 같은 문제를 풀 때, $$A^{-1}$$이 존재한다면, 문제를 쉽게 풀 수 있지만, 존재하지 않는 경우가 많다. 이런 경우 pseudo inverse를 이용하여 문제를 풀 수 있다.
Pseudo inverse는 $$A^+$$로 표시되며 임의의 $$m \times n$$ 행렬에 대해 정의될 수 있다.

SVD가 아래와 같이 표현될 때, 
$$
A=U\Sigma V^T
$$
Pseudo inverse는 
$$
A^+=V\Sigma^+ U^T
$$
로 표현되며, $$\Sigma^+$$는 sigular value를 역수를 취한 후, transpose하여 계산된다.
-> $$m\times n$$ matrix이기 때문에 transpose를 하면 $$n\times m$$ matrix가 된다.

$$
A=U\begin{pmatrix} \sigma_1 &   \\ & \ddots & \\ && \sigma_s \\ 0&0 & 0  \end{pmatrix}V^T \rightarrow A^+=V\begin{pmatrix} 1/\sigma_1 & & &0  \\ & \ddots & &0\\ && 1/\sigma_s&0 \end{pmatrix}U^T
$$

만약, 모든 singular vector가 양수라면
- $$m\ge n$$일 때, $$A^+A$$는 $$n\times n$$ identity matrix
- $$m\leq n$$일 때, $$A^+A$$는 $$m\times m$$ identity matrix
가 된다.

일반적으로 선형방정식에서 $$m\ge n$$이기 때문에 미지수가 많을 때 pseudo inverse를 적용한다.
-> $$A^+Ax=A^+b$$ -> $$x=A^+b$$

추가적으로 threshold에 대해 truncated SVD를 수행하고, 이에 대해 pseudo inverse를 수행할 수 있다. \
이때의 threshold를 tolerance라고 부른다. \
Threshold는 시스템에서 어느 정도로 noise를 잡을지와 같다. \


## **LU Decomposition**
LU decomposition은 L, U의 곱으로 표현된다.  \
L: diagonal이 모두 1인 Lower triangular matrix \
U: Upper trangular matrix \

**LU Decomposition**
1. L은 대각선이 1이고 나머지가 0인 단위 하삼각행렬, U는 A와 같은 행렬로 설정
2. A에 가우스 소거법을 적용하여, U를 Upper trangular matrix로 만듦
3. 가우스 소거법을 적용하며, 곱셈의 계수로 L matrix를 업데이트 

일반적으로, LU decomposition은 연립방정식을 풀거나, determinant 없이 역행렬을 구할 때, 사용한다.

## **QR Decomposition**
QR decomposition은 Q, R의 곱으로 표현되고 2가지 방법으로 수행할 수 있다. \
Q: orthogonal matrix \
R: upper triangular matrix \

### Gram-Schmidt
1. 주어진 행렬 $$A$$의 열 벡터를 $$a_1,a_2, \dots, a_n$$​으로 정의
2. 직교화
   1. $$a_1$$​을 normalization하여 $$q_1$$ 계산
   2. $$u_i=a_i-\sum^{i-1}_{j=1}\text{proj}_{q_j}a_i$$
      1. $$\text{proj}_{q_j}a_i$$는 $$a_i$$를 $$q_j$$에 projection한 것
   3. $$R$$은 $$Q$$와 $$A$$의 곱으로 정의됨
      1. $$R=Q^TA$$

### Householder transformation
reflection를 사용하여 직교행렬 $$Q$$를 구성하는 방법

1. Householder matrix 구성
	1. 각 column vector에 대해,
	2. $$H-k=I-2\frac{vv^T}{v^Tv}$$
		1. $$v$$는 reflection 하고자 하는 vector
2. $$A$$에 반복적으로 Householder transformation를 적용하여 $$R$$을 계산
3. $$Q$$는 Householder matrix의 곱으로 구성됨
	1. $$Q=H_1H_2\cdots H_k$$

QR decomposition은 Linear system, eigenvalue 계산 등에 활용할 수 있다. \
QR decomposition은 수치적으로 안정적이고, eigenvalue 문제를 해결할 때 유용하다.


## **PCA**
PCA는 데이터 분포에서 주성분을 찾는 방법 \
주성분은 데이터의 분포 특성을 가장 잘 설명할 수 있는 분산이 가장 큰 벡터값을 의미한다.
데이터의 공분산 행렬을 구하고 이에 대한 고유값과 벡터를 구하는 것으로 분석할 수 있습니다. PCA는 결국, 분산이 큰 방향 벡터와 분산의 scale인 고윳값을 구한 것이므로 고윳값이 작은 차원을 없애, 데이터를 압축할 수 있다. 
영상 데이터에선 eigenface와 같이, 영상 데이터의 분포를 이용해 recognition, detection 등을 수행할 수 있다.
1. 분산이 큰 것이 왜 대표적인 특성이 되는가? \
    Eigenface의 경우, pixel dimension vector에 대한 data의 공분산을 구하게 된다. 공분산이 큰 방향 벡터는 각 pixel 값의 데이터 전반에 걸쳐 관계성이 큰 방향을 나타내기 때문에 평균적인 얼굴이라고 생각할 수 있다.


## **Jacobian**
Jacobian은 Multivariable Vector Function의 변화량을 나타내는 matrix다. 즉, 임의의 함수 $$f(x_1, x_2, \cdots, x_N)$$의 국소적인 선형 변환을 의미한다. \
다변수 벡터 함수 $$f:R^n \rightarrow R^m$$ 가 주어졌을 때, $$f(x)$$의 Jacobian 행렬 $$J$$은 다음과 같이 정의됩니다

$$
J = 
\begin{bmatrix}
   \frac{\rho f_1}{\rho x_1} & \cdots & \frac{\rho f_1}{\rho x_n} \\
   \vdots & \ddots & \vdots \\
   \frac{\rho f_m}{\rho x_1} & \cdots & \frac{\rho f_m}{\rho x_n}
\end{bmatrix}
$$

함수에서 입력 근처의 근사 다항식을 구하기 위해 taylor series를 사용하듯이 Multivariable Vector Function에서는 jacobian을 활용하여 $$x_0$$에서의 taylor series의 1차 근사를 할 수 있다. \
$$\rightarrow f(x) \approx f(x_0​)+J(x_0​) \cdot (x−x_0​)$$

이를 통해, 비선형 시스템을 국소적으로 선형화나 초기 조건에 대한 매개변수의 민감도를 측정하는데 활용할 수 있다.
Computer Vision에서는 
1. 이미지 변환(ex: rotation, expand, distortion)에서 픽셀 좌표의 변화
2. 연속된 영상 프레임에서 물체의 움직임을 추적하는 Optical Flow
3. Camera Parameter 및 스테레오 매칭을 통해 카메라의 위치와 자세를 추정

에 활용된다.

Matrix는 basis의 변환이라고 할 수 있다. determinant는 이 변환에서 부피의 scaling factor라고 볼 수 있다. \
따라서, jacobian의 determinant는 국소 공간의 scaling factor로 볼 수 있고, 
1. 선형 변환의 scaling 및 direction 
2. 변수 변환에 따른 probability function의 transform

에 활용된다. 

## **Appexdix**
### 1. Why diagonalization is important
Diagonal matrix는 주대각선을 제외한 모든 성분이 0인 행렬말하며, scaling matrix라고 할 수 있다.
Diagonalization은 행렬과 동적 시스템을 인수 분해하고, 변환을 이해하는 것에 도움을 주기 때문에 중요하다.

#### Diagonalization
Diagonalization은 기본적으로 eigenvalues와 eigenvector를 이용하여 행렬의 동작을 더 단순하게 분석하기 위함이다.

Diagonalization은 $$n\times n$$ matrix $$A$$가 $$n$$개의 linearly independent eigenvalue를 가지고 있을 때 가능하다.
예를 들어, matrix $$A$$의 eigenvector를 column으로 가지는 matrix $$P$$가 있을 때, 
$$
A=PDP^{-1}
$$
가 되며 이 때, $$D$$는 eigenvalue를 element로 가지는 diagonal matrix다. \
-> **Eigen Decomposition**

Diagonalization을 수행하면 다음과 같이, matrix의 exponent를 쉽게 계산할 수 있다.
$$
A^k=PD^kP^{-1}
$$

하지만, Eigen Decomposition은 square matrix만 대각화할 수 있다.  

Square Matrix가 아닌 경우, SVD, QR decomposition 등을 활용할 수 있다. 