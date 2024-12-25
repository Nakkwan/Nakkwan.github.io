---
layout: default
title: 3D Vision a& Machine Perception
nav_order: "2023.12.14"
parent: Et cetera
permalink: /docs/etc/3d_vision
---

# Etc
{: .no_toc}
Summary of 3D Vision a& Machine Perception lectures

Table of Contents
{: .text-delta }
1. TOC
{:toc}

## **Image Processing**

### **lens effects**
Lens flare: 매우 밝은 광원에 대한 빛의 흩어진 상호 반사

Chromatic aberration: wavelength에 따른 다양한 굴절률

Vignetting: 공간에서 균일하지 않은 밝기의 광원

Spherical aberration: 구면 렌즈에 의한 초점 안맞음

Radial distortion: 불완전한 렌즈로 인한 편차 발생

### **Filter**
**Sobel Filter** <br>
Vertical과 horizontal로 이뤄짐. <br>
gaussian smooth와 비슷하게, [1,2,1]과 미분에 해당하는 [-1, 0, 1]의 행렬 곱으로 구성됨

### **Detecting corner**

## **Corner**
### **Harris corner detector**
픽셀값의 차이가 커 영상의 특징이라고 할 수 있는 튀어나온 부분등은 의미있는 특징점으로 사용될 수 있음 <br>
$$\rightarrow$$ Scale에 invariant하지 않은 단점이 있음

1. Compute image gradients over small region $$\rightarrow$$ ex. Sobel <br>
2. Subtract mean from each image gradient $$\rightarrow$$ ‘DC’ offset is removed <br>
3. Compute the covariance matrix <br>
    <table>
    <tr><td width="40%">
    $$\begin{gather}\begin{bmatrix}
    \sum_{p\in P}I_xI_x  & \sum_{p\in P}I_xI_y \\
    \sum_{p\in P}I_yI_x  & \sum_{p\in P}I_yI_y
    \end{bmatrix}\end{gather}$$
    </td><td>
    <center><img src="/assets/images/etc/3dvision_fig1.jpg" width="100%" alt="Figure 1"></center>
    </td></tr>
    </table>
    
4. Compute eigenvectors and eigenvalues <br>
    Eigenvector의 경우 변화의 방향을 나타냄

5. Use threshold on eigenvalues to detect corners <br>
    Eigenvalue가 크면, eigenvector로의 변화가 크다는 것을 의미
    

Scale에 대한 단점을 해결하기 위해, Laplacian filter를 사용할 수 있음 <br>
$$\rightarrow$$ Filter와 신호가 모양이 비슷할 때, 가장 큰 값을 출력함 <br>
$$\rightarrow$$ 따라서, Laplacian filter의 sigma를 다른 level로 곱해가며, 출력이 높은 신호를 찾음

<center><img src="/assets/images/etc/3dvision_fig2.jpg" width="95%" alt="Figure 2"></center>

$$\rightarrow$$ 해당 신호의 scale이 corner의 scale과 같다고 볼 수 있음

<center><img src="/assets/images/etc/3dvision_fig3.jpg" width="95%" alt="Figure 3"></center>

## **Feature Descriptors**

이미지 내의 특정 지점이나 객체를 수학적으로 표현하는 방법 <br>
디스크립터는 이미지의 특징점(Feature Point) 주변의 정보를 요약하여, 그 지점이나 객체를 고유하게 식별할 수 있는 벡터로 변환

- **고유성(Uniqueness)**: 각 특징점 주변의 정보를 고유한 방식으로 표현해야 함 <br>
- **강인성(Robustness)**: 조명, 회전, 스케일 변화 등에 강인해야 함 <br>
- **효율성(Efficiency)**: 계산과 저장 공간 측면에서 효율적이어야 함

1. **SIFT (Scale-Invariant Feature Transform)**
    1. **스케일 공간 극단값 검출 (Scale-Space Extrema Detection)**
        - 이미지를 다양한 scale과 blur한 이미지들을 얻음. ex) 4개 scale에 대해 5개의 blur <br>
        - 이후 각 이웃 octav를 빼가는 DoG를 통해 extrima 얻음
    2. **키포인트 정제 (Keypoint Localization)**
        - 극단값 중에서 노이즈나 에지에 의한 반응을 제거하여 실제 특징점을 정제 <br>
        - 테일러 급수 확장을 사용하여 특징점의 위치, 스케일, 비율을 정밀하게 조정
    3. **방향 할당 (Orientation Assignment)**
        - 각 키포인트에 대해 주변 영역의 그라디언트 방향과 크기를 계산 <br>
        - 이 정보를 사용하여 각 키포인트에 하나 이상의 방향 할당 $$\rightarrow$$ 특징점이 회전에 불변+
    4. **키포인트 디스크립터 생성 (Keypoint Descriptor)**
        - 각 키포인트 주변의 그라디언트 정보를 사용하여 특징점 기술자를 생성 <br>
            - 일반적으로 128차원의 벡터로 표현(16cell x 8 direc) <br>
        - 특징점 주변의 지역적 그라디언트 패턴을 요약 <br>
    - 스케일 불변성을 가지며, 회전에도 강인한 특징점을 제공합니다. <br>
    - 각 특징점 주변의 그라디언트 방향과 크기를 기반으로 128차원 벡터를 생성합니다.
2. **SURF (Speeded Up Robust Features)**
    - SIFT보다 계산이 빠르면서 비슷한 성능을 제공 <br>
    - 박스 필터를 사용하여 빠르게 특징점 주변의 정보를 요약
3. **ORB (Oriented FAST and Rotated BRIEF)**
    - 회전에 강인하고, 계산이 매우 빠릅니다. <br>
    - FAST 알고리즘으로 특징점을 검출하고, BRIEF 디스크립터를 회전 불변성이 있도록 개선합니다.
4. **HOG (Histogram of Oriented Gradients)**
    1. 8x8의 cell에 대해 gradient historgram 계산 <br>
        - gradient는 각도와 magnitude에 대한 값이 있음
    2. 2x2의 cell block 단위로 histogram 정규화
    3. 각 정규화된 block의 histogram이 Descriptor가 됨
    - 객체의 형태와 외관을 기술하는 데 사용됩니다. <br>
    - 이미지의 지역적 그라디언트 방향의 분포를 히스토그램으로 나타냅니다.
5. Gabor filter
    1. Gaussian (공간적인 정보를 나타냄)과 Cos (주기성) 의 곱
    - 특정 방향성과 주기성을 가진 패턴을 감지할 수 있음
6. GIST
    1. Gabor filter의 bank로부터 이미지 계산
    2. 4 x 4로 image patch를 나눔 (cell)
    3. 각 cell의 filter response를 계산
    4. 4 x 4 x N Size의 descriptor

## **2D transformation**
### **Translation, Rotation, Aspect, Affine, Perspective, Cylindrical**

<center><img src="/assets/images/etc/3dvision_fig4.jpg" width="95%" alt="Figure 4"></center>

<table>
<tr><td><center>Scale</center></td><td><center>Rotate</center></td><td><center>Shear</center></td></tr>
<tr><td>
$$\begin{bmatrix}
s_x  & 0 \\
0  & s_y
\end{bmatrix}$$
</td><td>
$$\begin{bmatrix}
\cos\theta  & -\sin\theta \\
\sin\theta  & \cos\theta
\end{bmatrix}$$
</td><td>
$$\begin{bmatrix}
1  & s_x \\
s_y  & 1
\end{bmatrix}$$
</td></tr>
<tr><td><center>Flip across y</center></td><td><center>Flip across origin</center></td><td><center>Identity</center></td></tr>
<tr><td>
$$\begin{bmatrix}
-1  & 0 \\
0  & 1
\end{bmatrix}$$
</td><td>
$$\begin{bmatrix}
-1  & 0 \\
0  & -1
\end{bmatrix}$$
</td><td>
$$\begin{bmatrix}
1  & 0 \\
0  & 1
\end{bmatrix}$$
</td></tr>
</table>


### **Homogeneous coordinate**
n차원 사영 공간을 n+1개의 좌표로 나타내는 좌표계

#### **Transformations in projective geometry**
<table>
<tr><td><center>Translation</center></td><td><center>Scaling</center></td></tr>
<tr><td>
$$\begin{bmatrix}
x' \\ y' \\ 1
\end{bmatrix} = \begin{bmatrix}
1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}$$
</td><td>
$$\begin{bmatrix}
x' \\ y' \\ 1
\end{bmatrix} = \begin{bmatrix}
s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}$$
</td></tr>
<tr><td><center>Rotation</center></td><td><center>Shearing</center></td></tr>
<tr><td>
$$\begin{bmatrix}
x' \\ y' \\ 1
\end{bmatrix} = \begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}$$
</td><td>
$$\begin{bmatrix}
x' \\ y' \\ 1
\end{bmatrix} = \begin{bmatrix}
1 & 0 \beta_ 0 \\ \beta_y & 1 & 0 \\ 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}$$
</td></tr>
</table>

<table>
<tr><td>
$$\begin{bmatrix}
x' \\ y' \\ w'
\end{bmatrix} = \left(\begin{bmatrix}
1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
\cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1
\end{bmatrix}\right)\begin{bmatrix}
x \\ y \\ w
\end{bmatrix}$$
</td></tr>
<tr><td><center>$$p' = \mathrm{translation}(t_x,t_y) \cdot \mathrm{rotation}(\theta) \cdot \mathrm{scale}(s,s) \cdot p$$</center></td></tr>
</table>

#### **Determining unknown (affine) 2D transformation**

$$
\begin{gather}
    E_{LS} = \sum_i \parallel f(x_i;P)-x'_i \parallel^2
\end{gather}
$$

$$\rightarrow$$ $$x_i$$에 $$f$$ 변환을 적용한 것과 projection된 $$x'_i$$사이의 error <br>
$$\rightarrow$$ transform function이 linear일 때만 가능

#### **The direct linear transform (DLT)**

$$
\begin{gather}
P' = H \cdot P \quad\mathrm{or}\quad \left[\begin{matrix}
    x' \\ y' \\ 1
\end{matrix}\right] = \alpha\left[\begin{matrix}
    h_1 & h_2 & h_3 \\ h_4 & h_5 & h_6 \\ h_7 & h_8 & h_9 
\end{matrix}\right]
\end{gather}
$$

$$
\begin{gather}
    x'(h_7x+h_8y+h_9) = (h_1x+h_2y+h_3) \\
    x'(h_7x+h_8y+h_9) = (h_4x+h_5y+h_6) \\
    h_7xx'+h_8yx'+h_9x'-h_1x-h_2y-h_3 = 0 \\
    h_7xy'+h_8yy'+h_9y'-h_4x-h_5y-h_6 = 0
\end{gather}
$$

$$
\begin{gather}
    A_ih = 0 \\
    A_i = \left[\begin{matrix}
        -x & -y & -1 & 0 & 0 & 0 & xx' & yx' & x' \\
        0 & 0 & 0 & -x & -y & -1 & xy' & yy' & y'
    \end{matrix}\right] \\
    h = \left[\begin{matrix}
        h_1 & h_2 & h_3 & h_4 & h_5 & h_6 & h_7 & h_8 & h_9
    \end{matrix}\right]^T
\end{gather}
$$

1. For each correspondence, create 2x9 matrix $$𝐴_𝑖$$ <br>
2. Concatenate into single 2n x 9 matrix $$𝐴$$ <br>
3. Compute SVD of $$𝐴 = 𝑈Σ𝑉^𝑇$$ <br>
4. Store singular vector of the smallest singular value $$h =v_{\hat{i}}$$ <br>
5. Reshape to get $$H$$

### **Random Sample Consensus (RANSAC)**
임의의 feature 쌍들을 sampling하여 DLT <br>
inlier가 가장 많은 것을 고름

## **Camera Model**
3D world의 point를 camera matrix를 거쳐, 2D image point로 transformation

<table>
<tr><center>x = PX</center></tr>
<tr><center>
$$
\left[\begin{matrix}
    x \\ y\\ w
\end{matrix}\right] = \left[\begin{matrix}
    p_1 & p_2 & p_3 & p_4 \\
    p_5 & p_6 & p_7 & p_8 \\
    p_9 & p_{10} & p_{11} & p_{12} 
\end{matrix}\right]\left[\begin{matrix}
    X \\ Y \\ Z \\ 1
\end{matrix}\right]
$$
</center></tr>
<tr><td><center>Homogeneous img coord (3x1)</center></td>
<td><center>Camera Matrix (3x4)</center></td>
<td><center>Homogeneous world coord (4x1)</center></td></tr>
<tr><center>
$$
P = \left[\begin{matrix}
    f & 0 & p_x \\
    0 & f & p_y \\
    0 & 0 & 1
\end{matrix}\right]\left[\begin{matrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 
\end{matrix}\right]
$$
</center></tr>
</table>
$$\rightarrow$$ camera coordinate system에서 image coordinate system으로 transform하는 $$K$$(2D $$\rightarrow$$ 2D), <br>
$$\rightarrow$$ 3D to 2D에 관한 perspective projection로 구성됨 (위 수식에선 $$z=1$$)

### **World-to-camera transformation**
$$\tilde{X}_c=R\cdot(\tilde{X}_w-\tilde{C})$$ <br>
$$\rightarrow$$ Rotation, Translation으로 구성
최종적으로, <br>
<table>
<tr><center>
$$
P = \left[\begin{matrix}
    f & 0 & p_x \\
    0 & f & p_y \\
    0 & 0 & 1
\end{matrix}\right]\left[\begin{matrix}
    I & \mid & 0
\end{matrix}\right]\left[\begin{matrix}
    R & -R\tilde{C} \\
    0 & 1
\end{matrix}\right]
$$
</center></tr>
<tr><td><center>intrinsic param</center></td>
<td><center>perspective proj</center></td>
<td><center>extrinsic param</center></td></tr>
</table>

Intrinsic param과 Extrinsic Param을 나눠보면,

$$
P = K[R\mid t], \qquad P = \left[\begin{matrix}
    f & 0 & p_x \\
    0 & f & p_y \\
    0 & 0 & 1
\end{matrix}\right] \left[\begin{matrix}
    r_1 & r_2 & r_3 & t_1 \\
    r_4 & r_5 & r_6 & t_2 \\
    r_7 & r_8 & r_9 & t_3 
\end{matrix}\right]
$$

## **Epipolar Geometry**
Epipolar geometry는 동일한 장면에 대한 영상을 서로 다른 두 지점에서 획득했을 때, 영상 A와 영상 B의 feature들 사이의 기하학적 관계에 관한 것 <br>
$$\rightarrow$$ 공간상의 위치 P가 영상 A에 투영된 점과 두 카메라의 위치 관계를 알고 있을 때, 영상 B에서의 투영점도 알 수 있음 <br>
$$\rightarrow$$ 반대로 두 영상에서의 매칭쌍을 알고 있을 때, 두 카메라 위치 관계에 대해서 알 수 있음

Epipole: 두 카메라의 원점을 잇는 선과 이미지가 만나는 점들 <br>
Epiline: 투영점과 epipole을 잇는 이미지 평면 위의 선

각 투영된 두 점을 $$p=[u,v,1], p'=[u',v',1]$$이라고 할 때, $$p'Ep=0$$ <br>
Essential Matrix: 정규화된 이미지 평면에서의 매칭 쌍들 사이의 기하학적 관계를 나타낸 행렬 <br>
Fundamental Matrix: 카메라 파라미터까지 포함한 두 이미지의 실제 픽셀(pixel) 좌표 사이의 기하학적 관계를 표현하는 행렬

### **Epipolar constraint**
5쌍의 매칭점으로부터 Essential, Fundamental  Matrix를 얻을 수 있음 <br>
$$E$$는 $$R, t$$로 구성되는데 회전변환 $$R$$이 3 자유도, 스케일을 무시한 평행이동 $$t$$가 2 자유도, 도합 5 자유도이므로 5쌍의 매칭점을 필요로 함

### **Triangulation**
두 이미지 평면 사이의 기하학적 관계 ($$E$$ or $$F$$)가 주어지고 평면상의 매칭쌍 $$p, p'$$이 주어지면 이로부터 원래의 3D 공간좌표 P를 결정할 수 있음

## **Streo**
### **Disparity**
Disparity: streo camera에서 같은 지점에 대한 픽셀 차를 의미 <br>
$$\rightarrow$$ 두 카메라의 위치가 다르기 때문에 같은 물체를 볼 때, 영상에서 픽셀의 위치 차이가 발생 <br>
$$\rightarrow$$ 가까운 물체는 위치 차이가 크고, 먼 물체는 차이가 적은데, 이 거리 이미지가 disparity map <br>
- Disparity와 camera parameter를 이용하여 물체까지의 거리를 측정할 수 있는데, 이미지에서 거리를 표현한 것이 depth map

### **Stereo rectification**
카메라 센터와 평행하도록 두 이미지 평면을 reprojection <br>
$$\rightarrow$$ 각 이미지 평면의 reprojection을 위해 두 개의 homography가 필요

1. Compute $$E$$ to get $$R$$ <br>
2. Rotate right image by $$R$$ <br>
3. Rotate both images by $$R_{rect}$$ <br>
4. Scale both images by $$H$$

### **Stereo matching**
Epipolar line에서 가장 matching이 높은 매칭쌍을 선택 <br>
각 매칭쌍의 점을 알고 있어야 함

### **Improving stereo matching**
### **Depth estimation**

<center><img src="/assets/images/etc/3dvision_fig5.jpg" width="80%" alt="Figure 5"></center>

$$d=x-x'=\frac{bf}{Z}$$

## **SFM (Structure from Motion)**

<!-- <center><img src="/assets/images/etc/3dvision_fig6.jpg" width="80%" alt="Figure 6"></center> -->
