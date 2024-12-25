---
layout: default
title: Light Modeling
nav_order: "2024-06-29"
parent: Blender
grand_parent: Development
permalink: /docs/development/blender/light_modeling_2024_06_29
---

## Light
### Options
#### Light
* Color: 조명의 색
* Power: 조명의 세기
	* 50W: 일반적인 밝은 전구
	* 100W: 스튜디오 전구
* Diffuse: 빛이 물체에 닿았을 때 물체 표면에 얼마나 고르게 분포되는지 (default: 1.0)
* Specular: 물체 표면에서 빛이 거울처럼 반사되는 것 (default: 1.0)
* Volume: 3D 공간 상에서 빛이 어떻게 확산, 흡수되는지 (default: 1.0)
	* 안개, 연기, 물과 같은 반투명 물질에서 중요하게 작용
* Radius: 빛의 확산 범위와 강도
	* Point에서는 point 빛을 방출하는 최대 거리
	* Spot에서는 원뿔의 끝부분에서 얼마나 넓게 퍼질지
	* Area에서는 광원의 크기
	* Sun에서는 큰 차이가 없음
	* 클수록 빛이 닿는 곳과 닿지 않는 곳의 범위가 부드러워짐

#### Shadow
* Clip Start: 그림자가 렌더링이 되기 시작하는 최소 거리 (default: 0.05)
* Bias: 그림자가 표면에서 얼마나 떨어져서 계산되는지
	* self-shadowing 방지
* Contact Shadow: 그림자의 디테일한 부분에 대해 표현하는 기능
	* 가깝거나 세밀한 요소가 많은 object에서는 그림자가 잘 표현되지 않을 수 있음
	* Distance: Contact Shadow가 영향을 미치는 최대 거리를 설정
	* Bias: 그림자가 표면에서 떨어져 계산되는 정도를 설정
	* Thickness: 그림자가 생성될 때 샘플링되는 두께를 설정

### Type
#### Point
광원으로부터 퍼져나감

#### Sun
태양과 같은 직진성 빛

#### Spot
Point와 다르게 원하는 각도에 cone 형식으로만 퍼지는 빛
ex) 손전등, 헤드라이트
* Spot Shape
	* Size: Cone의 퍼지는 각도
	* Blend: 가장자리 부드러움

#### Area
직사각형 형태의 영역


## World
World에 단색이 아니라, sky texture를 넣어서 기본적인 빛을 설정할 수 있음
* World -> Surface -> Color -> Sky Texture

##### Types
1. **Nishita**: 물리적으로 정확한 대기 산란 모델을 사용하여 매우 사실적인 하늘을 생성
2. **Preetham**: 간단한 하늘 모델로, 비교적 빠르고 효율적인 렌더링을 제공
3. **Hosek/Wilkie**: 보다 정확한 하늘 모델로, 다양한 하늘 조건을 시뮬레이션

##### Options
- Air Density: 대기의 밀도를 설정 (높을수록 하늘이 더 뿌옇게 보임)
- Dust Density: 대기 중 먼지의 밀도를 설정 (높을수록 하늘이 더 붉게 보임)
- Ozone Density: 대기 중 오존의 밀도를 설정 (높을수록 하늘이 더 푸르게 보임)

### HDRI (High Dynamic Range Image)
조명 정보가 들어간 map ([Poly Heaven](https://polyhaven.com/hdris))
Environment Texture로 사용할 수 있음
- World -> Surface -> Color -> Environment Texture


## Engine에 따른 조명
### Cycles
 일반적으로 Cycles는 빛의 경로를 다 계산하기 때문에 연산량이 높다.
 Render tab의 light path에서 최대 빛 반사 횟수를 조절하여 연산량을 줄일 수 있다.
 
 ##### Sampling: 빛의 시뮬레이션을 몇 번 할 것인가?
 낮을수록 부하가 없지만 노이즈가 많음. 즉 렌더링이 덜 된다고 볼 수 있음
 - 뷰포트에서만 낮게 설정도 가능
복잡한 Scene에서는 sampling을 높여도 노이즈가 남을 수 있음
- Denoising을 사용

조명의 경우, extra light addon을 이용하여 기존에 설정된 조명들을 가져올 수 있음
##### IES 조명: 강도, 형태 등을 설정해놓은 txt 파일을 통해 조명의 형태를 로드
- IES Texture Node를 통해 사용할 수 있음

##### Color Managements: 빛의 노출, 감마 등 색상을 조절할 수 있는 것
- Render -> Scene에서 설정 가능
- False Color는 조명의 노출을 색으로 볼 수 있음 (초록색이 대부분 눈이 편한 정도)

##### HDRI: 실제 세계의 환경에 대해 조명 영향을 설정할 수 있음
- 즉, 이미지를 조명 대신 쓰는 것
- World->Background에서 설정

##### Volume: 안개, 입자 등을 의미함

##### Image Texture: 조명에도 이미지 텍스쳐를 넣어, 빔 프로젝터같이 쓸 수 있음

##### Sky Texture: 하늘같은 효과를 줄 수 있는 텍스쳐


### EEVEE
화면에 보이는 것만 렌더링하기 때문에 계산이 가볍다.
EEVEE Options
- Sampling 횟수를 이용하여, 계산량과 성능의 trade-off를 조절할 수 있다.
- Shadow
- Screen Space Reflections
- Ambient Occlusion
- Bloom

Light Probe: Bake 방식으로 Cycle 과 비슷한 화면을 구성
- 용도에 따라 배치 후, EEVEE options에서 Indirect Lightning
- Bake Indirect Lightning 설정 후 동작
- 각 probe에 영역의 조명 정보 저장
- 조명이나 객체가 이동하면 다시 baking하는 것이 좋음
	- 제일 마지막에 하는 것이 좋음
