---
layout: default
title: Shading
nav_order: "2024-06-29"
parent: Blender
grand_parent: Development
permalink: /docs/development/blender/shading_2024_06_29
---

# Shading
## Shader Editor
#### Principle BSDF
복합적으로 shader에 관한 기본적인 option들이 모여있음
Diffuse BSDF, Glossy BSDF 등 기본 기능들을 개별적으로 조절하는 shader도 따로 있음
	Diffuse는 metalic이 0, Glossy는 metalic이 1인 상태와 같음

* Specular의 Anisotropic: 비등방성
	* 빛 반사의 표현이 매끈하지 않고 재질에 따라 달라짐
		*  ex) 냄비 뚜껑
* Alpha: 투명도
	* EEVEE에서는 material setting을 blend mode를 alpha blend로 바꿔줘야 함
* Transmission: 얼마나 빛을 투과시키는가
	* ex) 유리
	* IOR(굴절률)과 관련이 되어있음
	* Alpha는 픽셀 자체의 투명화지만, transmission은 재질의 투과율에 관련된 것
	* render tap에서 빛의 투과 횟수를 조절할 수 있음 (Light Path -> Max Bounce)
* Coat: 코팅과 같은 효과
* Subsurface: 표면 아래 확산과 관련된 것
* Sheen: 광택
	* 천과 같은 재질의 가장자리에 생기는 광택
* Emission: 스스로 빛을 내는 것

#### Principle Volume
체적에 관련된 option들이 있음
Material 뿐만 아니라, world에도 이를 이용하여 안개 등을 깔아줄 수 있음
- Density: 체적 밀도
- Emission: 체적 자체적인 발광
- Absorption: 흡수
	- 빛같은것이 들어오면, 점점 흡수하여 어두워짐
- Scatter: 확산
	- 빛이 들어오면, 확산됨
- Blackbody: 자체적인 열이 존재할 때 색을 표현해줌

#### Displacement
요철 등을 의미함. 폴리곤을 조정하지 않고, surfece의 vector는 조절하는 것처럼 보이게 하는 것이라면, displacement는 실제로 폴리곤에 변화를 줌

사용하기 위해, material option에서 displacement 부분을 displacement only 또는 displacement and bump로 변경해줘야 함


#### Material Output
* Surface: 위에서 언급한 BSDF와 같은 재질 등이 들어감
* Volume: 체적
	* 즉, 입자가 차있는것
	* ex) 안개, 구름
	* **Principle BSDF**가 따로 존재함
* Displacement: 객체의 요철
	* 객체의 튀어나온 곳, 들어간 곳을 쉐이더 단에서 조절

#### Etc
* Mix Shader: 두 shader를 섞어주는 것
* Add Shader: 두 shader를 더해주는 것
	* Noise Texture 등으로 랜덤하게 mix, add 해줄 수도 있음
	* Masking 등 여러 방면으로 사용 가능
* Translucent BSDF: Transmission과 비슷한데, 물체 반대편의 빛을 고려함
	* 예를 들어, 나뭇잎을 태양에 비춰봤을 때, 눈으로 보는 부분이 밝게 표현됨
	* 마찬가지로, 빛이 투과된 반대편에 빛이 밝아지는 걸 표현해줌
	* Transmission에서도 가능하지만, translucent에서는 색을 조절할 수 있음
* Subsurface Scattering: 표면 아래의 산란
	* 피부에서 혈관이 보이듯이 표면을 뚫고 내부에 흡수된 것에 대한 것
	* ex) 액체, 피부

# Materials
## Material Properties
모든 vertex 혹은 face 등은 하나의 material만 가질 수 있다.
	여러 texture를 object에 포함할 순 있지만, 실제 적용은 face나 vertex 등엔 하나만 적용할 수 있다는 뜻
object나 face 등을 선택하고 material properties에서 assign을 눌러주면 적용이 된다. 
	Edit Mode에서 해야 함

### Options
- Metalic: 금속 성질을 부여하는 것
- Specular: 빛이 비춰졌을 때 반사율 (대부분 0.5)
- Roughness: 거칠기 (사포질 여부처럼 생각하면 됨)
- Emission: 스스로 발광하는지에 대한 것


# Texture
Shading Editor에서 회색에 해당하는, 즉, 색에 관련된 것을 어떻게 다루는지
- Texture Coordinate: Texture가 object에 적용되는 방식에 대한 노드
	- 좌표형식으로 object에 들어가고, 어떤 좌표방식을 사용하느냐에 대한 것
	- 좌표는 Vector 형식이고, Vector node들을 통해 다시 조정할 수 있음
		- ex) Vector -> mapping

### Blending Mode (Mix Node)
- Mix: 두 색 또는 벡터를 Mix
더 어두워지는 옵션
- Darken: 어두운 부분을 선택
- Multiply: 두 [0, 1]의 곱이라 마찬가지로 어두워짐
- Color Burn: 흰 부분은 과다 노출
더 밝아지는 옵션
- Lighten: 밝은 부분을 선택
- Screen: Multiply를 반대로 수행
- Color Dodge: 비슷하게 밝아지지만, 중간톤이 강조됨
- Add: 두 색을 더 함

- Overlay: 밝은쪽은 screen, 어두운쪽은 multiply가 적용됨
- Soft Light: Overlay에서 부드럽게 색상이 변함
- Linear Light: 밝은곳은 Dodge, 어두운 곳은 burn의 느낌


- Difference: 색의 교차영역 (같으면 검은색이 됨)
- Exclusion: 
- Subtract: 
- Divide: 

- Hue, Saturation, Color: 

#### Vector
Shader의 Normal에 연결되면 가짜 음영 및 요철을 구현할 수 있음
##### Bump
흑백 값으로만 표면의 요철을 표현하는 것

##### Normal Map
각 face의 normal을 이미지의 RGB에 매칭하여 이미지 데이터로 만든 것


# Etc
### PBR (Physical Base Rendering)
일반적으로 material의 option들이 어느 정도 값들을 가지면, 현실 세계의 어떤 재질과 비슷하게 rendering 된다는 렌더링 표현 체계

### 유리 재질 표현
**아래는 Cycles 기준**
Principle BSDF에서 Transimission을 1로 만들어 줌
	Alpha하는 눈으로 보기에 픽셀이 투명해지는 것이기 때문에 빛의 통과율인 transmission을 사용해야 함
유리이기 때문에 Roughness를 0으로 설정
굴절률인 IOR은 실제 유리의 값과 같게 설정 (1.5 ~ 1.7)
	공기는 1.0003
	다이아몬드 같은 재질은 2.419로, 높음
유리의 프리즘같은 빛 반사는 Refraction BSDF를 통해 설정 가능
	빛의 3원색을 각각 생성하여 Add Shader 노드를 통해 섞어줌

유리여도 약간 불투명한 느낌이 나고 싶으면, Glossy Shader를 통해 반짝이는 느낌을 줌
Fresnel 효과도 해당 fresnel node를 통해 줄 수 있음
	무슨 효과인지는 잘 모르겠음 
	밖과 안의 투과율을 다르게 해주는 느낌

**EEVEE**에서는 screen space reflection을 활성화 (보이는 부분에선 반사 렌더를 허용하는 듯)
	Refraction 활성화
Materials에서도 Blend Mode를 Opaque에서 Alpha BHashed로 변경 
	RayTrace Refraction도 활성화
	Depth값도 조금 높여서 굴절률 표현

Input에 Light Path라는 노드를 쓰면, 유리의 그림자를 더 현실적으로 표현할 수 있음
	EEVEE면 재질의 Shadow Mode도 Alpha Hashed로 해야 함


### Principle Hair BSDF Shader
- Roughness: 개별적인 머리카락의 거칠기
- Radial Roughness: 전체적인 머리 모양이 구형인지에 대한 옵션
- Offset: 하이라이트의 각도

- Absorption Coefficient: 각 빛의 값에 대한 흡수
- Melanin Concentration: 모발에서 멜라닌의 함유량

Curves Info Node와 Color Ramp Node를 통해, 머리 길이에 따른 색상 변화를 표현할 수 있음
	예를 들어, 뿌리 염색 등