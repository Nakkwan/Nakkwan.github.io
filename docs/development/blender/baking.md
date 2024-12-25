---
layout: default
title: Baking
nav_order: "2024-07-02"
parent: Blender
grand_parent: Development
permalink: /docs/development/blender/baking_2024_07_02
---

조명 정보, 객체 정보 등을 texture에 굽는 것
Baking을 위해선 해당 객체의 UV가 켜져 있고, Cycle을 사용해야 함
Texture에서 Image Texture 사용

### Render (Bake Option)


### Ambient Occulsion
조명으로는 부족한 물체의 음영에 관한 베이킹
Bake Type을 Ambient Occulsion으로 설정하여 사용 가능
 Geometry의 간격 등에 따라 계산됨
	 예를 들어, 좁은 틈의 경우 어둡게 baking 됨
	 Sampling 개수에 따라 정밀하게 계산되는 정도가 달라짐


### Normal Map Baking (Multi-Resolution)
##### Normal Map
3D 모델의 표면 디테일을 더 정교하게 표현하기 위해 사용하는 텍스처 맵
표면의 작은 굴곡이나 디테일을 실제로 모델링하지 않고도 표현할 수 있도록 해줌

RGB 채널을 사용하여 각 픽셀의 표면 방향(normal vector)을 저장하고, 이 정보를 렌더링 과정에서 빛의 반사를 계산하는 데 사용함

Blender Shader Editor에 Normal map을 Image Texture에 추가
Normal Map Node를 사용하여, texture를 불러오고, 이를 normal input에 연결

Multi-Resolution Modifier를 사용한 경우, baking 옵션에서 체크
	High-res에서 Low-res로 normal map을 전송하는 형식으로 multi-res가 구성됨
	High를 Low로 전송함으로써 Low로도 High poly처럼 보이게 할 수 있음
	**Selected to Active**로 전송 가능


