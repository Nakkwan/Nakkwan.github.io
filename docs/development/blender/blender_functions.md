---
layout: default
title: Blender Functions
nav_order: "2024-06-30"
parent: Blender
grand_parent: Development
permalink: /docs/development/blender/blender_functions_2024_06_30
---


# Greeds Pencil
3D 공간 상의 blender에서 2D 그림 등을 지원하는 기능 (Shift+A로 추가)
Draw Mode에서 그림을 그릴 수 있음
	그림은 카메라가 보는 방향과 수직으로 그려짐
	File -> New -> 2D Animation은 그림판처럼 쓸 수 있음 (3D임)\

### Properties
- Onion Skinning: 해당 프레임의 그림이 앞뒤 몇 프레임까지 남아있는지

### Modifier
- Build: 그림이나 글씨 stroke를 입력한 대로 애니메이션처럼 만들어줌
	- 예를 들어, 글씨를 쓰거나, 만화를 그리는 과정처럼 애니메이션이 됨
- Noise: animation에 random한 변경 효과를 줌
