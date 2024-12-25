---
layout: default
title: Camera & Rendering
nav_order: "2024-06-29"
parent: Blender
grand_parent: Development
permalink: /docs/development/blender/camera_rendering_2024_06_29
---

## Camera



## Renderer
### Cycle
빛, 재질, 상호 작용, 물리 등을 모두 렌더링하는 renderer


### EEVEE
Viewport의 실시간 rasterization을 위해 조명 및 재질만 사용하여 rendering을 수행
	즉, 화면에 보이는 것만 rendering
Unity 등 게임 엔진에 사용하는 것과 같이 real time으로 렌더링을 볼 수 있지만, 여러 trick이 들어갔기 때문에 실제 rendering과는 다를 수 있다. 

* Embient Occulsion: 빛이 잘 닿지 않는 곳도 더 세밀하게 rendering하게 하는 옵션
* Bloom: 빛이 있는 부분을 더 화사하게 표현하는 옵션
* Screen Space Reflection: 화면에 보이는 부분에 대해서 반사 적용
	* Cycle은 안보여도 반사를 적용하지만, 보이는 부분만 적용해준다.
	* 조금이나마 사실적으로 해주는 옵션
* Shadows: 그림자에 대한 해상도 옵션