---
layout: default
title: Sculpting
nav_order: "2024-07-01"
parent: Blender
grand_parent: Development
permalink: /docs/development/blender/sculpting_2024_07_01
---

모델을 세밀하게 조각할 수 있는 기능
다양한 브러쉬를 사용하거나 조작할 수 있다.
* Sculpting Mode에서 사용 가능

### Brush
- **Draw**: 기본적인 브러시로, 표면을 돌출시키거나 움푹 들어가게 함
- **Clay Strips**: 진흙을 쌓는 것처럼 레이어를 추가
- **Smooth**: 표면을 부드럽게 힘
- **Crease**: 깊은 주름이나 홈을 만듦
- **Grab**: 모델의 일부를 잡고 이동

### Dyntopo (Dynamic Topology)
모델의 디테일 수준을 조절할 수 있는 기능
필요에 따라 더 많은 기하학적 디테일을 추가
	즉, 기존 sculpting은 애초에 poly를 많이 생성하여 modeling을 시작하지만, dyntopo의 경우, sculpting을 하는 부분만 high-poly로 modeling을 수행함
	기존 모델에 적용할 때, vertex, UV 등이 깨질 수 있음
오른쪽 상단에 토글 옵션이 있음

### Multi-resolution Modifier
모델의 해상도를 여러 레벨로 설정할 수 있음
낮은 해상도에서는 큰 형태를 잡고, 높은 해상도에서는 세밀한 디테일을 작업
Modifier에서 해당 기능 추가 가능

나중에 이 기능을 활용하여 low-poly로 baking 가능

### Remesh
Mesh의 polygon을 완전히 다시 생성하는 것
Low->High, High->Low poly로 변환이 모두 가능

Sculpting 말고 일반 mesh properties에서도 사용 가능
- 해당 기능에서 Quad는 3D Voxel 기준이 아니라, poly를 가능하면 사각형으로 만드는 것