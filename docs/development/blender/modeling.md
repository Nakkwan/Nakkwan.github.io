---
layout: default
title: Modeling
nav_order: "2024-06-29"
parent: Blender
grand_parent: Development
permalink: /docs/development/blender/modeling_2024_06_29
---

## Modeling 용어
* Primitive: 기본적으로 정의되어있는 모델
	* Add-on에 extra object를 추가할 수 있음

## Modeling Tools
Tab 키를 눌러, edit mode에서 사용하는 modeling tools
- Extrude: 선택한 점, 선, 면 등을 돌출시킴 (Defaults: Region)
	- 확장 탭에 여러 옵션이 있음
	- Manifold: 기존에 있는걸 깍듯이 동작함
	- Along Normal: normal을 따라 돌출 (붙어있는 면은 붙어서 extrude될 수 있음)
	- Individual: 여러 선택 면에 대해 각자 normal에 따라 돌출
	- Cursor: 클릭하는 곳까지 extrude
- Inset face: face를 안에서 생성
	- 벽을 세우거나, 면의 일부만 돌출 시키거나 할 때 유용하게 사용
	- 어떻게 inset할지, 다양한 옵션이 있음
	- **i**는 individual의 단축키
- Knife: 모델을 자를 수 있는 기능
	- 원하는 부분을 선택하고 스페이스 바를 누르면 적용이 됨
- Bisect: knife는 칼집만 내지만, bisect는 아예 잘라내버림
	- Outer, Inner, Fill 등의 옵션이 있음
- Bevel(Ctrl+B): 선택한 부분을 깍아내는 기능
	- segment 옵션을 사용하면 둥글게 나눠서 깍아줄 수 있음 (wheel로도 조정 가능)
	- bevel은 object의 scale이 면마다 다르면 모양이 다르게 scale 될 수 있음
	- Ctrl+A를 통해 scale을 1:1로 만들고 다시 bevel
	- Profile Type에서 custom을 사용하면, 2D UI에서 더 세밀하게 깍을 수 있음 
	- Edge 뿐만 아니라, vertex도 깍을 수 있음
- Loop Cut(Ctrl+R): 선을 object에 둘러서 넣기
	- wheel로 개수 조정 가능
	- 클릭을 꾹 눌러서 위치 조정
	- 위치 조정 중, E키를 누르면 선의 모양을 근처  edge에 참고하여 변경 가능
	- alt 또는 ctrl+alt 선택으로 링 전체 선택 가능 
	- Ctrl+Shift+R: 선택한 edge를 기준으로 양쪽으로 line을 생성
- Slide: Edge, vertex 등의 위치 변경
	- move와 다르게 옆 선, 면을 따라서 이동하기 떄문에 깔끔하게 이동 가능
	- G를 누르면 move, G를 두번 누르면 slide
- Spin: 커서를 중심으로 선택 object를 회전
	- 회전 UI에서 생성된 축을 이동하여, spin에 offset도 줄 수 있음
- Shear: x,y,z 중 누른 축에 대하여 선택한 것을 중심으로 기울임
- Shrink: 크기를 조절하는 scale과 다르게, 원하는 부분만 수축
- Fill(F or Alt+F): 빈 face 부분을 채우기
	- 채울 곳의 개수들이 맞을 땐, grid fill
- Rip/Merge: 붙은 mesh를 뜯거나 붙임
- Join/Seperate: 두 mesh를 합치거나 분리함


#### Etc (우클릭)
- Bridge Edge Loop: 짝이 맞는 떨어진 edge들을 이어서 면으로 만듦
- Subdivide: 절반으로 분할
	- 반대는 Un-subdivide
- Triangulate: mesh를 모두 삼각화
	- 사각은 tris to quad
	- poke face는 NGON을 중심을 기준으로 모음
- 

## Tool bar
### Mesh
- Transform
	- Randomize: Noise가 생기듯이 vertex 등을 랜덤하게 형태가 변형됨
	- To sphere(Shift+Alt+S): 모든  vertex를 최대한 구 형태로 만들어줌


## Modifier
### Modify


### Generate
- Array: 객체를 나열하는 것
- Bevel: Modeling Tool의 bevel과 유사한 기능
	- modeling tool은 적용을 되돌릴 수 없기 때문에 modifier에서 적용할 수 있음
- Boolean: 객체간의 형태를 합치거나 깍는 것
- Build: 모델을 만드는 것 같은 애니메이션을 추가하는 것
- Decimate: 폴리곤을 줄일 때 사용
- Edge Split: 특정 조건의 edge를 분리함
- Mask: 특정 polygon을 안보이게 해줌
- Mirror: 복제를 하는 것 (거울과 비슷하게)
	- 모델링 시, 반을 쪼개서 양쪽을 대칭되게 모델링 하고 싶을 때 사용
- Multi-resolution: 여러 해상도의 정보를 가지고, 적용할 수 있음
- Remesh: Mesh를 어떤 알고리즘에 의해 다시 polygon을 생성하는 것
- Screw: 객체를 vertex를 이용하여 꼬아주는 것
- Skin: 객체에 살을 붙이는 것
	- 예를 들어, vertex 또는 edge에 살이 붙음 (ex. 나뭇가지)
	- Armature에서 뼈대도 만들어 줄 수 있음
- Solidify: 면 등에 두께를 만들어줌
	- Plat만 만들고 이후에 solidify로 두께감을 줄 수 있음
- Subdivide Surface: 표면을 세분화함
- Traiangluate: Polygon을 traiangle로 만듦
- Volume to Mesh: 입자, 구름과 같은 object를 mesh로 변환 
	- volume: vdb와 같은 형식의 파일 
- Weld: 용접과 같은 느낌
	- 가까운 polygon들을 붙이는 것
- Wireframe: 원래 있는 객체의 edge를 철망처럼 만듦


### Deform


### Physics



## Reference
1. [블렌더3D 아카데미 심화 - 모델링 시리즈 - YouTube](https://www.youtube.com/playlist?list=PLUgnJ9nL1WEOmsREQ1uvLLDLyeKFFt85p)
2. 