---
layout: default
title: Short Cut
nav_order: "2024.05.24"
parent: Blender
grand_parent: Development
permalink: /docs/development/blender/short_cut_2024_05_24
---

기본적으로 blender에서 **ctrl**, **alt**는 해당 동작의 반대로 수행하는 키.
**Shift**는 팝업같은 추가 동작이나, 비슷한 동작을 수행하는 키.

# Move
1.  **Wheel**: 시점 회전
2.  **Shift + Wheel**: 시점 이동

# Layout
### PopUp
1. **Shift+S**: Cursor 관련 메뉴
2. **Shift+A**: 객체 추가 관련 메뉴 (Add)
3. **Z**: Viewport 변경 PopUp

### Mode
1. **Tab**: Edit 모드 (객체 선택 후 가능)
2.  **Shift+Tab**: Snap
	1. 객체를 움직일 때, increment (칸 단위), Vertex (점에 붙어서) 등을 선택 가능
3. **O**: Proportion Editing (Transform을 적용할 때, 주변 객체 및 요소에도 같이 영향을 미침)
4. **Alt+Z**: X-ray 모드 활성화
5. **Ctrl+Space**: 전체 화면

# Object
### Transform
각 단축키에서 x,y,z를 누르면 해당 축으로면 변환을 할 수 있음
단축키를 입력한 상태에서 숫자를 누르면 해당 값만큼 변환을 수행할 수 있음
1.  **W**: 객체 선택 모드 
2.  **G**: 객체 이동
3.  **R**: 객체 회전
4.  **S**: 객체 scaling

### Select
1.  **A**: 모든 객체 선택
2.  **Shift+Click**: 객체 추가 선택
3.  **Ctrl+I**: 선택한 객체 이외의 모든 객체 선택 (Invert)

### Relation
1.  **Ctrl+P**: 객체 2개 선택 후 입력하면, 마지막에 선택된 객체가 부모가 됨
2.  **Alt+P**: 부모 관계 해제 

### Operate
1.  **Shift+D**: 객체 복제
2.  **Alt+D**: 객체 복제 (Object data 공유 (원본 데이터 수정하면 같이 수정됨))
3.  **H**: 해당 객체 숨기기
4.  **Alt+H**: 숨겼던 객체 표시
5.  **M**: 해당 객체를 collection에서 이동
6.  **Shift+M**: 해당 객체를 다른 collection에 Link

# Mode
## Edit Mode 
### Select
1. 숫자 **1**: Vertex 편집
2. 숫자 **2**: Edge 편집
3. 숫자 **3**: Face 편집

### Operate
1.  **X**: 삭제 (편집 모드에서 삭제기 때문에 객체 자체는 남아있음)
2.  **Ctrl+L**: Vertex가 연결되어 있는 객체만 전체 선택
3.  **Ctrl+J**: 선택된 객체들을 하나의 객체로 합치기 (join)
4.  **P**: Seperate (객체에서 파트 분리)
5.  **Alt+N**: 객체의 normal 뒤집기
