---
title: Divide & Conquer
tags:
  - Algorithm
  - Code
  - BaekJoon
---

Divide & Conquer (분할 정복)은 주어진 문제를 둘 이상의 부분 문제로 나눈 뒤 문제에 대한 답을 재귀 호출을 이용해 계산하고, 각 부분 문제의 답으로부터 전체 답을 도출해 내는 알고리즘입니다.
<!--more-->

다음 그림은 분할 정복의 대략적인 모습입니다.
<img src="https://user-images.githubusercontent.com/48177363/100746330-e8d98000-3423-11eb-81e3-de946810ff78.PNG" width = "600" height = "400">

1. 문제를 더 작은 문제로 나누고 (divide)
2. base case (더 나누지 않고 바로 풀 수 있는 문제)
3. 하위의 답을 병합하여 상위의 답을 구하는 것<br>
을 통해 문제를 풀 수 있습니다. 

분할 정복 알고리즘은 전체 문제의 계산량에 비하여 base case의 계산량이 압도적으로 작을 때 큰 효율을 보입니다.
분할 정복을 활용한 가장 유명한 사례는 Quick sort입니다. <br>
Quick sort는 pivot이라고 불리는 기준점을 중심으로 더 큰 수는 오른쪽으로, 작은 수는 왼쪽으로 보냅니다. (오름차순 정렬)
이렇게 나눈 왼쪽, 오른쪽으로 보내진 수들에서 다시 pivot을 잡아 분할하는 형식으로 정렬을 구현합니다.
Quick sort의 시간 복잡도는 O(nlogn)이기 때문에 O(n^2)의 시간 복잡도를 갖는 bubble sort나 insert sort보다 훨씬 빠르다.

`BaekJoon 문제`

- [Z(1074)](https://github.com/Nakkwan/Algorithm/blob/master/Baekjoon/Algorithm/Divide%20and%20Conquer/Z(1074).cpp) <br>
- [종이의 개수 (1780)](https://github.com/Nakkwan/Algorithm/blob/master/Baekjoon/Algorithm/Divide%20and%20Conquer/%EC%A2%85%EC%9D%B4%EC%9D%98%20%EA%B0%9C%EC%88%98(1780).cpp)<br>
- [쿼드 트리 (1992)](https://github.com/Nakkwan/Algorithm/blob/master/Baekjoon/Algorithm/Divide%20and%20Conquer/%EC%BF%BC%EB%93%9C%ED%8A%B8%EB%A6%AC(1992).cpp)<br>
