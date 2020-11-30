--
title: Greedy Algorithm
tags:
  - Algorithm
  - Code
  - BaekJoon
---

Greedy Algorithm은 탐욕법을 이용한 알고리즘입니다. 탐욕법은 지금 당장 가장 좋은 방법을 선택해, 문제를 풀어나가는 방법입니다.<br>
에를 들어, 동전의 개수를 최소로 사용하여, 특정 금액을 나타내야하는 문제가 이에 해당합니다. 최소의 동전을 사용하기 위해, 당장 가장 큰 금액의 동전을 최대한 많이 사용하는 겁니다. <br>
또 다른 예로, 도시들을 순회하는데 필요한 거리의 합을 최소화하는 외판원 문제를 생각해 봅시다. Greedy Algorithm에서는 동전 문제와 마찬가지로 현재 지점에서 당장 다음 도시까지 가장 가까운 거리만을 생각합니다. 하지만 이 경우에는 최적해를 찾을 수 없습니다. Greedy Algorithm은 지금의 선택이 다음 경우에 어떤 영향을 끼칠지 생각하지 않기 때문에 많은 경우에 최적의 해를 구할 수 없습니다. <br>
따라서 Greedy Algorithm을 쓰는 경우의 수는 크게 두가지로 나뉩니다.
- Greedy Algorithm을 사용해도 항상 최적해를 찾을 수 있는 경우
  - 동전 문제가 이에 해당합니다. 큰 크기의 동전은 작은 크기의 동전의 배수가 되기 때문입니다. (작은 크기의 동전으로 큰 동전을 표현할 수 있죠)
- 시간적, 공간적 제약으로 인해, 최적해 대신 인접한 임의의 답을 구하기 위한 경우

첫번째 경우에는 직관적으로 항상 최적해를 찾을 수 있는지 알기 힘들기 때문에 실수에 주의해서 문제를 풀어야합니다.

'BaekJoon 문제'

- [캠핑(4796)](https://github.com/Nakkwan/Algorithm/blob/master/Baekjoon/Algorithm/Greedy%20Algorithm/%EC%BA%A0%ED%95%91(4796).cpp) <br>
- [ATM (11399)](https://github.com/Nakkwan/Algorithm/blob/master/Baekjoon/Algorithm/Greedy%20Algorithm/ATM(11399).cpp)<br>
- [과제 (13904)](https://github.com/Nakkwan/Algorithm/blob/master/Baekjoon/Algorithm/Greedy%20Algorithm/%EA%B3%BC%EC%A0%9C(13904).cpp)<br>
- [And the winner is ourselves (17509)](https://github.com/Nakkwan/Algorithm/blob/master/Baekjoon/Algorithm/Greedy%20Algorithm/And%20the%20winner%20is%20ourselves(17509).cpp)

