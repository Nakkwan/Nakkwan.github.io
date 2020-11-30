---
title: Why numbering should start at Zero (Dijikstra)
---

대부분의 프로그래밍 언어에서는 half-open interval과 zero-based numbering을 사용한다. 다익스트라는 그것이 타당한 이유에 대해 설명하였다.

우선 interval을 표현하는 방식으로는
1) [a, b)
2) (a, b]
3) (a, b)
4) [a, b]
네가지가 있다.

half-open interval은 a), b) 두가지 방식을 얘기한다. 우선적으로 c), d)에 비해 a), b)의 방식이 더 좋은데,
1) b - a가 그 interval의 크기와 같기 때문이다.
2) 두 개의 연속하는 집합을 붙였을 때 지우거나 더해야하는 것 없이 깔끔하기 때문이다.
	ex) [a, b) + [b, c) = [a, c)
또한 a)와 b) 중에는 a)의 방식을 사용하는 것이 더 타당하다. 제일 작은 자연수를 0이라 했을 때, interval의 시작 수를 포함하지 않았을 경우, [-1, b]의 형식으로, 깔끔하지 않다. 
또한 끝 수를 포함한다면 공집합을 표현할 때, -1 같은 수를 넣어야하기 때문에 깔끔하지 않다. ex)(a, -1]
따라서 interval의 표현방식은 [a, b)가 가장 좋은 표현방식이라 할 수 있다.

zero-based numbering의 경우에는 for문 같은 반복문을 쓸때, index가 1부터 시작한다면 for(int i = 1; i < N + 1; i++)의 형태로, N + 1같이 깔끔하지 않다.
하지만 0부터 시작한다면 for(int i = 0; i < N; i++)같이 깔끔하고 interval의 크기를 바로 알수있기 때문에 index는 0부터 시작하는 것이  효율적이고 타당하다.
