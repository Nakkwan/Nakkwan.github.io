---
title: Clean Code in Python 2
tags:
  - Code
  - Pytorch
---
Method for Construct Clean Code in Python <br>
General feature of good code
<!--more-->

---
### General feature of good code
---

##### Design by Contract
Software Component간 통신 중에 지켜야할 암묵적 규칙<br>

- Precondition
코드가 실행되기 이전에 처리되어야 하는 조건. 일반적으로 parameter에 대한 유효성 검사 등이 포함된다. <br>
검증 로직은 client나 함수 내부에 둘 수 있다.

- Postcondition
함수의 return에 대한 유효성 검사. 호출자가 return값을 제대로 받았는지 확인하기 위해 수행.

- Invariant
함수가 실행되는 동안 일정하게 유지되는 것으로 Docstring에 문서화하는 것이 좋다.

- Side-Effect
코드에 대한 부작용. 마찬가지로 Docstring에 정리할 수 있다.

##### Defensive Programming
에측 가능한 오류(Procedure)와 발생하지 않아야할 오류(Assertion)에 대한 예외처리

- Error Handling
  1) 값 대체
데이터가 제공되지 않았을 때 기본 데이터를 쓰는 방식 등이 있지만 적용 분야에 한계가 있다.

  2) Error logging
  3) Exception
외부 component의 문제로 error가 발생했을 시, 해당 error에 대한 정보를 알려주기 위해 exception 처리를 한다. <br>
호출자에게 잘못을 알려주지만, 캡슐화를 약화시키기 때문에 사용해야할 상황을 신중히 고르는 것이 좋다. <br>
함수 내부적으로 예외처리를 할 수 있다면 호출 단계에서 exception을 수행하기보단 함수 내부에서 exception을 수행하여 호출 함수에서는 호출만 하는 것이 코드를 읽고 이해하기 편하다.
      ```py
      try:
          function()
      except:
          pass
      ```
    
      와 같이 광범위한 exception보다 자세한 예외 (ex: KeyError, ValueError)와 같은 예외를 사용한다면, 사용자가 프로그램을 유지보수하기 쉽다. <br>
      오류 처리 과정에서 오류의 타입을 바꾸고 싶다면, ```raise <e> from <original_error>``` 구문을 사용한다면, 새로운 exception에 원본의 traceback이 포함이 되고 ```__cause__```속성으로 
      설정이 된다.
    
  4) Assertion
Assertion은 절대로 일어나지 않아야 할 상황에 사용되므로 발생하게 된다면 보통 프로그램을 중지해야 할 상황이다. 

##### Cohesion & Coupling
- Cohesion
객체는 작고 잘 정의된 목적을 가져야한다. 즉, 한가지 일만 수행하여야 재사용성이 좋고 응집력이 높다.

- Coupling
두 개 이상의 객체가 서로 너무 의존적이라면 바람직하지 않다. 두 객체가 서로 너무 의존적이라면, 서로 영향을 끼치기 때문에 **재사용성이 낮고**, 하나를 바꿨을 때 다른 객체에도 영향을 끼치는 **ripple**이 생기고, **낮은 추상화 수준**을 가지게 되기 때문이다. 








