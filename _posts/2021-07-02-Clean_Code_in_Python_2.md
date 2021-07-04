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

- Error Handling <br>
  1) 값 대체 <br>
데이터가 제공되지 않았을 때 기본 데이터를 쓰는 방식 등이 있지만 적용 분야에 한계가 있다.

  2) Error logging <br>
  3) Exception <br>
외부 component의 문제로 error가 발생했을 시, 해당 error에 대한 정보를 알려주기 위해 exception 처리를 한다. <br>
호출자에게 잘못을 알려주지만, 캡슐화를 약화시키기 때문에 사용해야할 상황을 신중히 고르는 것이 좋다. <br>
함수 내부적으로 예외처리를 할 수 있다면 호출 단계에서 exception을 수행하기보단 함수 내부에서 exception을 수행하여 호출 함수에서는 호출만 하는 것이 코드를 읽고 이해하기 편하다. <br>
```py
try:
    function()
except:
    pass
```
와 같이 광범위한 exception보다 자세한 예외 (ex: KeyError, ValueError)와 같은 예외를 사용한다면, 사용자가 프로그램을 유지보수하기 쉽다. <br>
오류 처리 과정에서 오류의 타입을 바꾸고 싶다면, ```raise <e> from <original_error>``` 구문을 사용한다면, 새로운 exception에 원본의 traceback이 포함이 되고 ```__cause__```속성으로 
설정이 된다.<br>
    
  4) Assertion <br>
Assertion은 절대로 일어나지 않아야 할 상황에 사용되므로 발생하게 된다면 보통 프로그램을 중지해야 할 상황이다. 

##### Cohesion & Coupling
- Cohesion
객체는 작고 잘 정의된 목적을 가져야한다. 즉, 한가지 일만 수행하여야 재사용성이 좋고 응집력이 높다.

- Coupling
두 개 이상의 객체가 서로 너무 의존적이라면 바람직하지 않다. 두 객체가 서로 너무 의존적이라면, 서로 영향을 끼치기 때문에 **재사용성이 낮고**, 하나를 바꿨을 때 다른 객체에도 영향을 끼치는 **ripple**이 생기고, **낮은 추상화 수준**을 가지게 되기 때문이다. 

---
#### Composition & Inheritance

상속은 부모 클래스의 method를 얻을 수 있어서 코두 재사용에 좋지만 쓸모 없느 기능까지 가져오게 되는 단점이 있다. 상속은 부모 클래스의 속성을 그대로 이어받으면서 기능을 추가하려고 한 경우가 가장 좋은 상속의 예다.<br>

예를 들어, 고객에 대한 정보를 관리하는 코드의 경우 
```py
class Policy(collections.UserDict):
    def change_policy(self, customer_id, **new_policy_data):
        self[customer_id].update(**new_policy_data)
```
의 경우 UserDict을 사용해서 원하는 기능을 수행할 수 있게 되지만, UserDict을 상속받았기 때문에, ```pop```이나 ```items```같은 원치 않는 method도 같이 포함되게 된다. <br> 
따라서 이런 경우, ```__getitem__```과 ```__len__``` method를 추가해, private으로 설정된 dict을 가져오는 것이 좋다.
```py
class Policy:
    def __init__(self, policy_Data, **extra_data):
        self._data = {**policy_Data, **extra_data}
        
    def change_policy(self, customer_id, **new_policy_data):
        self._data[customer_id].update(**new_policy_data)
        
    def __getitem__(self, customer_id):
        return self._data[customer_id]
        
    delf __len__(self):
        return lem(self._data)
```
위와 같은 코드는 재사용성과 확장성도 더 뛰어나다.

##### MultiInheritance

Python은 다중 상속을 지원하지만 올바르게 구현되지 않으면 문제를 야기할 수 있다.(ex: 두 부모에 같은 이름의 method가 존재할 경우) <br>
Python에서는 오른쪽에서 왼쪽의 순서로 상속을 하게 되는데 먼저 상속된 class를 덮어버리는 형식으로 상속된다. 즉, 왼쪽 class가 더 상위 class가 된다.
```py
class base1
    def Mixin(self):
        print("base1 class")
        
class base2
    def Mixin(self):
        print("base2 class")

class Child(base1, base2):
    def Mixin(self):
        pass

>>> a = Child()
>>> a.Mixin
base1 class
```

##### 가변인자
argument에 대해 packing 또는 unpacking을 한고 싶다면, variable 앞에 asterisk(\*)를 붙이면 된다.
```py
def series(first, second, third):
    print(first)
    print(second)
    print(third)
    
>>> l = [1, 2, 3]
>>> f(*l)
1
2
3
```
```py
>>> a, *b, c = [1, 2, 3, 4]
>>> a
1
>>> b
[2, 3]
>>> c
4
```




