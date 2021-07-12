---
title: Clean Code in Python 3
tags:
  - Code
  - Pytorch
---
Method for Construct Clean Code in Python <br>
SOLID principle
<!--more-->

---
### SOLID principle
---

- S : Single Responsibility Principle
- O : Open Close Principle
- L : Liskov Subsititution Principle
- I : Interface Segregation Principle
- D : Dependency Inverse Principle


##### SPR (Single Responsibility Principle)
Software Component가 단 하나의 구체적인 일을 해야한다는 원칙이다. Component 하나가 너무 많은 일을 하게 된다면, 유지보수가 어려워진다. class의 method들이 서로 상호 배타적이며, 독립적일 때, 코드의 유지보수가 쉬워지고 응집력있는 추상화를 구현할 수 있다. <br>
예를 들어, 어떤 data를 받아, 처리 후 내보내는 application이 있다면, data를 받는 class, 처리하는 class, 내보내는 class를 따로 만드는 것이 유지보수와 응집력에 좋고, class의 재사용 또한 쉬워진다. 

##### OCP (Open Close Principle)
Class가 Open이라는 말은 코드의 확장에 개방되어 있다는 의미이다. 새로운 기능을 추가할 때, 기존의 코드를 고치지 않고, 확장을 하여 수정할 수 있다. <br>
Class가 Close라는 말은 class를 수정해야하는 상황이 있을 때, 해당 기능에 해당하는 class만 수정하면 되도록 코드를 구성한다는 의미이다. <br>
class를 확장에는 개방되고 수정에는 폐쇄되도록 구성하면, 코드의 유지보수가 쉬워진다. 

##### LSP (Liskov Substitution Principle)
설계 시, 안정성을 위해 객체의 타입을 유지해야하는 특성을 말한다. 즉, 파생 클래스와 부모 클래스가 있을 때,  사용자는 두 타입의 객체를 치환해도 application의 실행에 실패하지 않아야 한다. 상속 클래스가 부모 클래스의 다형성을 유지하도록 하기 위해 LSP를 지키는 것이 좋다. 

##### ISP (Interface Segregation Principle)
OOP에서 interface는 객체가 노출하는 method의 집합이다. 
Python에서 interface는 method의 형태를 보고 암묵적으로 결정된다. SPR과 마찬가지로, class는 구체적인 역할의 구분이 있어야하기 때문에 재사용성을 높이기 위해, interface는 대부분 적은 수의 method로 이루어진다. 

#### DIP (Dependency Inverse Principle)
외부 라이브러리나 다른 팀의 모듈을 사용할 때, 그 모듈에 의존하게 된다면 코드가 바뀔 때, 원래의 코드가 깨지게 된다. 따라서 의존성을 역전시킬 필요성이 있다.


