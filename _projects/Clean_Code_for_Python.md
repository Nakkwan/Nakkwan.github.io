---
title: Clean Code for Python
---
Pythonic한 code를 위한 Clean code method <br>
PEP-8 (Python Enhancement Proposal 8번)에 따른 Style Guide for python code.

---
### Code Formatting
---
Grepability를 위해서, keyword argument에는 띄어쓰기를 사용하지 않고, value에 값을 할당할 때에는 띄어쓰기를 사용한다. <br>
```py
$ grep -nr "location=".
./core/py:13: location=current_location,
```
```py
$ grep -nr "location =".
./core/py:10: location = get_location()
```

##### Docstring
docstring은 주석이 아니라 코드로 분류된다. 코드를 설명하는 주석을 다는 것보다(주석은 오해를 불러일으킬 수도 있음) 코드의 특징 component에 대한 문서화를 위해 doctring을 사용한다. 결국 clean code는 개발자가 개발자에게 이해하기 쉽도록 coding을 하는 것이다. <br>
docstring은 module, class, method들의 입력, 출력 등을 문서화하기 때문에 사용자가 함수에 대한 동작을 이해하기 쉽다. docstring은 console이나 Ipython에서도 확인할 수 있다. ```In [1]: dict.update??```와 같이 입력한다면, dict.update 함수에 대한 docstring을 출력할 수 있다. 또한 객체에 ```__doc__``` method가 정의되어 있다면, 런타임 중이나 소스코드 내에서도 docstring에 접근할 수 있다.

##### Annotation
Annotation은 PEP-3107에서 소개되어 있다. Annotation은 dynamic typing을 하는 python에서 인자로 어떤 타입이 와야하는지 힌트를 줄 수 있다. 타입뿐만 아니라 변수에 대한 설명같은 str이 올 수 있으며 어떤 metadata 형식이더라도 사용될 수 있다. 하지만 annotation의 타입이 compile시에 강제되는 것은 아니다. annotation이 사용된다면, ```__annoation__```이라는 attribute가 생기기때문에 
```py
In [1]: locate.__annotation__
{'latitude': float, 'return': float}
```
과 같이 사용될 수 있다.
