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
Annotation은 PEP-3107에서 소개되어 있다. Annotation은 dynamic typing을 하는 python에서 인자로 어떤 타입이 와야하는지 힌트를 줄 수 있다. 타입뿐만 아니라 변수에 대한 설명같은 str이 올 수 있으며 어떤 metadata 형식이더라도 사용될 수 있다. <br> 
하지만 annotation의 타입이 compile시에 강제되는 것은 아니다. annotation이 사용된다면, ```__annoation__```이라는 attribute가 생기기때문에 
```py
In [1]: locate.__annotation__
{'latitude': float, 'return': float}
```
과 같이 사용될 수 있다.

##### index
Python에서는 index 접근 방식에 많은 방법이 존재한다. 음수 indexing과, slicing을 이용한 indexing 모두 가능한데, 사용자 클래스에서도 ```__getitem__```과 ```__len__```method를 통해 구현할 수 있다.

- __getitem__
```py
def __getitem__(self, item):
    pass
```
와 같이 ```__getitem__``` method를 구현할 수 있다. item (index)값에 따라서 return 값이 바뀌게 코드를 짜면 되지만, <br>
1. indexing의 결과는 해당 class와 같은 타입이어야 한다. <br>
2. slicing의 범위는 python과 같이 첫 요소는 포함, 마지막 요소는 제외해야 한다. <br>
라는 규칙을 따르면 헷갈리지 않고, 사용할 수 있다.

##### Context Manage

Context Manager는 코드 실행 전후 작업에 유용하다. 보통 리소스 작업을 할 때 많이 쓰인다. 예를 들어, 파일을 열고 작업을 끝내면 파일을 닫아야, leak을 방지할 수 있다. 예외나 오류처리를 위해선 디버깅의 어려움을 해결하기 위해 ```finally:```블록에 리소스 해제 코드를 작성할 수 있다. python에서는 같은 기능을 ```with```을 사용하여 구현할 수 있다.

- with
```py
with torch.no_grad():
    pass
```
```with```은 ```__enter__``` method를 호출한다. 반환값이 있다면, 구문 마지막의 ```as``` 이후 변수에 할당될 수 있다. 코드 블록이 끝나거나 예외, 오류가 발생하면 ```__exit__``` method를 호출해, 정리 코드를 실행할 수 있다. 위의 코드의 경우, ```with``` 구문이 시작이 되면 gradient를 계산을 멈추고, 구문이 끝난 이후에는 gradient 계산을 다시 실행하는 것을 확인할 수 있다. 또한 ```with``` 구문은 정리 코드뿐만 아니라, 코드를 분리하거나, 특이사항을 처리할 때 쓰이기도 한다. <br>
Context Manager는 ```__enter__```와 ```__exit__``` 매직 매소드 이외에도 contextlib이라는 module을 이용하여, 더 간결하게 코드를 적용할 수 있다. 








