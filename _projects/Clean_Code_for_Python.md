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

##### Properties, Attribute

Python은 다른 언어와 다르게 public, protected, private을 구별하지 않고 모든 property와 함수는 public이다. 하지만 관습적으로 범위를 조정해주는 사항이 있다. (강제는 아님) 모든 속성, 함수는 public이지만 이름 앞에 _ 가 붙으면 내부에서만 사용되고 밖에서는 호출되지 않아야 한다는 암묵적 관습이다. <br>
__ 은 이름 맹글링(name mangling)이라고 한다. ```_<class_name>__<atrribute_name>``` 이름의 속성을 만든다. 확장되는 class의 속성을 이름 충돌없이 override하기 위한 것이다. 

- property
attribute에 대한 접근 제어를 할 때, property를 사용한다. 
```py
class User:
    def __init__(self, username):
        self.username = username
        self._ID = None
@property
def ID(self):
    return self._ID

@property.setter
def ID(self, new_ID):
    if not is_valid_ID(new_ID):
        raise ValueError("Not Valid ID")
    self._ID = new_ID
```

property.setter의 경우 <user>.ID = <new_ID>가 실행될 때 호출된다. 새 값으로 속성을 할당하는 것 뿐만 아니라 유효성 검사도 할 수 있다. <br> 한 method에서는 하나 이상의 일을 하지 않는 것이 좋기 때문에 property를 사용하여, 혼동을 피하는 것이 좋다.

##### Iterable
Python에서 내장 반복형 객체뿐만 아니라 자체 이터러블을 만들 수도 있다. ```for i in object:```의 형태로 객체를 반복하기 위해선 
1. 객체가 __next__ 나 __iter__ 메서드 중 하나를 갖고 있는지
2. 객체가 시퀀스이고, __len__과 __getitem__을 모두 가졌는지
가 필요하다. 
    
##### Container
Container는 ```__contains__``` method를 구현한 객체다. <br>
```__contains__```는 ```in```키워드에서 호출되고 보통 boolean값을 반환한다. 

##### Callable
객체를 함수처럼 동작하게 하기 위해서 ```__call__``` method를 사용한다. 함수와 다르게 좋은 점은 객체엔 상태가 있기 때문에 정보를 저장할 수 있다는 것이다. 

##### Magic Method

|Content|Magic Method|etc|
|:------------|:------------|:------------|
|obj&#91;key&#93; <br> obj&#91;i&#58;j&#93;|&#95;&#95;getitem&#95;&#95;(key)| |
|with obj&#58;|&#95;&#95;enter&#95;&#95; / &#95;&#95;exit&#95;&#95;|context manager|
|for i in obj&#58;|&#95;&#95;iter&#95;&#95; / &#95;&#95;next&#95;&#95; <br> &#95;&#95;len&#95;&#95; / &#95;&#95;getitem&#95;&#95;|iterable <br> sequence|
|obj&#46;&#60;attribute&#62;|&#95;&#95;getattr&#95;&#95;|Dynamic attribute|
|obj(&#42;args, &#42;&#42;kwargs)|&#95;&#95;call&#95;&#95;(&#42;args, &#42;&#42;kwargs)|callable|

