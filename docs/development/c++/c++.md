---
layout: default
title: C++
nav_order: 2
parent: Development
has_children: true
permalink: /docs/development/c++
---

# C++
{: .no_toc}
Solutions and workarounds for errors encountered during the development process

1. TOC
{:toc}

궁금한점

1. 왜 int main 뒤에 void가 들어가는지
	1. 사실, void가 안들어가도 됨
	2. C언어에서는 void가 들어가지 않으면, 아무변수나 함수에 인수가 될 수 있었기 때문에 인수가 없다는 뜻으로 void를 사용
	3. C++에서는 main()이어도 main(int)로 인수가 사용될 수 없음
	4. 따라서, c++에서는 main(void)가 아니어도 됨
2. namespace가 정확히 뭔지
	1. 겹치는 함수 명에 대한 구분을 해주기 위하여
	2. 이름 공간안에 이름공간을 넣을 수 있고, ::를 통해서 각 이름공간과 함수를 부를 수 있음
		a. namespace::function
	3. 이름 공간 또한 정의부와 선언부를 나눌 수 있음
	4. using이라는 선언은 using이 선언된 범위에선, 해당 이름 공간을 사용하겠다는 뜻
		a. ex) using namespace std; -> 해당 공간에서 std는 생략해도 됨
		b. ex) using hybrid::Hybrid; -> Hybrid라는 function이 선언될 떈, 현재 공간에선, hybrid라는 이름 공간을 사용하겠다.
		c. 선언 지역을 벗어나면, 효과가 사라짐
	5. 이름공간이 과도하게 겹칠 경우, 다음과 같이 간소화할 수 있음
		a. ex) namespace AAA=AAA::BBB::CCC
	6. 또한 전역변수와 지역변수가 겹쳐질때, 지역변수가 우선시됨
		a. namespace를 통해 전역변수에 접근할 수 있음
		b. ex) ::val += 7 -> 지역변수에 val이라는 변수가 있더라도, 전역변수의 val에 7을 더함
3. DLL이 뭔지
4. compiler별 차이에 대해서
	1. gcc, g++, make, Cmake
		a. GCC는 C 언어 전용 complier로, c++을 사용할 시엔, g++을 사용하는 것이 더 좋다
		b.  CMake는 make를 위한 makefile들을 자동으로 생성해줄 수 있음
		c. make는 실제 빌드를 위한 것으로, g++, gcc 같은 compiler들을 사용함
		d. compiler들이 많기 때문에 make를 통해 이런 것들을 실행, 조정할 수 있음
		e. 따라서, cmake (플랫폼별로 자동 생성해줌)로 프로젝트에 대한 makefile을 만들고, makefile을 기반으로 make를 실행하여 compile
5. 헤더파일을 왜 쓰는걸까?
	1. 함수의 선언과 정의를 구분하기 위하여
	2. 파일을 나누는 기준
		a. h파일	
			i. 클래스의 선언을 담고 있음
			ii. 헤더파일 중복을 해결하기 위해서 #ifdef, #define, #endif를 사용해줄 수 잇음
			iii. #ifndef __header_def_
			iv. #define __header_def_
			v. #endif
			vi. 즉, 다른 헤더파일에서 __header_def_를 선언하지 않았다면, 선언해주고 해당 헤더파일을 컴파일해줌
			vii.  인라인 함수 또한 header에 포함
		b. Cpp파일	
			i. 클래스의 정의를 담고 있음
			ii. 생성자도 예외없이 cpp에 구현
		c. main 파일
			i. .h 파일을 include함
6. sln에 대해서?
7. collection에 대해서
8. C언어의 메모리 구조: 스택, 힙, 데이터 (데이터 세그먼트, 텍스트 세그먼트)
9. 포인터와 참조
10. const란?
	1. 해당 변수 및 함수를 상수로 취급하겠다는 뜻
	2. 즉, const가 선언된 값들은 변경할 수 없음
		a. const int *y의 경우 y가 가르키는 주소의 변수값을 변경할 수 없고,
		b. Int* const y의 경우 주소값 자체를 상수화하는 것이기 때문에 y가 가르키는 주소를 변경할 수 없음
	3. const int *&y = 50과 같이도 사용할 수 있는데, 50을 임시 변수에 넣엇기 때문
	4. 함수 뒤에 const가 들어가는 경우
		a. ex) int number(int x, int y) const{return 0;}
		b. 함수 내에서 멤버 변수의 값을 변경하지 않겠다는 것
		c. const 함수에서는 const 함수만 호출이 가능함
		d. 또한 해당 class의 instance가 const로 다른 함수에 참조됐다면, 역시 const 멤버 함수만 호출 가능
	5. const도 overloading의 조건에 해당됨
11. 전처리기부분의 함수들
	1. #define
	2. #include
	3. #ifdef
12. struct vs class
	1. struct는 단순히 변수의 모음이라고 볼 수 있음
		a. 새로운 변수를 만들 때 사용
		b. function도 포함가능
	2. class는 새로운 객체를 만들 때 사용
		a. 접근 제어자 사용 가능
		b. struct도 접근 제어자를 사용할 수 있지만, 기본이 public이고 class는 기본이 private임
13. Class
	1. 생성자
   	1. class와 이름이 같고 반환형이 없는 함수
   	2. classname(int x, int y){cx = x, cy = y}와 같이 선언할 수도 있지만, member initializer도 사용 가능
   	3. classname(int x, int y):cx(c), cy(y) {}
   	4. member initializer는 정의부(cpp)에서 작성함 
   	5. 생성자가 없으면 defualt constructor가 실행됨
   	6. member initializer는 선언과 동시에 초기화가 이뤄지는 binary 코드이기 때문에 성능 이점이 있음
      	1. 따라서 const 멤버 변수도 member initializer를 사용하면 초기화 가능
   	7. 복사 생성자
      	1. Classname inst1 = inst2; (또는 Classname inst1(inst2); )의 경우, Classname을 인자로 받는 생성자가 없어도 동작함
         	1. default 복사 생성자가 있기 때문
      	2. AAA obj1 = obj2의 경우, AAA obj1(obj2)로 묵시적으로 변환됨
         	1. 이런 묵시적 변환을 막고 싶다면, explicit 키워드를 사용하여 복사 생성자를 선언
            	1. AAA obj1 = obj2; 가 안됨 
      	3. 또한 복사 생성자의 매개 변수는 참조형이어야 함
         	1. ex) classn(classn &copy):num1(copy.num1), num2(copy.num2)
      	4. 호출 시점
         	1. 기존 객체를 이용하여 새로운 객체 초기화
         	2. call-by-value
         	3. 참조형이 아닌 객체 반환
	2. 소멸자
   	1. class 이름에 ~가 앞에 붙은 것
   	2. 일반적으로 생성자에서 new를 통해 메모리를 할당한 것이 있으면, 소멸자에서 delete를 통해 메모리 할당을 해제함
14. 포인터
	1. this
   	1. 멤버 함수 내에서 자기 자신을 가르키는 포인터
   	2. 멤버 변수와 멤버 함수의 매개변수 이름이 겹칠 때, 함수 내에선 매개 변수가 우선시 됨
      	1. 따라서, this->param 식으로 호출하면, 멤버 변수를 의미함
   	3. self-reference
      	1. 객체 자신을 참조할 수 있는 참조자를 의미함
      	2. 즉, 멤버 함수에서 자기 자신을 반환하도록 할 수 있음
         	1. Selfref class의 멤버 함수 Adder와 ShowTwoMember가 모두 자기 자신을 반환하는 Selfref& Adder();와 같이 선언되어 있을 때, ref.Adder(1).ShowTwoMember().Adder(2).ShowTwoMember() 같이 참조값이 반환되는 것을 이용하여 코드 구성이 가능함
15. 깊은 복사, 앝은 복사
    1.  포인터(주소값)을 단순 복사하는 경우, 주소에 있는 값까지 새로 복사하는 것이 아니라, 주소값만 가져오는 얕은 복사가 일어날 수 있음
        1.  따라서, char *addr1 = addr2일 때, addr2가 delete 된 이후에 delete []addr1을 시도한다면 문제가 될 수 있음
        2.  cstring의 strcpy를 쓰면 깊은 복사가 가능
    2.  주소값뿐 아니라, 주소 내 데이터까지 복사하는 "깊은 복사"와 주소값만 가져오는 "앝은 복사"를 구분할 필요가 있음
16. friend 키워드
    1.  다른 class가 내 private에 접근이 가능하게 하는 것
        1.  양방향은 아님. A->B friend면, A는 B의 private에 접근 못함. 따로 B->A friend를 해줘야 함
17. static 키워드
	1. 전역 변수에서 선언: 선언된 파일 내에서만 참조를 허용
	2. 지역 변수에서 선언: 한번만 초기화되고, 지역을 빠져나가도 소멸되지 않음
	3. static 멤버 변수의 경우, class 당 하나씩만 생성되기 때문에 class끼리 값을 공유한다고 볼 수 있음
   	1. 클래스 변수라고도 부름
   	2. 클래스 변수의 경우 하나의 class가 소유한 것이 아니기 때문에 생성자에서 초기화하면 안됨
   	3. int classname::classvar = 0; 과 같이 class 외부에서 초기화해줌
   	4. 또한 instance가 아닌, class 이름으로도 호출이 가능함
	4. static 멤버 함수의 경우, 클래스 변수와 비슷하게 동작함
   	1. 따라서, class 내에 있는 것이 아니기 때문에, static 변수가 아닌 멤버 변수는 접근할 수 없음
18. const static 멤버
	1. 일반적인 const 멤버 변수는 member initializer에서만 초기화가 가능하지만, const static 멤버 변수의 경우 선언과 동시에 초기화가 가능
19. mutable 키워드
	1. const 함수 내에서의 값의 변경을 예외적으로 허용하는 키워드
	2. 가급적이면 사용 자제해야 함
20. inheritance
	1. 기본적으로 코드의 재사용을 위해 사용된다고 하지만, 다른 용도도 많음
	2. class는 데이터적 성격이 강한 것과 기능적 성격이 강한 class로 나뉨
   	1. 예를 들어, handler는 기능적 성격이 강함
	3. 상속도 접근 제어자에 따라서 접근 가능한 정보가 달라짐
	4. 자식 class에서 부모 class도 초기화를 진행해줘야 함
   	1. 해당 class의 멤버 변수에 접근할 수 있기 때문
   	2. 일반적으로 member initializer에서 생성자를 호출하여 초기화함
   	3. 초기화가 진행되지 않으면, 부모 class의 void 생성자가 호출됨
	5. 자식 클래스가 생성될 때, 부모 클래스의 초기화가 먼저 진행된 후 자식 클래스의 초기화가 진행됨
	6. 소멸자의 경우, 자식 클래스의 소멸자가 먼저 호출됨
   	1. 생성자와 소멸자가 stack에 쌓인 방식이라고 생각하면 편함
	7. 상속 방식
   	1. protected로 선언된 경우, public으로 상속해도 접근이 가능함
      	1. 하지만 일반적으로 외부에선  private과 같이 접근이 불가능
   	2. 상속 받을 때, 접근 제어자에 따라, 이후 자식 클래스에서는 좁은 범위로 접근이 취급됨
      	1. 예를 들어, A 부모 클래스의 public을 B가 private으로 상속받았다면, C는 A의 public도 private이라고 취급함
   	3. 하지만 대부분 public 상속이라고 생각하면 됨
21. Polymorphism
	1. 모습은 같은데, 형태는 다름
   	1. 즉, 문장은 같은데, 결과가 다름
	2. AA라는 클래스가 있을 때, AA 포인터는 AA를 간접 상속하는 자식 클래스들도 기리킬 수 있음
   	1. ex) AA * ptr1 = new BB();
   	2. ex) AA * ptr2 = new CC();
	3. BB(또는 BB를 상속한 CC)는 AA를 상속했기 때문에, BB도 AA 클래스의 일종이라고 생각하기 때문
	4. 즉, handler class에서 BB나 CC가 아닌, AA 포인터로 변수를 선언하면, 이후에 BB, CC로 자유롭게 초기화할 수 있음
   	1. python의 importlib이랑 비슷하게 사용되는 듯 함 (입력 string에 따라서 import 및 할당 클래스를 바꾸는...)
	5.  자식 클래스에서 부모 클래스의 함수를 overriding(overloading 아님)하는 경우, 할당된 class의 함수가 호출됨
    	1.  예를 들어, Parents* ptr1 = new Children();으로 할당되었을 때, ptr1->overridedFunc();이면 Children의 overridedFunc이 호출됨
    	2.  부모클래스::멤버함수의 형태로 overriding에서도 부모 클래스의 함수를 호출할 수 있음
    	3.  instance에서도 비슷하게 호출할 수 있음
        	1.  ex) SalesWorker seller("Choi", 1000, 0.1); 
                 seller.PermanentWorker::ShowSalaryInfo();
	6. 하지만, pointer 연산의 경우, 자료형으로 판단하기 때문에, 부모 클래스에 없는 함수 및 변수는 자식 클래스가 할당되었다고 해도 호출할 수 없음
   	1. ex) Parents* ptr1 = new Children();
             ptr1->OnlyChildFunc(); (=> compile error)
	7. 마찬가지로, 복사 및 할당도 제한이 있음
   	1. ex) Parents* ptr1 = new Children();
            Children* ptr2 = ptr1;  (=> compile error)
      2. 반대는 가능함
22. Virtual Function
	1. virutal 키워드를 통해 선언됨
	2. 위의 polymorphism에서의 문제를 해결하기 위함
	3. 가상함로 선언되면, 포인터의 자료형이 아닌, 실제 가리키는 객체를 기반으로 함수를 호출함
	4. 소멸자에도 virtual 키워드를 붙여줘야 제대로 소멸됨
	5. Pure Virtual Function
   	1. 함수의 몸체가 정의되지 않은 함수
   	2. ex) virtual int GetPay() const = 0 ;
   	3. Pure Virtual Function이 존재하는 object가 instance화될 경우 compile error 발생
	6. Abstract Class
   	1. 하나 이상의 멤버 함수를 Pure Virtual Function으로 선언한 class를 의미
   	2. 즉, 객체 생성이 불가능한 class
23. 멤버 함수 및 가상함수 동작 원리
    1. 객체가 생성되면, 멤버 변수는 객체 내에 존재하지만, 멤버 함수는 메모리의 별도의 공간에 위치한 후, 모든 객체가 메모리를 공유하는 방식으로 구현됨
    2. 즉, 함수의 포인터를 객체가 가지고 있다고 생각할 수 있지만, 디테일하게 기억할 필요는 없음
    3. 위와 같은 원리로 polymorphism이 구현될 수 있음
       1. 사실상 pointer를 가지고 있는 것이기 때문에, 함수 이름만 같으면 (할당될 수 있는 것이 같다면), 내용이 다르더라도 할당 가능
    4. 가상함수를 포함하는 함수의 경우, 가상 테이블 (V-Table)을 생성함
24. 다중 상속
    1.  다중 상속은 웬만하면 사용하지 않는 것이 좋음
        1.  다중 상속의 부모 클래스들에 같은 이름의 함수가 존재하는 경우 ambiguous함
        2.  어느 부모 클래스에서 호출하는지 명시해줘야 함 ex) classN::FuncN();
    2.  이를 해결하기 위한 것으로는 virtual inheritance가 있음
        1.  다중 상속에서 virtual inheritance를 하게 되면, 동시에 상속하는 부모 클래스는 1개가 됨
        2.  즉, 2번 상속되는 것은 공유하게 됨
25. 연산자 오버로딩
    1.  classN operator+(const classN $c) 와 같은 형식으로 선언되어야 함
    2.  함수와 매개변수 모두 const를 선언하는 것이 좋음
    3.  전역 함수에 friend를 선언하여 구현하는 것도 가능 (특수한 경우가 아니면, 멤버 함수로 구현하는 것이 좋음)
    4.  단항연산자
        1.   ++와 --같은 단항 연산자도 overloading 가능
        2.   
26. standard library에 대한 정보
	1. iostream
	2. cstring
	3. cstdlib
	4. ctime
