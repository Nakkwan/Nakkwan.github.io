---
layout: default
title: Docker Error
nav_order: 1
parent: Error Fix
grand_parent: Et cetera
permalink: /docs/etc/error/docker
---

# Docker
{: .no_toc}
Error occurred in docker

1. TOC
{:toc}

## Dockerfile
### Dockerfile Exit Code Error
Dockerfile에서 틀린 것이 없는데, exit code error가 나는 경우, shell의 문제일 수 있음 <br>
dockerfile에서 build 시 기본 shell은 `/bin/sh`로 설정됨 <br>
따라서, 정상적인 RUN command에 에러가 발생한다면, `SHELL [’/bin/bash’, ‘-c’]`을 사용하여 기본 shell을 바꿔주자! <br>
<details open markdown="block">
<summary>shell 추가 정보</summary>
<ul><li>
Shell은 운영 체제 속 내용물에 접근할 수 있는 기능을 제공하는 프로그램</li><li>
즉, 운영체제와 상호작용을 할 수 있는 프로그램이라고 할 수 있음</li><li>
Script란 interpreter 방식으로 동작하는 것을 의미</li><li>
따라서, <b>shell script</b>는 interpreter 방식으로 동작하며 운영체제와 상호작용하는 프로그램</li> <br><li>
<code>/bin/sh</code>의 경우, dash shell를 사용하고 <code>/bin/bash</code>는 bash shell을 사용함</li><li>
dash는 더 빠르고 적은 기능을 제공, bash는 history 등 상대적으로 많은 기능을 제공함</li><li>
기본 shell을 dash를 사용하는 경우도 있기 때문에 대부분 쉘 스크립트에서 맨 윗줄에 <code>#!/bin/bash</code>와 같은 라인을 추가하여 어떤 shell을 쓸건지 정해줌
</li></ul>
</details>

### Dockerfile Run Command
dockerfile은 RUN 명령어를 실행할 때마다 github에서 commit하듯이 layer가 쌓이며(기록되며) 이미지 파일을 생성함 <br>
따라서, RUN을 하나의 커맨드로 묶어서 작성하는 것이 이미지 파일의 크기와 효율성에 좋음 <br>
각각의 RUN은 WORKDIR 명령어의 path에서 시작된다고 볼 수 있음 <br>
따라서, 이전에 RUN cd ~ 와 같이 command를 작성했더라도 다음 **RUN 명령어에는 WORKDIR에서 설정했던 경로에서 command가 실행**됨 <br>
$$\rightarrow$$ 이 부분을 간과해서 동작에 오류가 났었던 경우가 있음

<details open markdown="block">
<summary>RUN 명령어는 2가지 방식으로 동작시킬 수 있음</summary>
1. sh 형식, 
2. exec 형식

sh는 일반적으로 CLI에서 실행하는 것과 같음. 즉, `/bin/bash`나 `/bin/sh`로 command가 실행되는 것과 같음. <br>
ex) `RUN python app.py` <br>
반대로 json 형식으로 직접 sh를 설정하는 것도 가능함  <br>
ex) `RUN ['/bin/bash', '-c', 'python', 'app.py']` <br>
Dockerfile에서 기본 shell은 `/bin/sh`이므로 SHELL 명령어를 통해 기본 sh를 bash로 바꿔줌
</details>