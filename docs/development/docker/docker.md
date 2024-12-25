---
layout: default
title: Docker
nav_order: 5
parent: Development
has_children: true
permalink: /docs/development/docker
---

# Docker
{: .no_toc}
Solutions and workarounds for errors encountered during the development process

1. TOC
{:toc}


## Docker의 개념
도커는 크게 이미지와 컨테이너로 구성됨\
하나의 듀얼 부팅 환경이라고 생각하면 쉬움
### Image
Ubuntu를 설치할 때, iso 확장자의 이미지 파일을 받아서, rufus를 이용하여 usb에 푼 후에 설치하는 것과 같이, 하나의 개발 환경을 가지고 있는 이미지\
이 이미지를 실제 환경으로 실행한 것이 container라고 보면 됨\
즉, 이미지를 컨테이너로 만드는 것은 위에서 말한 우분투 설치 과정과 같다고 생각하면 됨\
\
docker hub에 여러 환경에 대한 official 이미지들이 보관되어 있음\
github에서 코드를 다운받는 것과 같이, 이런 이미지 파일들을 그대로 다운 받아서 사용할 수 있음\
ex) `docker pull ubuntu:20.04`\
다운 받은 이미지 파일은 docker run 명령어를 통해 container가 됨
### Container
Container는 remote 개발 환경이라고 생각하면 편함 (Server랑 비슷)\
독립적인 개발환경이고 container를 종료하면 container에 적었던 내용은 다 사라짐 (서버가 없어지는 것과 같음)\
따라서, `-v` option을 이용하여, host pc (local pc)의 storage와 container의 storage를 연결하여, 종료하더라도 local pc에 파일이 남아있게 할 수 있음\
예를 들어, `docker run -i -v ~/Desktop/app:/app [image name]:[tag]`\
와 같이 container를 생성하면, container의 `/app` 경로에 변경하거나 만든 파일 및 폴더는 local pc의 `~/Desktop/app`에 그대로 적용됨\
따라서, Container는 말그대로 환경이기 때문에 코드같은 부분은 -v로 연결해주는 것을 추천함\
\
그리고 위에서 말한 것과 같이 독립된 환경이기 때문에, network, gui, port 등을 다 연결해주는 설정이 필요함\
이런 것들은 `-v`와 같이 이미지에서 컨테이너를 생성하는 docker run에서 옵션으로 설정해줄 수 있음\
ex) `--network`, `--port`, `-v`, etc...

### Dockerfile
Dockerfile은 이미지가 만들어진 과정을 기록한 파일이라고 볼 수 있음\
따라서, dockerfile을 실행시키면 local pc에 해당 이미지가 생김\
ex) `docker build -t [생성될 이미지 이름]:[tag] .`

dockerfile은 RUN 명령어를 실행할 때마다 github에서 commit하듯이 layer가 쌓이며(기록되며) 이미지 파일을 생성함\
따라서, RUN을 하나의 커맨드로 묶어서 작성하는 것이 이미지 파일의 크기와 효율성에 좋음\
각각의 RUN은 WORKDIR 명령어의 path에서 시작된다고 볼 수 있음\
따라서, 이전에 `RUN cd ~` 와 같이 command를 작성했더라도 다음 RUN 명령어에는 WORKDIR에서 설정했던 경로에서 command가 실행됨\
그리고 RUN 명령어는 2가지 방식으로 동작시킬 수 있음
1. sh 형식, 
2. exec 형식

sh는 일반적으로 CLI에서 실행하는 것과 같음. 즉, `/bin/bash`나 `/bin/sh`로 command가 실행되는 것과 같음. ex) `RUN python app.py`\
반대로 json 형식으로 직접 sh를 설정하는 것도 가능함 ex) `RUN ['/bin/bash', '-c', 'python', 'app.py']`\
dockerfile에서 기본 shell은 `/bin/sh`이므로 SHELL 명령어를 통해 기본 sh를 bash로 바꿔줌