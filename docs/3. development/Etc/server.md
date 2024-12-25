---
layout: default
title: Server Setting
nav_order: "2023.12.15"
parent: Etc
grand_parent: Development
permalink: /docs/development/etc/server_setting_2023_12_15
---

# Server Setting
{: .no_toc}

Table of Contents
{: .text-delta}
1. TOC
{:toc}

## Ubuntu 설정
### 계정 관련
- adduser [계정이름] <br>
  - 계정 생성 및 home directory 설정 등을 한번에 할 수 있음

- useradd [계정 이름] <br>
  - 계정만 생성됨. 추후 추가 설정해줘야함

### 그룹 관련
- 현재 모든 그룹 확인: `cat /etc/group` <br>
- 그룹에 유저 추가: `usermod -a -G [group] [user id]` <br>
- 현재 내가 속한 그룹 확인: `groups` <br>
- 그룹 생성: `groupadd [group]` <br>

## Samba 설정
Samba는 윈도우에서 파일 탐색기로 서버에 접근하기 위한 툴 <br>
그룹에 추가되어 있어야, 접근이 가능하도록 설정할 수 있음

- Samba에 user 추가: sudo smbpasswd -a <user id> <br>
- Samba configuration 설정 : sudo vim /etc/samba/smb.conf <br>
- Samba 설정 변경 후, service restart: sudo service smbd restart <br>
- 현재 samba user list 확인: sudo pdbedit -L -v <br>
- Window에서 접속: 파일 탐색기에서 \\ip adress\user <br>
  - ex) <code>\\192.168.0.2\username</code>

## Anaconda 설정
Server는 보통 많은 사람들이 같이 쓰기 때문에 개인 home directory에 깔지 않고, root directory에 보통 깔아줌 <br>
Root director에는 보통, `home, bin, root, opt, usr, boot, etc` 등의 폴더가 있음 <br>
- Home: server user의 home director가 있는 곳. 바탕화면 (Desktop)도 이 폴더 안에 있음 <br>
- Bin: 보통 terminal과 관련된 명령어 설정이 포함되어 있음 <br>
- Root: root user의 개인 home directory <br>
- Lib: 각종 library 파일들이 존재하는 directory <br>
- Usr: 시스템이 아닌 일반 user들이 주로 사용하는 directory <br>
  - 보통 아나콘다같은 공용 툴들을 여기다가 깖 <br>
- Boot: GRUD와 같은 bootloader에 관한 파일들을 포함 <br>
- Etc: 시스템 대부분의 모든 설정 파일이 존재하는 directory

아나콘다를 새로 설치할 때는 앞과 같이, usr path에다가 깔아주면 되고, 해당 path에 맞게, bashrc 파일을 수정 <br>
Bashrc는 terminal이 새로 켜질 때, 먼저 실행되는 파일 (각 user의 환경 설정 등이 적혀있음) <br>
Bashrc 수정: `sudo vim ~/.bashrc` <br>
수정된 bashrc 적용(수정하고 난 후 써줘야 적용됨): `source ~/.bashrc` <br>
새로 user를 추가했을 시, bashrc 파일을 수정해줘야 함

ex) <code>export PATH=/usr/anaconda/bin:/usr/anaconda3/condabin:$PATH</code>

## Remote GUI 설정
X11 forward를 진행해줘야 함

{: .note-title}
> TODO: X11 forwarding 관련 업로드
> TODO: container -> Server X11 연결 관련 업로드

## 설치했던 툴
### glances
터미널에서 server의 resource를 모니터링할 수 있음 <br>
실시간으로 update되기 때문에 gpu 상태나, 실행중인 프로세스를 볼 때, nvidia-smi보다 편했음

### Cockpit
서버에 관한 웹 인터페이스 <br>
CPU, GPU 사용량 및 storage, network를 구성, log를 검사 등과 같은 server의 상태를 web에서 9090 (default) port를 통해 볼 수 있음