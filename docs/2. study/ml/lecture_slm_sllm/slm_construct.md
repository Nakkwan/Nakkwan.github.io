---
layout: default
title: Construct sLM
nav_order: "2025.03.11"
parent: Lecture sLM, sLLM
grand_parent: ML
ancestor: Study
permalink: /docs/study/ml/lecture_slm_sllm/slm_construct
math: katex
---

# **Basic of construct sLM**
{: .no_toc}

1. TOC
{:toc}

### AI Framework & Tools
##### Huggingface
AI와 관련하여 다양한 tool과 dataset, library를 제공하는 platform

##### Ollama
Opensource LLM을 local에서 쉽게 실행할 수 있도록 함 <br>
모델 weight, setting, dataset을 하나의 package로 묶어, modelfile로 관리 <br>

##### LangChain
sLM을 구축한다는 것은 모델 구축, 프롬프트, UI 등의 작업을 하는데, LangChain platform을 통해 이런 요소들을 묶을 수 있음 <br>

##### RAG
RAG(Retrieval-Augmented Generation)는 대규모 언어 모델의 출력을 최적화하여 응답을 생성하기 전에 학습 데이터 소스 외부의 신뢰할 수 있는 지식 베이스를 참조하도록 하는 프로세스
