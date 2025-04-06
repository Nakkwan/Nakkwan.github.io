---
layout: default
title: Lecture sLM, sLLM
nav_order: "2025.03.10"
parent: ML
grand_parent: Study
permalink: /docs/study/ml/lecture_slm_sllm/
math: katex
---

# **Lecture sLM, sLLM**
{: .no_toc}


<details close markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

LLM은 수천억에서 수조개의 parameter를 가지고 있기 때문에, 이를 학습하기 위해선 많은 resource가 필요하다. 따라서, 빠른 모델 훈련과 배포를 위해 sLM(small Language Model), sLLM(smaller Large Language Model)의 필요성이 대두된다. <br>

sLLM은 일반적으로 10B 이하의 parameter를 가지고, sLM은 1B 이하의 parameter를 가진다. <br>

## **LM Model 평가 방법**
LLM에서 평가는 
1. LLM 모델 자체에 대한 평가 
   1. 기본 LLM에 대한 전반적인 성능 평가
   2. GT와 비교하여 점수를 매기는 리더보드 활용
      1. ex) HellaSwag, TruthfulQA, MMLU, ARC, ...
2. LLM 기반 시스템에 대한 평가
   1. 시스템에서 제어하는 구성 요소에 대한 평가
   2. 프롬프트 및 컨텍스트에 따른 결과를 나타냄
   3. Extracting structured information: LLM이 정보를 얼마나 잘 추출하는지
   4. QA: 사용자의 질문에 얼마나 잘 답변하는지
   5. RAG: 검색된 문서와의 관련성
      1. ex) Diversity, User feedback, GT-based Metrics, Answer Relevance, QA Correction, Hallucinations, Toxicity, ...

로 나뉜다.

### **Coding Tasks**
Coding에 대한 평가 방법은 HamanEval, MBPP(Mostly Basic Python Programming)로 나눌 수 있다.
1. HamanEval: LLM의 성능을 측정하기 위해, 프로그래밍 과제 및 문제를 데이터셋으로 가지고 있는 평가 도구
   1. 실질적인 문제들로 평가하며, LLM에서 생성된 python 코드가 문제를 얼마나 pass하는지를 측정함
2. MBPP: python 프로그램을 합성하는 능력을 측정
   1. 974개의 프로그램이 포함된 데이터셋
   2. 프로그래밍 표준 문법 및 라이브러리를 다루는 능력 및 품질을 평가

<figure>
    <center><img src="/assets/images/study/ml/sllm_slm_figure1.jpg" width="90%" alt="Figure 1"></center>
	<center><figcaption><b>[Figure 1]</b></figcaption></center>
</figure>

##### **HumanEval**
<figure>
    <center><img src="/assets/images/study/ml/sllm_slm_figure2.jpg" width="90%" alt="Figure 2"></center>
	<center><figcaption><b>[Figure 2]</b></figcaption></center>
</figure>

아래와 같이, 문제에 대한 prompt와 정답 코드, output check를 위한 assert를 포함하고 있다. 

<figure>
    <center><img src="/assets/images/study/ml/sllm_slm_figure3.jpg" width="90%" alt="Figure 3"></center>
	<center><figcaption><b>[Figure 3]</b></figcaption></center>
</figure>


##### **MBPP**
<figure>
    <center><img src="/assets/images/study/ml/sllm_slm_figure4.jpg" width="90%" alt="Figure 4"></center>
	<center><figcaption><b>[Figure 4]</b></figcaption></center>
</figure>

MBPP는 python에 특화된 metric으로, 코드 n 번 생성에 대한 실행 횟수, 코드의 품질 및 정확성을 테스트할 수 있다.

<figure>
    <center><img src="/assets/images/study/ml/sllm_slm_figure5.jpg" width="90%" alt="Figure 5"></center>
	<center><figcaption><b>[Figure 5]</b></figcaption></center>
</figure>

### **Chatbot Assistance**
Chatbot Assistance에 대한 평가 방법도 2가지로 나눌 수 있다.
1. Chatbot Arena: 익명의 모델 n개를 사람이 직접 사용해보고 투표를 진행하는 방식
2. MT Bench: 대화나 지침 준수에 대한 능력을 평가함
   1. n번 질문을 했을 때, 어떤 성능 및 특성을 가지는지 확인을 하는 특성
   2. Writing, Extraction, Reasoning, Math, Coding 등 사람이 많이 사용하는 8개의 category들에 대해 대답을 잘하는지에 대한 평가를 수행

##### **Chatbot Arena**
[LMSYS](https://lmarena.ai/)에서 운영하고 있음.
A와 B 모델에 대해 두 LLM 모델이 답변을 하고, 답변에 대해 투표

<figure>
    <center><img src="/assets/images/study/ml/sllm_slm_figure6.jpg" width="90%" alt="Figure 6"></center>
	<center><figcaption><b>[Figure 6]</b></figcaption></center>
</figure>

아래와 같이, LLM에 대한 순위 및 여러 task에 대한 성능도 시각적으로 표기해준다.

<figure>
    <center><img src="/assets/images/study/ml/sllm_slm_figure7.jpg" width="90%" alt="Figure 7"></center>
	<center><figcaption><b>[Figure 7]</b></figcaption></center>
</figure>

##### **MT Bench**
모델의 이해력 및 추리 능력, 대화형 응답에 대한 능력을 종합적으로 평가 <br>
질문의 category, 질문, 답변에 대한 refernce로 데이터셋이 구성되며, 여러 범주에 대한 성능을 객관적으로 평가할 수 있다.

n번 질문들 통해, 공통적인 답변을 하는지 판단하고, GPT-4를 통해 답변의 정확성을 판단함

### **Reasoning**
주어진 정보를 바탕으로 논리적인 답변을 만들어내는 것을 추론이라고 함
1. **ARC Benchmark**: 과학적 지식 및 문제 추론에 대한 평가
2. **HellaSwag**: 일반적인 상식 추론을 평가하는데 사용
3. **MMLU**: 언어 자체에 대한 이해 
4. **TriviaQA**: 퀴즈에 대한 정답을 평가
5. **WinoGrande**: 문제가 주어졌을 때, 논리적으로 추론을 잘하는가? 문맥에 대한 이해 능력
6. **GSM8k**: 수학 문제에 대한 추론 능력 평가

## **LLM 평가 툴**
1. [DeepEval](https://docs.confident-ai.com/)
2. [MLFlow](https://mlflow.org/)
3. [RAGAs](https://docs.ragas.io/en/stable/)
4. [Deepchecks](https://www.deepchecks.com/)
5. [Phoenix](https://github.com/Arize-ai/phoenix)