#### 프로젝트명

GPU의 구조를 고려한 거대 언어 모델 (LLM) 추론 효율성 향상 및 가속화
#### 작성일

2023년10월08일
#### 팀번호 및 팀명

01. Optimus GPrime
#### 팀구성원들의
역할분담

황재은(2176427) 팀장,	관련 선행 연구 분석, 프로파일링			
정은비(2173109) 팀원,	관련 선행 연구 분석, 백그라운드 서치			
최이경(2276327) 팀원,	관련 선행 연구 분석, 라이브러리 및 프레임워크 조사 			
#### 팀지도교수

윤명국 교수님
#### 멘토소속회사

ETRI
#### 멘토성함, 직함

전원 연구원
#### 엘리베이터 스피치 
문장

거대 언어 모델 (Large Language Model, 이하 LLM)의 수요가 증가함에 따라 추론 비용의 감소는 더욱 중요해지고 있다. LLM의 추론을 위해서는 다량의 GPU가 사용되는데, GPU의 구조를 고려한 LLM 최적화 기법은 널리 알려지지 않았다. 따라서 본 팀에서는 GPU의 구조를 고려한 LLM 최적화 방법을 제시하여 LLM의 추론을 가속하고, 이와 더불어 데이터 센터의 에너지 소모 절감에도 기여하고자 한다.
#### 문제의 정의

ChatGPT[1]와 같은 LLM의 수요는 점차 증가하고 있다. LLM은 수억 개의 가중치를 사용함에 따라 긴 시간의 연산을 요구하며, 이러한 연산을 빠르게 처리하기 위해 다량의 GPU를 필요로 한다. 이러한 GPU 서버를 모아둔 데이터 센터에서는 LLM 수요가 증가함에 따라 많은 양의 에너지를 소모하고 있다[12]. 따라서 GPU을 LLM 추론에 최적화하여 추론 비용을 감소시킴으로써, GPU의 에너지 소모를 감소시키는 것이 매우 중요한 문제로 대두되고 있다. 
#### 관련연구/서비스/
시스템조사결과 
및 한계점

[관련 연구]
NVIDIA에서 LLM의 추론을 가속화하기 위한 라이브러리인 FasterTransformer[5]를 구현했다. FasterTransformer는 다양한 가속 기법을 제공하는데, KV Cache의 효율적인 사용을 위한 memory reservation을 대표적인 예로 들 수 있다. 기존의 LLM 추론 과정에서는, 하나의 토큰을 생성할 때마다 해당하는 KV Cache 공간을 할당하는 방식을 사용했다. 하지만, FasterTransformer는 일정 개수의 토큰에 대한 메모리 공간을 미리 reservation 함으로써, 기존의 on-demand 방식의 memory allocation으로 발생한 overhead를 감소시킨다.

[한계점] 
위와 같은 memory reservation 기법은 주로 LLM이 생성할 수 있는 최대 토큰 개수에 비례하는 메모리 공간을 미리 reservation 한다. 따라서, 경우에 따라 불필요한 메모리 공간까지 reservation을 하여 상당한 양의 메모리 낭비를 초래할 수 있다는 문제점이 있다[3]. 또한, 메모리의 사용량은 GPU가 한 번에 처리할 수 있는 request의 수를 제한하는 요소 중 하나이므로, 메모리 낭비는 결국 throughput 감소의 문제로 이어질 수 있다.
#### 제안내용

LLM은 KV Cache의 메모리 사용이 LLM 성능의 병목으로 작용할 수 있다. 따라서, KV Cache를 효율적으로 사용함으로써 GPU의 LLM 추론을 가속화 할 방법을 제안하고자 한다.
#### 구현방법

프레임워크로 PyTorch[6]를 선택하여 LLM 추론 가속을 위한 최적화 기법을 구현한다.
성능 평가를 위해 NVIDIA에서 제공하는 Nsight Systems[9]와 Nsight Compute[10] 프로파일링 툴을 활용한다. 
NVIDIA GeForce RTX 3090 서버 2대를 실험 환경으로 사용하여, LLM의 성능을 평가한다. 
#### 사용할 세부기술

사용 언어: CUDA, python, C++
프레임워크: PyTorch[6]
오픈소스 라이브러리: cuBLAS[7], Huggingface Transformers[8], FasterTransformer[5]
#### 기대효과 및 의의

KV Cache의 최적화를 통해 한정된 GPU의 메모리 공간을 효율적으로 활용함으로써, 같은 시간 안에 더 많은 request를 처리할 수 있다. 이로써 throughput이 증가하여, LLM의 추론 성능을 향상시킬 수 있다. 

더불어 LLM의 폭발적인 수요로 데이터 센터 내의 GPU 사용량이 급격히 증가함에 따라, 데이터 센터의 전력 소모가 심각한 문제로 주목받고 있다. 뿐만 아니라 데이터 센터의 온도 조절을 위해 사용되는 냉각수 소모 또한 중요한 이슈로 다루어진다. 본 연구를 통해 LLM의 throughput을 증가시킴으로써 이와 같은 데이터 센터의 전력 및 냉각수 소모 문제 해결에 기여할 수 있을 것으로 기대된다.

이러한 기대효과를 통해 본 연구는 효율적인 GPU 자원 활용과 지속 가능한 기술 발전에 기여하며, LLM을 활용한 다양한 응용 분야에서의 성능 향상에도 긍정적인 영향을 미칠 것으로 보인다.
#### 9월 진척사항

LLM에서 사용하는 Transformer[1] 모델 구조에 관한 전반적인 지식을 습득하였다. 
Transformer를 활용한 inference (추론) 과정은 attention 연산으로 이루어져 있다. attention 연산은 Query * Key 으로 구한 attention score에 Value를 곱함으로써, Query에 해당하는 단어가 다른 단어들에 주목할 확률을 구하는 것이다. chatGPT를 예로 들었을 때, 우리가 prompt에 문장을 입력하여 request를 주면, GPT model은 해당 문장을 query로 사용하여 단어들 간의 attention score을 계산한 뒤 다음 단어를 예측한다. 예측된 단어는 다시 Query로 사용되고, 이러한 attention 연산을 반복함으로써 request에 대한 output 문장을 생성한다. 
#### 10월 진척예정

LLM 중 하나인 LLaMA 7B[4]를 Nsight Systems[9] 및 Compute[10]를 통해 프로파일링 하여 Transformer의 연산 과정을 파악하고, 병목이 되는 부분을 확인하였다. 
KV Cache란, 가중치를 통해서 구한 Key, Value 값을 저장해두고 Query가 주어질 때 다시 계산을 하지 않고 저장된 캐시값을 사용하는 것으로 빠른 추론을 위해 고안된 방법이다. 하지만, 이는 GPU 메모리에서 많은 공간을 차지하기 때문에 batch size를 제한하게 되어 throughput의 증대를 방해한다. 이는 throughput 향상을 통해 latency 를 줄이는 GPU 구조에 적합하지 않을 가능성이 있다.  따라서 10월에는 KV Cache가 batch size와 throughput의 감소에 영향을 미치는지 실험을 통해 검증할 예정이다.
#### 11월 진척예정

KV Cache가 LLM 성능의 병목이 되는 것이 확인되면, KV Cache의 효율적인 메모리 사용 방법에 관한 선행 연구를 분석하여, GPU 구조에 적합한 KV Cache 활용 기법을 제안할 예정이다. 
#### 12월1일 데모 
시나리오

NVIDIA에서는 Unified Memory 기술을 제공하는데, Unified Memory란 CPU와 GPU 두 장치 모두 하나의 메모리 주소 공간을 사용하는 것을 의미한다[11]. 이를 통해 GPU 연산에서도 CPU 메모리 공간을 활용할 수 있다. 따라서, LLM의 추론에 Unified Memory 기술을 사용할 경우, KV Cache의 저장에 활용할 수 있는 메모리 공간이 증가하여 throughput 역시 향상시킬 수 있을 것이라 가설을 세우고 이를 검증하고자 한다. 
#### 기타

[References]
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
[2] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901.
[3] Yu, G. I., Jeong, J. S., Kim, G. W., Kim, S., & Chun, B. G. (2022). Orca: A distributed serving system for {Transformer-Based} generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22) (pp. 521-538).
[4] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.
[5] https://github.com/NVIDIA/FasterTransformer
[6] https://github.com/pytorch/pytorch
[7] https://developer.nvidia.com/cublas
[8] https://github.com/huggingface/transformers
[9] https://developer.nvidia.com/nsight-systems
[10] https://developer.nvidia.com/nsight-compute
[11] https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[12] https://cse.engin.umich.edu/stories/power-hungry-ai-researchers-evaluate-energy-consumption-across-models
