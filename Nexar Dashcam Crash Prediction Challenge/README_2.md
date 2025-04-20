# 💥Nexar DCP Challenge - Part 2

<aside>

🔑 참고자료 

https://openaccess.thecvf.com/content/WACV2025W/HAVI/papers/Kumamoto_AAT-DA_Accident_Anticipation_Transformer_with_Driver_Attention_WACVW_2025_paper.pdf

</aside>

<aside>

📌 현재 작성하고 있는 코드 

https://www.kaggle.com/code/haileyyewon/nexar-dcp-challenge-02

</aside>

# 1️⃣ **AAT-DA: Accident Anticipation Transformer with Driver Attention 논문**

## 📂 배경 및 문제의식

- 교통사고 예측은 자율주행 안전성 확보 및 사망자 감소에 중요
- 대부분의 기존 모델은 RNN 기반이나, 장기 시퀀스 처리의 한계 존재
- 최근 **Transformer 기반 모델들**이 영상 처리에 뛰어난 성능을 보이지만,
    - 복잡한 교통 환경에서는 RNN보다 예측력이 떨어짐
    - 실시간 예측이 어렵고, 사고 관련 객체에 집중할 수 없음
- 사고는 짧고 복잡한 순간에 일어나므로, 효율적인 객체 주의 매커니즘과 설명 가능성이 중요

---

## ⛑️ Transformer는 영상 처리에는 강하지만, 교통사고 예측에선 왜 약할까?

### ✅ Transformer가 영상처리에 강한 이유

- **병렬적으로 전체 프레임을 다 본 후** → 관계 분석 가능 (예: frame 1 ↔ frame 19 간 유사도 파악)
- **Self-Attention** 덕분에 장기 의존성도 문제없다
- 이미지 패치나 프레임 간 관계를 잘 포착한다

## 🛑 그런데 사고 예측에서는?

1. 데이터가 적음 (실제 사고 영상은 수가 적고 희귀하다) 
2. 사고는 대부분 짧은 순간에 발생 (초당 10~20 프레임만 존재)
3. 사고 예측 모델은 사고가 난 뒤가 아니라, 과거부터 현재까지만 보고 ‘앞으로’ 사고가 날지를 실시간으로 판단해야 함  
    1. 그러나 Transformer 구조는 과거/현재/미래를 다 본 후 사고 여부를 판단하는 ‘사후 판단’의 특징이 강함 
4. 사고에 관련 없는 정보(배경, 무관한 객체)가 많음 

## 🤔 Transformer의 단점

- 전체 시퀀스를 한꺼번에 입력해야 해서:
    - 학습 시 미래 프레임까지 포함되면 실시간 예측이 불가능
    - 사고 전 조짐이 적은 경우, 중요한 신호가 묻히거나 희석됨
- 모든 객체에 Attention을 고르게 분배하면, 사고 관련 중요한 객체를 놓치기 쉬움
- 데이터 수 부족 시, Transformer는 오히려 과적합(overfitting) 하기 쉬움

> **📌 따라서 사고 순간처럼 데이터가 적고 빠르게 판단해야 할 때는 오히려 Transformer 모델이 부적합할 수 있다.**
> 

---

## 🎯 연구 목표

> **Transformer 기반 구조**로 교통사고를 정확하고 빠르게 예측하되, 객체 중요도와 운전자 주의를 함께 반영한 설명 가능한 모델을 만들자
> 

# 🧠 제안 모델: AAT-DA

## 1️⃣ 핵심 개념

- **Spatial Transformer + Temporal Transformer** 구조를 이중으로 활용
- 운전자 주의(Driver Attention) 정보를 이용하여 사고 관련 객체에 집중
- 객체 간의 위치 기반 중요도(positional weighting)도 함께 고려
- **Attention Matrix**를 통해 사고 위험이 높은 객체 시각화

## 2️⃣ 모델 구조 요약

### 1) 입력 및 특징 추출 (Object Detection + Feature Extraction)

- 프레임에서 N개의 객체 검출 (차, 사람, 자전거 등)
    - **Faster R-CNN**
        
        2단계 객체 감지 알고리즘(two-stage detection)으로, 이미지 내 객체의 위치와 종류를 정확하게 파악하는 데 사용
        
        1️⃣ 어디에 객체가 있을 지 영역을 먼저 뽑고 난 뒤,
        
        2️⃣ 그 영역을 분석해서 어떤 객체인지 분류
        
    - **Cascade R-CNN**
        
        Faster R-CNN의 후속 모델로, 객체 감지 성능을 더욱 향상시키기 위해 제안된 **다단계 객체 감지 아키텍처**
        
        📌 마찬가지로 2단계(two-stage) 객체 감지 방식을 따르지만, **여러 개의 검출기(detector)를 순차적으로 연결**하여 각 단계마다 객체 후보 영역을 정제하고 분류 정확도를 높이는 특징을 가진다.
        
- **VGG-16이**라는 CNN을 써서 각 객체와 전체 이미지에서 **4096차원 벡터** 추출
    - 데이터의 특징을 4096개의 독립적인 수치로 표현하는 방식

### 2) Spatial Attention Module

- 객체 간 거리가 가까울수록 사고 위험이 크다고 판단해 가중치 부여 → **Position Weighting**
- 운전자 주의 예측 모델(Gate-DAP) 기반의 주의도 Attention 적용 → **Driver Attention Weighting**
    - 운전자가 어느 부분을 보고 있는지 예측
    - **Gate-DAP 모델**로 **Attention Map(확률 히트맵)**을 만들고, **객체와 겹치는 부분(객체의 위치 + 운전자 주의 영역)이 클수록 중요하다고 판단**
- 위 두 가중치를 종합하여 αᵢ 계산한 후, 각각의 객체 피처(Feature)에 곱해서 강조

### 3) Spatial Transformer

> 현재 프레임 안에서 객체들끼리 어떤 관계가 있는지 알아보자!
> 

### **🎯 목적**

- 위험 상황은 객체들 사이의 상호작용에서 발생 (예: 사람이 길을 건너는데 차가 다가옴)
- 즉, 단독 객체보다 **“A가 B에 가까워진다”와 같은 관계**가 중요하다

### ⚙️ 구조

- 이미지 + 객체 특징을 Transformer Encoder에 입력
- **객체 간 상호작용을 모델링**하는 Object Self-Attention Layer 포함

### 4) Temporal Transformer

> 시간 흐름에 따라 사고가 일어날 것 같은지 파악하자
> 
- 현재 및 과거 시점의 Spatial Feature를 입력으로 사고 확률 예측
- **미래 프레임 없이도 현재까지 정보를 바탕**으로 예측 가능 → 실시간 예측 가능

## 3️⃣ 학습 방식

- 프레임 단위 사고 예측을 위한 **Cross Entropy Loss + Temporal Weighting** 적용
- 사고 지점(t)에 가까울수록 가중치 증가 → 빠른 예측 유도
- 학습에는 Driveer Attention 예측은 별도로 학습된 모델 사용 → 멀티태스크 리스크 회피

## 4️⃣ 성능 평가

### (1) 사용 데이터셋

- **DAD (Taiwan Dashcam)**: 사고/비사고 영상 1750개 (FPS 20, 사고는 90프레임)
- **CCD (Chinese Crash Dataset)**: 총 4500개 영상 (FPS 10, 사고 지점은 마지막 2초 내)

### (2) 주요 지표

- **AP (Average Precision)**: 정확도 기반 평가
- **mTTA (mean Time-To-Accident)**: 예측 후 사고까지 남은 평균 시간

### (3) 결과 요약

| 모델  | DAP AP | DAD mTTA | CCD AP  | CCD mTTA |
| --- | --- | --- | --- | --- |
| AAT-DA (Ours) | 64.0 | 2.87 | 99.4 | 4.88 |
| 기존 최고 성능 | 56.1 (DSTA) | 3.66 (DSTA) | 99.6 (DSTA) | 4.87 (DSTA) |
- 📌 **Transformer 기반이 AP는 우수하나, mTTA는 RNN보다 낮을 수 있다 (짧은 시퀀스에선 RNN이 더 빠르게 예측 가능)**
    - **Transformer의 Attention 구조**는 모든 입력을 비교해서 판단하는 구조로, 정확하게는 맞추지만, 대신 타이밍이 늦을 수 있다.
    - 반면, **RNN은 순차적으로 정보를 쌓아가며 예측**하기 때문에, 점점 사고 확률이 올라가 사고가 가까워지면 더 빠른 감지가 가능하다. (빠르게 감지하지만, 정확도는 떨어질 수 있음)

## 5️⃣ 장점 및 한계

### (1) 장점

- Transformer 기반임에도 설명 가능성 확보
    - Transfomer의 Attention 구조는 복잡하고, 모든 객체끼리 관계를 계산하다 보니 원래는 설명하기 어려움
    - 그러나 **AAT-DA 모델**은 **Driver Attention + Position Weighting을 이용해서 사고와 관련 있는 객체에만 높은 attention을 주도록 유도**
- 운전자 주의 정보를 효과적으로 활용
- 두 데이터셋에서 SOTA 성능 달성
    - SOTA 성능 (State Of The Art = 최고성능) → 기존 모델보다 정확도(AP)가 훨씬 향상
    - 즉, 어떤 문제를 푸는 다양한 모델들 중에서, 가장 좋은 평가 결과를 낸 모델

### (2) 한계

- 실시간 처리 어려움 (탐지 및 Attention 계산 지연)
- 교통 신호 등 도로 구조 정보 미반영

## 6️⃣ 향후 연구 방향

- 도로 구조 및 교통 신호 정보 통합
- 실시간 처리 최적화
- 다양한 객체 및 행동 분류 강화