# 📌 New Notebook Overview

<aside>

현재 비디오 feature 추출 파이프라인 돌고 있는 중 

</aside>

# 🚗 Nexar Dashcam Crash Prediction - 비디오 Feature 추출 파이프라인

## 1. 데이터 구조 확인 및 사전 준비

- 비디오 파일 이름은 **5자리 zero-padding** (`02059.mp4` 등)

## 2. 데이터 로딩 및 전처리

## 3. 주요 기능 함수 정의

### 1️⃣ 비디오 주요 프레임 추출

- 짧은 영상 ➔ 균등 간격 프레임 추출
- 긴 영상 ➔ 후반부(사고 직전)에 집중하여 프레임 추출

### 2️⃣ 비디오 데이터 증강 파이프라인

<aside>

`class RandomHorizontalFlip, ColorJitter, AddFog, AddRain, RandomNoise, RandomOcclusion, Compose, ToTensor
def get_video_transforms()`

</aside>

- 다양한 증강 기법(Random Flip, Rain, Fog, Noise 등) 적용
- **train/val** 모드에 따라 증강 여부 다르게 설정

### 3️⃣ Optical Flow 계산

<aside>

`def compute_optical_flow(frames, skip_frames=1):`

</aside>

- Farneback 방법으로 프레임 간 움직임(모션) 추출

### 4️⃣ CNN + Optical Flow 기반 Feature 추출

- InceptionV3로 spatial feature 추출
- Optical flow로 motion feature 추출
- 둘을 합쳐서 최종 feature (1280 + 1차원)

### ✨ 중요 포인트

| 구분 | Optical Flow | InceptionV3 Feature |
| --- | --- | --- |
| 입력 데이터 | 프레임들 사이의 움직임(변화량) | 개별 프레임 이미지 자체 |
| 출력 데이터 | 움직임 벡터 or 이동 크기(norm) | 1280-dim or 2048-dim CNN Feature Vector |
| 쓰임새 | 이동 패턴 분석 (ex: 흔들림, 충돌 감지) | 이미지 시각 특징 분석 (ex: 도로, 차량, 상황 파악) |

✅ **즉, 같은 프레임들로 Optical Flow와 CNN Feature를 각각 따로 뽑을 수 있다**

### 🔑 지금까지 한 것

- 비디오(mp4) → 주요 프레임 추출
- 주요 프레임 → 증강 및 전처리
- 주요 프레임 → CNN (InceptionV3) feature 추출 + Optical Flow 추출
- **최종 feature → (1281차원) numpy array로 저장**

## 4. 전체 Feature 추출 루프 (with tqdm)

- tqdm으로 진행률 확인
- 비디오가 없는 경우 안전하게 continue
- feature 추출 완료 후 numpy array로 변환

---

## 💡 향후 방안

단일 feature vector만 있으니 간단한 MLP나 FCN 분류용으로 적합 ⇒ 아래 향후 방안의 구조와 맞지 않아, 넣기 좋은 형식으로 다시 다듬어야 함…😭

<aside>

`비디오
│
├─ RGB 프레임 → Spatial Transformer → Temporal Transformer
│
├─ Optical Flow 프레임 → Optical Stream Transformer (or CNN)
│
└─ 두 feature 합치기 → 최종 예측`

</aside>

**👉 2-Stream 구조 (RGB + Optical Flow 따로 입력해서 합치기)**

- "RGB는 사고 전 장면 모양"
- "Optical Flow는 움직임 빠르기와 방향"

# LSTM 모델도 추가할 수 있나?

✅ **완전 가능!**

**구조 예시:**

- Optical Flow 프레임을 CNN (ex: ResNet-18)으로 특징 추출하고,
- 그걸 **LSTM**에 넣어서 시간 순서에 따라 움직임 변화를 모델링할 수 있어.

즉, Optical Flow 전용 스트림에 Transformer 대신 **CNN + LSTM 조합**을 써도 전혀 문제 없어.