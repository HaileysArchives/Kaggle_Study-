# ✨New Notebook (Other Thesis)

| 단계 | 내용 |
| --- | --- |
| 1. 데이터 준비 | 비디오 경로 + 라벨 파일 읽기 |
| 2. 프레임 추출 | 비디오에서 필요한 프레임만 추출 |
| 3. Optical Flow 계산 | Farneback으로 연속 프레임 Optical Flow 계산 |
| 4. 특징 벡터 생성 | Optical Flow 크기(norm) 평균값 등으로 특징 벡터 만들기 |
| 5. Dataset 클래스 작성 | 학습용 Dataset 구성 (PyTorch Dataset 상속) |
| 6. 간단한 모델 작성 | (MLP 또는 Transformer) |
| 7. 학습 / 평가 루프 작성 | PyTorch 학습 루프 만들기  |

# ✅ 데이터 준비

1. 필요한 패키지 로드 (cv2, torch, numpy, sklearn 등)
2. 경고(warning) 무시 설정
3. GPU 사용 여부 체크 (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`)
4. 학습 데이터(train.csv), 테스트 데이터(test.csv), 제출 템플릿(sample_submission.csv) 불러오기
5. 데이터프레임 `head()`로 내용 확인
6. 결측값(NaN) 처리 (`time_of_event`, `time_of_alert` 컬럼 0으로 채움)

---

> Q: 위 코드 흐름 이후에 InceptionV3를 사용하면,
Farneback Optical Flow + 데이터 증강(AddFog, AddRain 등) 이후 프레임 특징을 추출할 수 있는 거야?
> 

✅ 답은 **거의 YES**,

하지만 **Farneback Optical Flow랑 CNN 특징 추출은 별개의 파이프라인**이라는 점만 주의하면 돼!

# 🎯 네가 구성한 흐름을 정확히 뜯어보자

| 순서 | 설명 |
| --- | --- |
| 1. extract_keyframes | 충돌 직전 집중해서 **프레임을 추출** |
| 2. get_video_transforms (Compose) | **데이터 증강** 적용 (Flip, Fog, Rain 등) |
| 3. compute_optical_flow | 추출한 프레임들로 **Optical Flow(움직임 벡터)** 계산 |
| 4. (선택) Optical Flow 벡터로 별도 특징 추출 (움직임 크기 등) |  |
| 5. InceptionV3 | **프레임 자체**에 대해 **CNN 특징 추출** |

# ✨ 아주 중요한 포인트

| 구분 | Optical Flow | InceptionV3 Feature |
| --- | --- | --- |
| 입력 데이터 | 프레임들 사이의 움직임(변화량) | 개별 프레임 이미지 자체 |
| 출력 데이터 | 움직임 벡터 or 이동 크기(norm) | 1280-dim or 2048-dim CNN Feature Vector |
| 쓰임새 | 이동 패턴 분석 (ex: 흔들림, 충돌 감지) | 이미지 시각 특징 분석 (ex: 도로, 차량, 상황 파악) |

✅ **즉, 같은 프레임들로 Optical Flow와 CNN Feature를 각각 따로 뽑을 수 있어!**

# 🧠 쉽게 비유하면

- **Optical Flow**:
    
    → "화면이 어떻게 움직였는지"를 숫자로 뽑아낸다 (움직임의 방향 + 세기)
    
- **InceptionV3 CNN Feature**:
    
    → "화면에 무엇이 보이는지"를 숫자로 뽑아낸다 (차, 사람, 도로, 주변 상황 등)
    

---

> get_hybrid_features(video_path, alert_time, event_time)를 완성하고 싶은데,
> 
> 
> 현재 없는 함수인 **`extract_critical_frames`** 부분을 현실적으로 대체하거나 수정해서
> 
> **CNN + Optical Flow 특성**을 안전하게 추출하는 걸 목표로 하고 있지?
> 

**완벽하게 이해했어.**

---

### 🔥 문제 포인트

- **`extract_critical_frames()`** 함수는 없지만,
    
    **`extract_keyframes()`** 함수는 네가 이미 만들어 놓았잖아? (후반부 프레임에 집중하는!)
    
- 그리고 **`alert_time`, `event_time`*을 활용하고 싶은 의도가 보여.
    
    (충돌 위험이 있는 구간만 뽑고 싶다는 거.)
    

---

### ✨ 그래서 내가 추천하는 수정 방향은 이거야:

| 기존 | 수정 제안 |
| --- | --- |
| `extract_critical_frames` | ➔ `extract_keyframes`를 수정해서 alert_time~event_time 구간에 집중 |
| sampling_interval | ➔ 총 `num_frames` 개수를 맞춰 균등 추출 |
| event_time 없는 경우(negative sample) | ➔ 그냥 마지막 3~5초를 기준으로 프레임 뽑기 |

## 📌 요약

- Positive이면 **alert~event** 사이만 뽑고,
- Negative이면 **마지막 3초 구간**을 기준으로 추출.
- 그리고 **num_frames**만큼 균등하게 뽑고 CNN + Optical Flow 둘 다 계산.

## ❓ 추가로 물어볼 것

- CNN backbone을 `InceptionV3` 계속 쓸까?
    
    (EfficientNetB0도 가져왔던데, 바꿀 계획 있으면 알려줘!)
    

## ✨ 현재 `get_hybrid_features()`가 하는 일

| 과정 | 설명 | 왜 하는가? |
| --- | --- | --- |
| 1. **프레임 추출** | 영상에서 주요 구간(주로 충돌 직전) 프레임들을 추출 | 사고 징후가 잘 보이는 시점에 집중하기 위해 |
| 2. **CNN 특성 추출** | InceptionV3로 각 프레임의 **시각적 특징(공간적 특성)** 추출 | (ex) 차량 모양, 거리, 장애물 등) |
| 3. **Optical Flow 특성 추출** | 프레임 간 **움직임 변화(모션 벡터)**를 계산해서 평균 이동량 추출 | (ex) 급격한 가속/회피/충돌 직전 모션) |
| 4. **CNN 특성과 Optical Flow 특성 결합** | 하나의 벡터로 만듦 (1280 + 1 = 1281차원) | 사고를 예측하기 위한 종합적인 "정적+동적 정보"를 학습하게 하기 위해 |

## 🎯 결론

> 네 get_hybrid_features()는
> 
> 
> **"프레임 기반의 공간적 특성 + 움직임 기반의 시간적 특성"**
> 
> 둘 다 추출해서 하나의 입력으로 만드는 **하이브리드 특성 추출 함수**야.
> 

## 🔥 아주 쉽게 비유하면

> 그냥 CNN으로만 하면 "사진만 보고 판단"하는 거고,
> 
> 
> Optical Flow까지 추가하면 "움직이는 상황을 같이 보고 판단"하는 거야.
> 

(사고는 정적인 상황보다 "급격한 변화"를 잡는 게 중요하니까,

이렇게 모션 정보를 추가하면 정확도가 확 올라가.)

## 🚀 그럼 다음 스텝

이 `get_hybrid_features()`를 기반으로

**1개 영상당 하나의 1281차원 피처**를 뽑을 수 있으니,

- 이제 Dataset →
- 그리고 이걸 받아서 학습하는 모델 (ex. LSTM, Transformer) 구조로 자연스럽게 이어가면 돼.

---

## ✨ 먼저, `get_video_transforms()` 역할부터 설명할게

**`def get_video_transforms()`는**

여러 개 증강 클래스 (`RandomHorizontalFlip`, `AddFog`, `AddRain` 등)를

**하나로 "묶어놓은 파이프라인"** 을 만드는 함수야.

| 역할 | 설명 |
| --- | --- |
| ✔️ | 여러 증강 클래스를 순서대로 적용할 수 있게 묶는다 |
| ✔️ | "Train"일 때는 강한 증강, "Val"일 때는 증강 없이 ToTensor만 적용할 수 있게 분기 |
| ✔️ | 코드가 깔끔해짐. Dataset 만들 때 바로 `transform['train'](frames)` 이런 식으로 쉽게 호출 가능 |

## ✨ 다음, `class ToTensor()`는 왜 필요한가?

**`ToTensor()`는**

프레임을 **numpy array ➔ PyTorch tensor**로 변환해주는 클래스야.

| 역할 | 설명 |
| --- | --- |
| ✔️ | OpenCV나 일반 numpy 기반 프레임은 `(T, H, W, C)` 순서인데 |
| ✔️ | PyTorch 모델은 `(T, C, H, W)` 순서를 요구해 |
| ✔️ | 그리고 데이터 타입도 float32로 바꾸고, 픽셀 범위를 0~1로 정규화해야 해 |

## ✨ 요약 정리

| 질문 | 답변 |
| --- | --- |
| `get_video_transforms()` | **(자동화용)** 여러 증강을 한번에 묶는 함수. 꼭 필요하진 않음 (수동처리하면 됨) |
| `ToTensor()` | **(필수)** PyTorch 모델 입력 포맷 `(T, C, H, W)`로 변환하는 거라 필요함 |