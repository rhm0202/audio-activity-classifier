# Audio Activity Classifier

> 오디오로 PC 사용 행동을 분류하는 CNN | Python | 머신러닝 수업 기말 프로젝트

녹음한 오디오를 분석해 사용자가 지금 무엇을 하고 있는지(채팅 / 게임 / 웹서핑 / 글쓰기) 분류하는 모델입니다.

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 분류 대상 | Chatting / Game / Web Surfing / Writing (4클래스) |
| 입력 | 10초 오디오 → 로그 멜 스펙트로그램 (64 × 1001) |
| 모델 | 2D CNN (TensorFlow/Keras) |
| 최종 정확도 | **92%** (5차 시도) |
| 기간 | 2025.12.02 ~ 2025.12.15 |
| 구성 | 개인 |

---

## 개발 과정

데이터와 전처리를 바꿔 가며 반복 실험해 정확도를 올렸습니다. 시도별 코드와 결과는 각 폴더에 그대로 남겨 두었습니다.

| 시도 | 결과 | 폴더 |
|---|---|---|
| 1차 | 실패 | `1차 실패/` |
| 2차 | 실패 | `2차 실패/` |
| 3차 | 81% | `3차  정확도81퍼/` |
| 4차 | 85% | `4차 정확도 85퍼/` |
| **5차** | **92%** | `5차 정확도 92퍼/` |

---

## 파이프라인

```
m4a 원본 (dataset/)
    ↓ convert_m4a_to_wav.py         wav 변환 (dataset_wav/)
    ↓ preprocess_audio_sliding.py   슬라이딩 윈도우 + 로그 멜 스펙트로그램 (data/processed/*.npz)
    ↓ make_model.ipynb              CNN 학습 (model_output/)
    ↓ test_audio_model.py           예측 및 평가
```

스크립트와 노트북은 레포 루트에 있으며 루트에서 실행합니다. 전처리 결과(`data/`)는 용량 때문에 레포에서 제외했습니다.

## 핵심 구현

### 전처리 (`preprocess_audio_sliding.py`)
- **슬라이딩 윈도우**: 10초 단위로 오디오 분할
- **가변 홉**: 데이터가 적은 클래스(Chatting, Web Surfing)는 0.5초, 많은 클래스(Game, Writing)는 3초 간격 → 클래스 불균형 완화
- **로그 멜 스펙트로그램**: 64 Mel bands, 25ms 윈도우, 10ms 홉
- **파일 단위 분할** (GroupShuffleSplit): 같은 파일의 조각이 학습·테스트에 함께 들어가는 데이터 누수 방지
- Train 기준 정규화, DC 오프셋 제거

### 학습
- Conv2D + MaxPooling + Dropout 구조의 CNN
- EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 콜백 사용

---

## 데이터셋

직접 녹음한 m4a 파일 (`dataset/`): 채팅, 게임(롤·메이플·배그), 웹서핑, 글쓰기 상황별 녹음

## 문서

레포 루트에 있습니다.

- 최종 보고서 (PDF / HWP), 발표 자료 (PDF)
- 1차 · 2차 중간 과정 기술서 (DOCX)
- 과제 명세 (`2025_머신러닝_기말프로젝트.pdf`)

## 기술 스택

- **Language**: Python
- **Library**: TensorFlow/Keras, librosa, numpy, scikit-learn, matplotlib
- **환경**: Jupyter Notebook
