# Machine_Learning_final_project

> 오디오 행동 분류 모델 | Python | 머신러닝 수업 기말 과제

사용자의 오디오를 분석해 현재 행동(채팅/게임/웹서핑/타이핑)을 분류하는 딥러닝 모델입니다.

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 분류 대상 | Chatting / Game / Web Surfing / Writing (4클래스) |
| 최종 정확도 | **92%** (5차 시도) |
| 개발 기간 | 2025.11 ~ 2025.12 |

---

## 개발 과정

반복 실험을 통해 정확도를 단계적으로 개선하였습니다.

| 시도 | 결과 |
|---|---|
| 1차 | 실패 |
| 2차 | 실패 |
| 3차 | 81% |
| 4차 | 85% |
| **5차** | **92%** |

---

## 파이프라인

```
m4a 원본 파일
    ↓ convert_m4a_to_wav.py        wav 변환
    ↓ preprocess_audio_sliding.py  슬라이딩 윈도우 + 로그 멜 스펙트로그램
    ↓ make_model.ipynb             모델 학습
    ↓ test_audio_model.py          예측 및 평가
```

---

## 핵심 구현

### 전처리 (preprocess_audio_sliding.py)
- **슬라이딩 윈도우**: 10초 단위 오디오 분할
- **가변 홉**: 데이터 적은 클래스(Chatting, Web Surfing) 1초 / 많은 클래스(Game, Writing) 3초 간격으로 클래스 불균형 완화
- **로그 멜 스펙트로그램**: 64 Mel bands, 25ms 윈도우, 10ms 홉
- **파일 단위 분할** (GroupShuffleSplit): 동일 파일 세그먼트가 학습·테스트에 동시 포함되는 데이터 누수 방지
- Train 기준 정규화 전체 적용, DC 오프셋 제거

---

## 기술 스택

- **Language**: Python
- **Library**: librosa, numpy, pandas, scikit-learn
- **학습**: Jupyter Notebook (make_model.ipynb)

---

## GitHub

[https://github.com/rhm0202/Machine_Learning_final_project](https://github.com/rhm0202/Machine_Learning_final_project)
