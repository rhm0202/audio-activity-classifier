import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# --- 1. 환경 설정 및 상수 정의 ---
OUT_DIR = "data/processed"
MODEL_DIR = "model_output"

TEST_PATH = os.path.join(OUT_DIR, "test.npz")
STATS_PATH = os.path.join(OUT_DIR, "norm_stats.json")
# .h5 파일 로드 시 경로 주의
MODEL_PATH = os.path.join(MODEL_DIR, "best_audio_model.h5")

INPUT_SHAPE = (64, 1001, 1)
NUM_CLASSES = 4

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] 모델 파일이 존재하지 않습니다: {MODEL_PATH}")
    exit()


def load_and_normalize_data():
    print("\n--- 테스트 데이터 로드 시작 ---")

    # 1. 클래스 이름 로드 (정규화 값은 필요 없음)
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
        CLASSES = stats["classes"]

    # 2. 테스트 데이터 로드 (이미 전처리/정규화 완료된 상태)
    data_te = np.load(TEST_PATH)
    X_test, y_test = data_te['X'], data_te['y']

    # [수정됨] 정규화 코드 삭제! (전처리 단계에서 이미 수행됨)
    # X_test = (X_test - mean) / (std + 1e-8)  <-- 이거 지워야 함

    # 3. 차원 추가 (N, 64, 1001) -> (N, 64, 1001, 1)
    # 모델 학습 시 expand_dims를 했으므로 여기서도 해줘야 함
    X_test = np.expand_dims(X_test, axis=-1)

    # 4. 정수 레이블을 원-핫 인코딩으로 변환
    y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)

    print(f"테스트 데이터 형태(X): {X_test.shape}")
    print(f"테스트 레이블 형태(y): {y_test_onehot.shape}")

    return X_test, y_test_onehot, CLASSES


def build_model_architecture():
    # 학습 때 사용한 모델 구조와 100% 동일해야 함
    model = Sequential([
        Input(shape=INPUT_SHAPE),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 4)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 4)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 4)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def test_model(X_test, y_test_onehot, CLASSES):
    print("\n--- 모델 로드 및 평가 시작 ---")

    # 방법 1: 구조 빌드 -> 가중치 로드 (기존 방식)
    model = build_model_architecture()

    try:
        model.load_weights(MODEL_PATH)
        print(f"성공적으로 모델 가중치 로드: {MODEL_PATH}")
    except Exception as e:
        print(f"[CRITICAL] 가중치 로드 실패. 모델 구조가 학습 때와 다른지 확인하세요.")
        print(f"에러 메시지: {e}")
        return

    # 컴파일 (평가를 위해 필수)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 최종 성능 평가
    print("\n[ 최종 테스트 데이터 성능 평가 ]")
    loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")  # 이제 0.0000이 나오지 않을 것입니다.

    # 샘플 예측
    print("\n[ 샘플 예측 결과 ]")
    num_samples = 10000
    curr_samples = min(num_samples, len(X_test))

    indices = np.random.choice(len(X_test), curr_samples, replace=False)
    X_sample = X_test[indices]
    y_true_labels = np.argmax(y_test_onehot[indices], axis=1)

    predictions = model.predict(X_sample)
    y_pred_labels = np.argmax(predictions, axis=1)

    print("=" * 40)
    print(f"▶ 손실값 (Loss)     : {loss:.4f}")
    print(f"▶ 정답률 (Accuracy) : {accuracy * 100:.2f}%")
    print("=" * 40)

    for i in range(curr_samples):
        true_name = CLASSES[y_true_labels[i]]
        pred_name = CLASSES[y_pred_labels[i]]
        confidence = predictions[i][y_pred_labels[i]] * 100
        result = "✅ 정답" if true_name == pred_name else "❌ 오답"
        print(f"Sample {i + 1}: [실제] {true_name} vs [예측] {pred_name} ({confidence:.1f}%) -> {result}")



if __name__ == "__main__":
    X_test, y_test_onehot, CLASSES = load_and_normalize_data()
    test_model(X_test, y_test_onehot, CLASSES)