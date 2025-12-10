import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. 환경 설정 및 상수 정의 ---
OUT_DIR = "data/processed"
MODEL_DIR = "model_output"

TEST_PATH = os.path.join(OUT_DIR, "test.npz")
STATS_PATH = os.path.join(OUT_DIR, "norm_stats.json")
# .h5 파일 로드 시 경로 주의
MODEL_NAME = os.path.join(MODEL_DIR, "best_audio_model.keras")

INPUT_SHAPE = (64, 1001, 1)
NUM_CLASSES = 4


def load_test_data():
    print("\n--- 테스트 데이터 로드 시작 ---")

    if not os.path.exists(STATS_PATH) or not os.path.exists(TEST_PATH):
        print("[ERROR] 데이터 파일이 없습니다. preprocess 코드를 먼저 실행하세요.")
        return None, None, None

    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
        CLASSES = stats["classes"]

    data_te = np.load(TEST_PATH)
    X_test, y_test = data_te['X'], data_te['y']
    X_test = np.expand_dims(X_test, axis=-1)
    y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)

    print(f"테스트 데이터 형태(X): {X_test.shape}")
    print(f"테스트 레이블 형태(y): {y_test_onehot.shape}")

    return X_test, y_test_onehot, CLASSES


def test_model(X_test, y_test_onehot, CLASSES):
    print("\n--- 모델 로드 및 평가 시작 ---")

    # ★ 한글 경로 문제 해결을 위한 트릭 ★
    # 1. 현재 작업 위치를 기억해둠
    original_dir = os.getcwd()

    try:
        # 2. 모델이 있는 폴더로 작업 위치를 강제로 이동
        os.chdir(MODEL_DIR)
        print(f"📂 폴더 이동 완료: {os.getcwd()}")

        # 3. 경로 없이 파일 이름만으로 로드 (한글 경로 에러 회피)
        if not os.path.exists(MODEL_NAME):
            raise FileNotFoundError(f"폴더 안에 '{MODEL_NAME}' 파일이 없습니다.")

        model = load_model(MODEL_NAME)
        print(f"✅ 모델 로드 성공!")

    except Exception as e:
        print(f"[CRITICAL] 모델 로드 실패: {e}")
        # 실패하더라도 원래 위치로 복귀
        os.chdir(original_dir)
        return
    finally:
        # 4. 원래 위치로 복귀
        os.chdir(original_dir)

    # 5. 전체 데이터 예측
    print("\n[ 전체 데이터 예측 중... ]")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test_onehot, axis=1)

    # 6. 최종 결과 출력
    loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)

    print("\n" + "=" * 50)
    print(f"📢 최종 테스트 결과 리포트")
    print("=" * 50)
    print(f"▶ 손실값 (Loss)      : {loss:.4f}")
    print(f"▶ 정확도 (Accuracy)  : {accuracy * 100:.2f}%")
    print("=" * 50)

    print("\n[ 상세 성능 지표 (F1-Score) ]")
    print(classification_report(y_true_labels, y_pred_labels, target_names=CLASSES))

    print("\n[ 혼동 행렬 (Confusion Matrix) ]")
    print(confusion_matrix(y_true_labels, y_pred_labels))
    print("-" * 50)


if __name__ == "__main__":
    X_test, y_test_onehot, CLASSES = load_test_data()
    if X_test is not None:
        test_model(X_test, y_test_onehot, CLASSES)