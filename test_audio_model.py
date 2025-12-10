import os
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

print("\n==============================")
print("📌 최종 모델 검증을 시작합니다")
print("==============================\n")

# --- 경로 설정 ---
BASE_DIR = os.getcwd()
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "model_output")

TEST_PATH  = os.path.join(OUT_DIR, "test.npz")
STATS_PATH = os.path.join(OUT_DIR, "norm_stats.json")

# 🔥 체크포인트로 저장된 모델 파일명 자동 선택 (.keras 또는 .h5)
keras_path = os.path.join(MODEL_DIR, "best_audio_model.keras")
h5_path    = os.path.join(MODEL_DIR, "best_audio_model.h5")

if os.path.exists(h5_path):
    MODEL_PATH = h5_path
elif os.path.exists(keras_path):
    MODEL_PATH = keras_path
else:
    raise FileNotFoundError("best_audio_model.{keras/h5} 파일을 찾을 수 없습니다.")

print(f"📄 MODEL_PATH = {MODEL_PATH}")

# ================================
# 1) 테스트 데이터 로드
# ================================
print("\n--- 1) 테스트 데이터 로드 ---")

with open(STATS_PATH, "r") as f:
    stats = json.load(f)
    CLASSES = stats["classes"]

data_te = np.load(TEST_PATH)
X_test, y_test = data_te["X"], data_te["y"]

X_test = np.expand_dims(X_test, axis=-1)
y_test_onehot = to_categorical(y_test, num_classes=len(CLASSES))

print(f"✔ X_test shape = {X_test.shape}")
print(f"✔ y_test shape = {y_test_onehot.shape}")


# ================================
# 2) 모델 로드
# ================================
print("\n--- 2) 모델 로드 ---")
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✔ 모델 로드 성공!")
except Exception as e:
    print("❌ 모델 로드 실패:", e)
    raise

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# ================================
# 3) 최종 평가
# ================================
print("\n--- 3) 최종 테스트 데이터 평가 ---")
loss, acc = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"🎯 Loss      : {loss:.4f}")
print(f"🎯 Accuracy  : {acc*100:.2f}%")

# ================================
# 4) 샘플 예측 출력
# ================================
print("\n--- 4) 샘플 예측 결과 ---")

num_samples = 1000
idx = np.random.choice(len(X_test), num_samples, replace=False)

pred = model.predict(X_test[idx])
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y_test_onehot[idx], axis=1)

for i in range(num_samples):
    t = CLASSES[y_true[i]]
    p = CLASSES[y_pred[i]]
    conf = pred[i][y_pred[i]] * 100

    result = "✅ 정답" if t == p else "❌ 오답"
    print(f"[{i+1}] 실제:{t} / 예측:{p} ({conf:.1f}%) → {result}")

print("\n==============================")
print("🔍 검증 코드 실행 완료!")
print("==============================")