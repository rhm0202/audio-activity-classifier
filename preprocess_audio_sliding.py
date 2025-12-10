import os, json, glob, warnings
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

RAW_DIR = "dataset_wav"

# 결과물을 저장할 폴더
OUT_DIR = "data/processed"

# 실제 폴더 이름과 정확히 일치해야 합니다.
CLASSES = ["Chatting", "Game", "web_surfing", "Writing"]

SR = 16000
WIN_SEC = 10.0  # 윈도우 길이 (10초)

# 데이터가 적은 클래스(Chatting, web_surfing)는 촘촘하게 자르기
# 데이터가 많은 클래스(Game, Writing)는 듬성듬성 자르기
HOP_SETTINGS = {
    "Chatting": 1.0,
    "web_surfing": 1.0,
    "Game": 3.0,
    "Writing": 3.0,
}

N_MELS = 64
WIN_MS = 25
HOP_MS = 10
TEST_SIZE = 0.20
VAL_SIZE = 0.20
SEED = 42

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def highpass_simple(x: np.ndarray):
    """DC 오프셋 제거 필터"""
    return np.concatenate([[x[0]], x[1:] - x[:-1]])


def load_mono_16k(path):
    """wav 파일 로드"""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def segment_sliding(y, sr=SR, win_sec=WIN_SEC, hop_sec=1.0):
    """지정된 hop_sec 간격으로 오디오 자르기"""
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if len(y) < win: return []
    starts = np.arange(0, len(y) - win + 1, hop, dtype=int)
    return [(s, y[s:s + win]) for s in starts]


def logmel(x, sr=SR, n_mels=N_MELS, win_ms=WIN_MS, hop_ms=HOP_MS):
    """로그 멜 스펙트로그램 특징 추출"""
    n_fft = int(sr * win_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, n_mels=n_mels, power=2.0
    )
    return np.log(S + 1e-6).astype(np.float32)


def scan_files():
    """dataset_wav 폴더에서 wav 파일 스캔"""
    files, labels = [], []
    for ci, c in enumerate(CLASSES):
        # dataset_wav 폴더 아래에서 검색
        files_c = glob.glob(os.path.join(RAW_DIR, c, "**", "*.wav"), recursive=True)
        files.extend(files_c)
        labels.extend([ci] * len(files_c))
    return np.array(files), np.array(labels, dtype=np.int64)


def split_by_file(files, labels):
    """파일 단위로 Train/Val/Test 분할 (컨닝 방지)"""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    idx = np.arange(len(files))
    train_idx, test_idx = next(gss.split(idx, labels, groups=files))
    files_tr, files_te = files[train_idx], files[test_idx]
    labels_tr, labels_te = labels[train_idx], labels[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
    idx2 = np.arange(len(files_tr))
    tr_idx, val_idx = next(gss2.split(idx2, labels_tr, groups=files_tr))
    files_tr2, files_va = files_tr[tr_idx], files_tr[val_idx]
    labels_tr2, labels_va = labels_tr[tr_idx], labels_tr[val_idx]

    return (files_tr2, labels_tr2), (files_va, labels_va), (files_te, labels_te)


# ===== 메인 실행 =====
def main():
    ensure_dirs()

    print(f"--- '{RAW_DIR}' 폴더에서 데이터 스캔 중... ---")
    files, labels = scan_files()

    if len(files) == 0:
        print(f"[ERROR] .wav 파일이 없습니다! 변환 코드를 먼저 실행했는지 확인하세요.")
        return

    print(f"총 발견된 파일 수: {len(files)}개")

    (tr_files, tr_lab), (va_files, va_lab), (te_files, te_lab) = split_by_file(files, labels)

    def extract_from_file_list(file_list, lab_list, split_name):
        X_list, y_list, meta_rows = [], [], []
        lab_to_name = {i: c for i, c in enumerate(CLASSES)}

        for path, lab in zip(file_list, lab_list):
            class_name = lab_to_name[int(lab)]
            current_hop = HOP_SETTINGS.get(class_name, 1.0)

            try:
                y = highpass_simple(load_mono_16k(path))
                segs = segment_sliding(y, SR, WIN_SEC, hop_sec=current_hop)

                if not segs: continue

                for start_idx, seg in segs:
                    M = logmel(seg, SR)
                    X_list.append(M)
                    y_list.append(lab)

                    meta_rows.append({
                        "src_path": path,
                        "class_id": int(lab),
                        "class_name": class_name,
                        "start_sec": round(start_idx / SR, 3),
                        "hop_sec": current_hop
                    })
            except Exception as e:
                print(f"[WARN] 처리 실패: {path} -> {e}")

        if len(X_list) == 0: return None, None, None

        T_max = max(M.shape[1] for M in X_list)
        X = np.zeros((len(X_list), N_MELS, T_max), dtype=np.float32)
        for i, M in enumerate(X_list):
            X[i, :, :M.shape[1]] = M

        return X, np.array(y_list, dtype=np.int64), pd.DataFrame(meta_rows)

    print("--- 훈련(Train) 데이터 추출 중 (가변 홉 적용) ---")
    X_tr, y_tr, meta_tr = extract_from_file_list(tr_files, tr_lab, "train")

    print("--- 검증(Val) 데이터 추출 중 ---")
    X_va, y_va, meta_va = extract_from_file_list(va_files, va_lab, "val")

    print("--- 테스트(Test) 데이터 추출 중 ---")
    X_te, y_te, meta_te = extract_from_file_list(te_files, te_lab, "test")

    # 4. 정규화 및 저장
    if X_tr is not None:
        mean = float(X_tr.mean())
        std = float(X_tr.std() + 1e-8)
        norm = lambda a: (a - mean) / std

        X_tr, X_va, X_te = norm(X_tr), norm(X_va), norm(X_te)

        np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), X=X_tr, y=y_tr)
        np.savez_compressed(os.path.join(OUT_DIR, "val.npz"), X=X_va, y=y_va)
        np.savez_compressed(os.path.join(OUT_DIR, "test.npz"), X=X_te, y=y_te)

        meta_all = pd.concat([
            meta_tr.assign(split="train"),
            meta_va.assign(split="val"),
            meta_te.assign(split="test")
        ], ignore_index=True)
        meta_all.to_csv(os.path.join(OUT_DIR, "meta.csv"), index=False, encoding="utf-8-sig")

        # 통계 저장
        with open(os.path.join(OUT_DIR, "norm_stats.json"), "w") as f:
            json.dump({"mean": mean, "std": std, "classes": CLASSES}, f, indent=2)

        print("\n[완료] 데이터 전처리가 끝났습니다!")
        print(f"- 저장 위치: {OUT_DIR}")
        print(f"- 생성된 데이터: train.npz ({len(X_tr)}개), val.npz ({len(X_va)}개), test.npz ({len(X_te)}개)")


if __name__ == "__main__":
    main()