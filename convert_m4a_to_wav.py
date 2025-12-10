import os
import glob
from pydub import AudioSegment

SRC_DIR = "dataset"
DST_DIR = "dataset_wav"
CLASSES = ["Chatting", "Game", "web_surfing", "Writing"]
TARGET_SR = 16000


def convert_m4a_files():
    print(f"--- .m4a -> .wav 변환 시작 ---")
    print(f"원본 위치: {SRC_DIR}")
    print(f"저장 위치: {DST_DIR}")

    converted_count = 0

    for c in CLASSES:
        src_class_dir = os.path.join(SRC_DIR, c)
        dst_class_dir = os.path.join(DST_DIR, c)

        os.makedirs(dst_class_dir, exist_ok=True)

        m4a_files = glob.glob(os.path.join(src_class_dir, "**", "*.m4a"), recursive=True)

        if not m4a_files:
            print(f"[INFO] '{c}' 클래스에 .m4a 파일이 없습니다.")
            continue

        print(f"\n[INFO] 클래스 '{c}' 처리 중... ({len(m4a_files)}개)")

        for m4a_path in m4a_files:
            filename = os.path.basename(m4a_path)
            wav_filename = filename.replace(".m4a", ".wav").replace(".M4A", ".wav")

            wav_path = os.path.join(dst_class_dir, wav_filename)

            if os.path.exists(wav_path):
                continue

            try:
                audio = AudioSegment.from_file(m4a_path, format="m4a")
                audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
                audio.export(wav_path, format="wav")

                print(f"  [DONE] 저장 완료: {wav_path}")
                converted_count += 1

            except Exception as e:
                print(f"  [FAIL] 변환 실패 ({m4a_path}): {e}")

    print(f"\n--- 총 {converted_count}개 파일 변환 및 이동 완료 ---")


if __name__ == "__main__":
    convert_m4a_files()