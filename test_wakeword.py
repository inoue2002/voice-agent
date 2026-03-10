"""
Step 2: ウェイクワード検出テスト (デバッグ版)
スコアを常時表示して、検出状況を確認する。
"""
import sounddevice as sd
import numpy as np
import sys
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms at 16kHz
THRESHOLD = 0.3  # 閾値を下げて検出しやすく

print("=== ウェイクワード検出テスト (デバッグ版) ===")
print("モデルをロード中...")

model = Model(
    wakeword_models=["hey_jarvis_v0.1"],
    inference_framework="onnx",
)

print()
print('ウェイクワード: "Hey Jarvis"')
print(f"検出閾値: {THRESHOLD}")
print()
print('マイクに向かって "Hey Jarvis" と言ってください')
print("スコアが0.01以上のとき表示します")
print("-" * 50)

count = 0
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        blocksize=CHUNK_SIZE) as stream:
        while True:
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            audio_int16 = audio_chunk.flatten()

            prediction = model.predict(audio_int16)

            for model_name, score in prediction.items():
                # スコアが0.01以上なら表示（反応を確認）
                if score > 0.01:
                    bar = "█" * int(score * 50)
                    print(f"  [{model_name}] {score:.3f} {bar}")
                    sys.stdout.flush()

                if score > THRESHOLD:
                    print(f"\n🎤 検出! スコア: {score:.3f}")
                    print("   → ウェイクワード認識成功!")
                    print()
                    sys.stdout.flush()
                    model.reset()

            count += 1
            # 5秒ごとにハートビート表示（動作確認用）
            if count % 62 == 0:  # ~5sec at 80ms chunks
                print("  ... listening ...", flush=True)

except KeyboardInterrupt:
    print("\n\n終了しました。")
