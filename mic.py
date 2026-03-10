"""Step 1: マイク動作確認 — 3秒間録音して音量を表示"""
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3

print("=== マイク動作確認 ===")
print(f"利用可能なデバイス:")
print(sd.query_devices())
print()
print(f"デフォルト入力: {sd.query_devices(kind='input')['name']}")
print()
print(f"3秒間録音します... 何か話してください")

audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()

audio_np = audio.flatten().astype(np.float32)
rms = np.sqrt(np.mean(audio_np ** 2))
peak = np.max(np.abs(audio_np))

print(f"\n録音完了!")
print(f"  RMS音量: {rms:.0f}")
print(f"  ピーク: {peak:.0f}")
print(f"  サンプル数: {len(audio_np)}")

if rms > 500:
    print("\n✓ マイクは正常に動作しています!")
elif rms > 100:
    print("\n△ 音が小さいです。マイクに近づいて話してみてください。")
else:
    print("\n✗ 音が検出できませんでした。マイクの接続を確認してください。")
