"""
Step 3: ウェイクワード検出 → STT 統合テスト

1. "Hey Jarvis" を検出
2. 通知音（ビープ）を鳴らす
3. 発話を録音（無音で自動停止）
4. faster-whisper で日本語テキスト変換
5. 結果を表示
6. ウェイクワード待ちに戻る

Ctrl+C で終了。
"""
import sounddevice as sd
import numpy as np
import sys
import io
import wave
import tempfile
from openwakeword.model import Model
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms
WAKEWORD_THRESHOLD = 0.3

# 録音設定
SILENCE_THRESHOLD = 300     # 無音判定のRMS閾値（環境に合わせて調整）
SILENCE_DURATION = 1.2      # 無音がこの秒数続いたら録音終了
MAX_RECORD_SECONDS = 10     # 最大録音時間
SPEECH_START_TIMEOUT = 3.0  # 発話開始までの待機時間

print("=== ウェイクワード → STT 統合テスト ===")
print()

# モデルロード
print("openWakeWord モデルをロード中...")
wakeword_model = Model(
    wakeword_models=["hey_jarvis_v0.1"],
    inference_framework="onnx",
)

print("Whisper モデルをロード中 (初回はダウンロードに時間がかかります)...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print("  → モデルロード完了!")
print()
print('マイクに向かって "Hey Jarvis" と言ってから、日本語で話してください')
print("例: 「Hey Jarvis」→「部屋の電気つけて」")
print("Ctrl+C で終了")
print("=" * 50)


def beep():
    """短いビープ音を鳴らす"""
    duration = 0.15
    freq = 800
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
    sd.play(tone, SAMPLE_RATE)
    sd.wait()


def record_until_silence():
    """無音を検出するまで録音し、音声データを返す"""
    print("  🔴 録音中... (話し終わると自動停止)")
    sys.stdout.flush()

    # まず環境ノイズレベルを測定（0.5秒）
    print("  📊 環境音レベル測定中...")
    sys.stdout.flush()
    noise_samples = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        blocksize=CHUNK_SIZE) as stream:
        for _ in range(int(0.5 / (CHUNK_SIZE / SAMPLE_RATE))):
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            rms = np.sqrt(np.mean(audio_chunk.flatten().astype(np.float32) ** 2))
            noise_samples.append(rms)

    noise_floor = np.mean(noise_samples)
    dynamic_threshold = max(noise_floor * 2, 200)
    print(f"  📊 ノイズ: {noise_floor:.0f}, 閾値: {dynamic_threshold:.0f}")
    print("  🔴 録音中... (話し終わると自動停止)")
    sys.stdout.flush()

    frames = []
    silence_chunks = 0
    silence_chunks_needed = int(SILENCE_DURATION / (CHUNK_SIZE / SAMPLE_RATE))
    max_chunks = int(MAX_RECORD_SECONDS / (CHUNK_SIZE / SAMPLE_RATE))
    start_timeout_chunks = int(SPEECH_START_TIMEOUT / (CHUNK_SIZE / SAMPLE_RATE))
    has_speech = False
    total_chunks = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        blocksize=CHUNK_SIZE) as stream:
        for _ in range(max_chunks):
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            audio_flat = audio_chunk.flatten()
            frames.append(audio_flat.copy())
            total_chunks += 1

            rms = np.sqrt(np.mean(audio_flat.astype(np.float32) ** 2))

            # RMSをリアルタイム表示（デバッグ用）
            if total_chunks % 5 == 0:  # 0.4秒ごと
                marker = "🟢" if rms > dynamic_threshold else "⚪"
                print(f"    {marker} RMS: {rms:.0f} (閾値: {dynamic_threshold:.0f})", flush=True)

            if rms > dynamic_threshold:
                has_speech = True
                silence_chunks = 0
            else:
                silence_chunks += 1

            # 発話があった後に無音が続いたら終了
            if has_speech and silence_chunks >= silence_chunks_needed:
                break

            # 発話が始まらないままタイムアウト
            if not has_speech and total_chunks >= start_timeout_chunks:
                print("  ⚠️  発話が検出されませんでした")
                sys.stdout.flush()
                break

    audio_data = np.concatenate(frames)
    duration = len(audio_data) / SAMPLE_RATE
    print(f"  ⏹️  録音完了 ({duration:.1f}秒)")
    sys.stdout.flush()
    return audio_data


def transcribe(audio_data):
    """faster-whisperで音声をテキストに変換"""
    print("  🔄 テキスト変換中...")
    sys.stdout.flush()

    # int16 → float32 に変換（-1.0〜1.0）
    audio_float = audio_data.astype(np.float32) / 32768.0

    segments, info = whisper_model.transcribe(
        audio_float,
        language="ja",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=1000,
            speech_pad_ms=200,
        ),
    )

    text = "".join(segment.text for segment in segments).strip()
    return text


# メインループ
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        blocksize=CHUNK_SIZE) as stream:
        print("\n👂 ウェイクワード待ち中...")
        sys.stdout.flush()

        while True:
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            audio_int16 = audio_chunk.flatten()

            prediction = wakeword_model.predict(audio_int16)

            for model_name, score in prediction.items():
                if score > WAKEWORD_THRESHOLD:
                    print(f"\n🎤 ウェイクワード検出! (スコア: {score:.3f})")
                    sys.stdout.flush()
                    wakeword_model.reset()

                    # ビープ音
                    beep()

                    # 録音
                    audio_data = record_until_silence()

                    # STT
                    text = transcribe(audio_data)

                    if text:
                        print(f"\n  📝 認識結果: 「{text}」")
                    else:
                        print(f"\n  ❌ 音声を認識できませんでした")

                    print()
                    print("=" * 50)
                    print("👂 ウェイクワード待ち中...")
                    sys.stdout.flush()

except KeyboardInterrupt:
    print("\n\n終了しました。")
