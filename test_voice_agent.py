"""
Step 4: 音声エージェント (フルパイプライン)

1. "Hey Jarvis" を検出
2. ビープ音
3. 発話を録音（無音で自動停止）
4. faster-whisper で日本語テキスト変換
5. 応答を生成（LLM or パターンマッチ）
6. macOS say で音声応答
7. ウェイクワード待ちに戻る

Ctrl+C で終了。
"""
import sounddevice as sd
import numpy as np
import sys
import os
import subprocess
from openwakeword.model import Model
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
WAKEWORD_THRESHOLD = 0.3

# 録音設定
SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1.2
MAX_RECORD_SECONDS = 10
SPEECH_START_TIMEOUT = 3.0

# TTS設定
TTS_VOICE = "Kyoko"  # macOS日本語音声

# LLM設定（APIキーがあればClaude使用）
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
USE_LLM = False

if ANTHROPIC_API_KEY:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        USE_LLM = True
        print("✓ Claude API 接続済み")
    except ImportError:
        print("△ anthropic パッケージ未インストール。パターンマッチモードで動作します。")
else:
    print("△ ANTHROPIC_API_KEY 未設定。パターンマッチモードで動作します。")

print()
print("=== 音声エージェント ===")
print()

# モデルロード
print("openWakeWord モデルをロード中...")
wakeword_model = Model(
    wakeword_models=["hey_jarvis_v0.1"],
    inference_framework="onnx",
)

print("Whisper モデルをロード中...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print("  → 準備完了!")
print()
print('マイクに向かって "Hey Jarvis" と言ってから、日本語で話してください')
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


def speak(text):
    """macOS say コマンドで音声応答"""
    print(f"  🔊 応答: 「{text}」")
    sys.stdout.flush()
    subprocess.run(["say", "-v", TTS_VOICE, text])


def record_until_silence():
    """無音を検出するまで録音"""
    import time
    # ビープ音の残響が消えるまで少し待つ
    time.sleep(0.3)

    # 環境ノイズ測定（0.3秒）
    noise_samples = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        blocksize=CHUNK_SIZE) as stream:
        for _ in range(int(0.3 / (CHUNK_SIZE / SAMPLE_RATE))):
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            rms = np.sqrt(np.mean(audio_chunk.flatten().astype(np.float32) ** 2))
            noise_samples.append(rms)

    noise_floor = np.mean(noise_samples)
    dynamic_threshold = max(noise_floor * 1.8, 150)
    print(f"  🔴 録音中... (ノイズ: {noise_floor:.0f}, 閾値: {dynamic_threshold:.0f})")
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

            if rms > dynamic_threshold:
                has_speech = True
                silence_chunks = 0
            else:
                silence_chunks += 1

            if has_speech and silence_chunks >= silence_chunks_needed:
                break

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
    """faster-whisperで音声→テキスト"""
    print("  🔄 テキスト変換中...")
    sys.stdout.flush()

    audio_float = audio_data.astype(np.float32) / 32768.0

    segments, info = whisper_model.transcribe(
        audio_float,
        language="ja",
        initial_prompt="電気をつけて、エアコンを消して、今何時、天気を教えて",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=800,
            speech_pad_ms=200,
        ),
    )

    text = "".join(segment.text for segment in segments).strip()
    return text


def generate_response_llm(text):
    """Claude APIで応答生成"""
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system="あなたは家庭用音声アシスタントです。日本語で簡潔に（1-2文で）答えてください。スマートホームの操作を頼まれたら、実行した体で返答してください。",
            messages=[{"role": "user", "content": text}],
        )
        return message.content[0].text
    except Exception as e:
        print(f"  ⚠️  LLMエラー: {e}")
        return generate_response_pattern(text)


def generate_response_pattern(text):
    """パターンマッチで簡易応答"""
    if not text:
        return "すみません、聞き取れませんでした。"

    # スマートホーム系
    if "電気" in text and ("つけ" in text or "付け" in text):
        return "はい、電気をつけました。"
    if "電気" in text and ("消" in text or "けし" in text):
        return "はい、電気を消しました。"
    if "エアコン" in text and ("つけ" in text or "付け" in text):
        return "はい、エアコンをつけました。"
    if "エアコン" in text and ("消" in text or "けし" in text):
        return "はい、エアコンを消しました。"

    # 時間
    if "何時" in text or "時間" in text:
        from datetime import datetime
        now = datetime.now()
        return f"今は{now.hour}時{now.minute}分です。"

    # 天気
    if "天気" in text:
        return "すみません、天気情報にはまだ対応していません。"

    # デフォルト
    return f"「{text}」と認識しました。この操作にはまだ対応していません。"


def generate_response(text):
    """応答生成（LLM or パターンマッチ）"""
    if USE_LLM:
        return generate_response_llm(text)
    else:
        return generate_response_pattern(text)


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
                    print(f"  📝 認識: 「{text}」")
                    sys.stdout.flush()

                    # 応答生成
                    response = generate_response(text)

                    # TTS
                    speak(response)

                    print()
                    print("=" * 50)
                    print("👂 ウェイクワード待ち中...")
                    sys.stdout.flush()

except KeyboardInterrupt:
    print("\n\n終了しました。")
