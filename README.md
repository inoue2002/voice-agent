# voice-agent

「OK クロー」のようなウェイクワードで起動する、ローカル動作の音声エージェント。

## アーキテクチャ

```
マイク → ウェイクワード検出 → STT → 応答生成 → TTS → スピーカー
         (openWakeWord)    (Whisper) (LLM/パターン) (say/VOICEVOX)
```

## スクリプト

| ファイル | 説明 |
|---------|------|
| `mic.py` | マイク動作確認（3秒録音してRMS表示） |
| `wakeword.py` | ウェイクワード検出テスト（スコアのリアルタイム表示） |
| `wakeword_stt.py` | ウェイクワード → STT 統合テスト |
| `voice_agent.py` | フルパイプライン（Wake Word → STT → 応答生成 → TTS） |

## セットアップ

```bash
# 依存パッケージ (macOS)
brew install portaudio

# Python環境
python3 -m venv venv
./venv/bin/pip install sounddevice numpy openwakeword faster-whisper

# openWakeWordモデルのダウンロード
./venv/bin/python -c "from openwakeword.utils import download_models; download_models()"
```

## 使い方

```bash
# フルパイプライン起動
cd ~/github/inoue2002/voice-agent
PYTHONUNBUFFERED=1 ./venv/bin/python voice_agent.py
```

1. 「Hey Jarvis」と言う → ビープ音が鳴る
2. 日本語で話す（例:「部屋の電気つけて」）
3. 音声で応答が返ってくる

## 技術スタック

- **ウェイクワード**: openWakeWord (ONNX) — "Hey Jarvis" プリセット
- **STT**: faster-whisper (small モデル, CPU)
- **TTS**: macOS `say` コマンド (Kyoko) / 将来的にVOICEVOX
- **LLM**: Claude API (オプション、`ANTHROPIC_API_KEY` 設定時に有効)

## ロードマップ

- [ ] カスタムウェイクワード「OK クロー」の作成
- [ ] Claude API 連携によるLLM応答
- [ ] Windows PC (GPU) への移植 + faster-whisper large-v3-turbo
- [ ] VOICEVOX による日本語TTS
- [ ] Home Assistant + Wyoming Protocol 統合
- [ ] スマートホーム実機連携（SwitchBot等）

## 関連ドキュメント

- [仕様書 (Cosense)](https://scrapbox.io/youkan-brain/%E8%87%AA%E4%BD%9CTTS%E3%81%A7OK%E3%83%BB%E3%82%AF%E3%83%AD%E3%83%BC%E3%80%8C%E9%83%A8%E5%B1%8B%E3%81%AE%E9%9B%BB%E6%B0%97%E3%81%A4%E3%81%91%E3%81%A6%E3%80%8D%E3%82%92%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E3%81%AB%E7%9B%B4%E7%B5%90%E3%81%99%E3%82%8B)
