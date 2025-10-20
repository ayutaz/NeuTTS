# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

NeuTTS Airは、オンデバイスで動作する超リアルなTTS（Text-to-Speech）モデルです。0.5B LLMバックボーン（Qwen 0.5B）をベースに、3秒の音声サンプルから瞬時に声をクローンできる機能を持っています。

## 主要コマンド

### 依存関係のインストール

```bash
# eSpeak（必須の依存関係）のインストール
# Mac OS
brew install espeak

# Ubuntu/Debian
sudo apt install espeak

# Python依存関係のインストール
pip install -r requirements.txt

# オプション: GGUF モデル用
pip install llama-cpp-python

# オプション: ONNX デコーダ用
pip install onnxruntime

# オプション: ストリーミング例用
pip install pyaudio
```

### 基本的な推論の実行

```bash
# 標準的な例
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio samples/dave.wav \
  --ref_text samples/dave.txt

# GGUF バックボーンを使用
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio samples/dave.wav \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf

# ストリーミング推論（GGUF のみ）
python -m examples.basic_streaming_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf

# 最小レイテンシー例（ONNX デコーダ + 事前エンコード済み参照）
python -m examples.onnx_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf
```

### ファインチューニング

```bash
# ファインチューニングスクリプトの実行
python examples/finetune.py examples/finetune_config.yaml
```

### 開発用コマンド

```bash
# pre-commit フックのインストール（コントリビューター向け）
pip install pre-commit
pre-commit install
```

## アーキテクチャ概要

### コアコンポーネント

1. **NeuTTSAir クラス** (`neuttsair/neutts.py`):
   - メインの TTS インターフェース
   - バックボーンモデル（Transformer LLM）とコーデック（NeuCodec）を管理
   - PyTorch、GGUF（llama-cpp-python）、ONNX の 3 つのバックエンドをサポート

2. **バックボーンモデル**:
   - Qwen 0.5B ベースの因果的言語モデル
   - 音素シーケンスを音声トークン（speech tokens）に変換
   - 3 つのフォーマット: HuggingFace Transformers（標準）、GGUF（高速・低リソース）、ONNX（最小レイテンシー）

3. **音声コーデック（NeuCodec）**:
   - 50Hz のニューラル音声コーデック
   - 音声を離散トークン（0-65535）にエンコード/デコード
   - 3 つのバリエーション: `neucodec`（標準）、`distill-neucodec`（軽量）、`neucodec-onnx-decoder`（推論専用）

4. **音素化（Phonemization）**:
   - eSpeak バックエンドを使用してテキストを音素に変換
   - 英語のみサポート（`en-us`）
   - ストレス付き音素表記を使用

### データフロー

```
入力テキスト → 音素化 → トークン化 → バックボーン生成 → 音声トークン → コーデックデコード → 音声波形 → ウォーターマーク → 出力
                                    ↑
                                参照音声（エンコード済み） + 参照テキスト
```

### チャットテンプレート形式

モデルは特殊なチャット形式を使用します:

```
user: Convert the text to speech:<|TEXT_PROMPT_START|>{phonemized_text}<|TEXT_PROMPT_END|>
assistant:<|SPEECH_GENERATION_START|>{speech_tokens}<|SPEECH_GENERATION_END|>
```

音声トークンは `<|speech_0|>` から `<|speech_65535|>` の形式で表現されます。

### ストリーミング推論

- GGUF バックボーンでのみサポート
- チャンクサイズ: 25 フレーム（`streaming_frames_per_chunk`）
- ルックフォワード: 5 フレーム、ルックバック: 50 フレーム
- 線形オーバーラップ加算（`_linear_overlap_add`）を使用してスムーズな再生を実現

### ファインチューニングアーキテクチャ

- HuggingFace Trainer を使用
- Emilia-YODAS データセット（NeuCodec でエンコード済み）がサンプルとして提供
- 学習率: 1e-5 〜 4e-5 が推奨
- ラベルマスキング: `<|SPEECH_GENERATION_START|>` 以降のみを学習対象とする

### パフォーマンス最適化

レイテンシーを最小化するには:
1. GGUF バックボーンを使用（`neutts-air-q4-gguf` または `neutts-air-q8-gguf`）
2. 参照音声を事前にエンコード（`encode_reference()` を使用し `.pt` ファイルとして保存）
3. ONNX コーデックデコーダを使用（`neucodec-onnx-decoder`）

### 重要な実装詳細

- **コンテキストウィンドウ**: 2048 トークン（約 30 秒の音声を処理可能）
- **サンプリングレート**: 24,000 Hz
- **ホップ長**: 480 サンプル
- **ウォーターマーキング**: 全ての出力に Perth Watermarker が自動適用される
- **参照音声要件**: モノラル、16-44kHz、3-15秒、クリーンな音声

## 注意事項

### macOS での eSpeak 設定

macOS ユーザーは、`neutts.py` の先頭に以下を追加する必要がある場合があります:

```python
from phonemizer.backend.espeak.wrapper import EspeakWrapper
_ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)
```

### Windows での環境変数設定

```pwsh
$env:PHONEMIZER_ESPEAK_LIBRARY = "c:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:PHONEMIZER_ESPEAK_PATH = "c:\Program Files\eSpeak NG"
```

### GPU サポート

- llama-cpp-python: CUDA や MPS サポートについては https://pypi.org/project/llama-cpp-python/ を参照
- PyTorch: `backbone_device="cuda"` または `codec_device="cuda"` を指定
