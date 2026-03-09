# ベースTTSシステム (NeuTTS) とコーデックの詳細

## NeuTTS: 事前学習済みコーデックベースTTSシステム

MSpoof-TTS のベースジェネレーターとして使用される事前学習済みコーデックベースTTSシステムである。フレームワーク全体を通じてパラメータは固定され、一切の再訓練・微調整は行われない。

- **リポジトリ**: https://github.com/neuphonic/neutts

### 動作原理

NeuTTS は音声を離散コーデックトークンの列としてモデル化し、自己回帰的にトークンを生成する。

- **訓練時**: ground-truth のコーデックトークン列に条件付けて次トークンを予測する（teacher forcing）
- **推論時**: 以前に自身が生成したトークンに条件付けて次トークンを予測する

この訓練時と推論時の条件付けの違いが、training-inference discrepancy（exposure bias）の原因となる。

### ゼロショット音声合成

NeuTTS はゼロショット音声合成に対応しており、テキスト入力とリファレンス音声から任意の話者の音声を生成できる。

- リファレンス発話によって話者特性を条件付け
- デフォルト推論スキーム: top-k sampling (k=50, temperature=1.0)

---

## 音声コーデック: NeuCodec

NeuCodec はデフォルトの音声トークナイザーであり、以下の 2 つの変換を担う。

1. **エンコード**: 音声波形 → 離散コーデックトークン列
2. **デコード**: 離散コーデックトークン列 → 音声波形

MSpoof-TTS の訓練データもデフォルトの NeuCodec でトークン化される。スプーフ検出モデルの訓練においては、golden（実音声由来）および合成音声の両方が NeuCodec によって離散トークンに変換され、トークンレベルでの二値分類の対象となる。

---

## Exposure Bias 問題

コーデックベース TTS における根本的な課題であり、MSpoof-TTS が解決を目指す中心的な問題である。

### 原因

| フェーズ | 条件付けの対象 |
|----------|---------------|
| 訓練時 | ground-truth コーデックトークン列（teacher forcing） |
| 推論時 | 自己生成コーデックトークン列 |

訓練時には常に正しいトークン列を参照して次トークンを予測するが、推論時には自身の過去の出力（誤りを含み得る）に基づいて生成を継続する。この不一致により、生成が進むにつれて合成トークン列が自然なコーデック分布から徐々に逸脱する。

### 分布ギャップの可視化

Figure 2 の t-SNE 可視化により、golden トークン列と synthetic トークン列の間の分布ギャップが確認されている。重要な知見として、この分布ギャップは発話全体レベルだけでなく、複数のセグメント長（L = 50, 25, 10）の全解像度において存在する。この事実が、MSpoof-TTS のマルチ解像度スプーフ検出アプローチの理論的根拠となっている。

---

## 関連するコーデックベースTTSシステム

論文では、以下のコーデックベース TTS システムが関連研究として言及されている。

| システム | 特徴 |
|----------|------|
| NaturalSpeech 3 [2] | Factored codec + diffusion |
| CosyVoice [3] | Supervised semantic tokens |
| LLasa [4] | Llama-based |
| F5-TTS [8] | Flow matching |
| VALL-E 2 [16] | Repetition-aware sampling |
| ELLA-V [7] | Alignment-guided sequence reordering |
| VALL-E R [17] | Monotonic alignment |

これらのシステムは、コーデックベース TTS の品質向上に対してそれぞれ異なるアプローチを採用している。特に VALL-E 2 の Repetition-Aware Sampling (RAS) は、MSpoof-TTS の Entropy-Aware Sampling (EAS) の直接的な改良元となっている。

---

## MSpoof-TTS の位置付け

MSpoof-TTS は、上記の関連システムとは根本的に異なるアプローチを採用する。

### 設計原則

1. **ベース TTS モデルを変更しない**: NeuTTS のパラメータは完全に固定され、推論時のデコーディングのみを改善する
2. **Training-free**: TTS モデル自体に対する追加の訓練、微調整、強化学習は一切不要（スプーフ検出器の訓練は必要だが、TTS モデルのパラメータは不変）
3. **汎用性**: 特定の TTS アーキテクチャに依存せず、任意のコーデックベース TTS に適用可能な汎用フレームワークとして設計されている

### 既存手法との対比

| 分類 | 手法例 | MSpoof-TTS との違い |
|------|--------|-------------------|
| 追加学習を要する手法 | SpeechAlign, 選好学習 | MSpoof-TTS は TTS モデルの再訓練不要 |
| デコーディング調整 | RAS, ELLA-V リオーダリング | MSpoof-TTS はスプーフ検出によるマルチ解像度評価を導入 |
| コーデック設計の改良 | NaturalSpeech 3, CosyVoice | MSpoof-TTS は既存コーデックをそのまま使用 |

MSpoof-TTS の核心的な貢献は、スプーフ検出という従来は事後的な音声分類に使われていた技術を、デコーディング過程にリアルタイムで統合し、生成中のトークン列の品質を継続的に評価・制御する点にある。

---

## 実装の対応関係

### コード構成

| 論文の概念 | 実装ファイル | クラス/関数 |
|-----------|------------|-----------|
| NeuTTSによる推論 | `inference.py` | `generate_with_eas()`, `generate_with_hierarchical()` |
| カスタムARループ | `inference.py` | `generate_with_eas()` 内のKV-cache付きループ |
| NeuCodecエンコード | `mspoof_tts/data/prepare.py` | `encode_audio_to_tokens()` |
| プロンプト構築 | `inference.py` | `_apply_chat_template()` |
| トークン→音声変換 | `inference.py` | `tokens_to_speech_string()` |
| 推論設定 | `configs/inference.yaml` | EAS/階層的デコーディングのハイパーパラメータ |
| 設定ローダー | `mspoof_tts/config.py` | `load_inference_config()` |

### カスタム生成ループの必要性

NeuTTSの標準 `generate()` メソッドは生のlogitsを外部に公開しないため、EASやHierarchical Decodingでトークンレベルのサンプリング制御を行うには、カスタムの自己回帰ループが必要である。`inference.py` では以下のアプローチを採用：

1. `backbone(input_ids)` でraw logitsを直接取得
2. KV-cacheを活用して効率的に逐次生成
3. EASサンプラーの `sample_step(logits)` でトークンを選択

### スクリプト

| スクリプト | 説明 |
|-----------|------|
| `scripts/inference.sh` | EAS/hierarchicalモードの切り替え、全ハイパーパラメータの引数/環境変数指定 |
