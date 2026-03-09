# 訓練データ準備と訓練手順の詳細

本ドキュメントでは、MSpoof-TTS 論文の Section 3.1 を中心に、訓練データの準備手順およびスプーフ検出器の訓練プロセスについて整理する。

---

## 1. 訓練データセット

### 1.1 ベースデータセット

- **データセット**: LibriTTS training split
- **規模**: 約 100 時間のクリーンな英語読み上げ音声
- **特徴**: マルチスピーカーコーパスであり、慎重なセグメンテーションとノイズフィルタリングが施されている

### 1.2 合成データ生成手順

各 ground-truth 発話に対して、以下の手順で合成データを生成する。

1. **リファレンス選択**: 同一話者の別の発話をリファレンスとして使用する
2. **合成対の生成**: NeuTTS のデフォルト推論スキーム（同一テキスト・同一話者条件）で **3 つの合成対**を生成する
3. **離散トークンへの変換**: 全ての実音声（golden）と合成音声を NeuCodec のデフォルト音声トークナイザーで離散トークン列に変換する

### 1.3 訓練データの構成

上記の手順により、各 ground-truth 発話に対して 3 つの合成対が得られる。これを以下のように二値分類データセットとして構成する。

| カテゴリ | ラベル | 説明 |
|----------|--------|------|
| Golden（実音声由来トークン） | Real（本物） | ground-truth 発話から変換された離散トークン列 |
| 合成音声由来トークン | Fake（偽物） | NeuTTS で合成された音声から変換された離散トークン列 |

---

## 2. セグメント構築

訓練データから実際にモデルへ入力するセグメントを構築するため、2 つの戦略が用いられる。

### 2.1 Contiguous Cropping（連続クロッピング）

発話レベルのトークン列からランダムに長さ L の連続部分列を抽出する方式。

- **L = 50**: M_50 の訓練に使用
- **L = 25**: M_25 の訓練に使用
- **L = 10**: M_10 の訓練に使用

### 2.2 Resolution-based Skip Sampling（スキップサンプリング）

50 トークンスパンから一定間隔 r でサンプリングし、ダウンサンプリングされたセグメントを得る方式。

- **r = 2**: 50 トークンスパンから 1 つおきにサンプリング → 25 トークンを取得（M_50←25 の訓練に使用）
- **r = 5**: 50 トークンスパンから 4 つおきにサンプリング → 10 トークンを取得（M_50←10 の訓練に使用）

各セグメントには元の発話のラベル（real / fake）がそのまま付与される。

---

## 3. スプーフ検出器の訓練

### 3.1 モデル構成

5 つのスプーフ検出モデルを**個別に**訓練する。アーキテクチャは全モデルで共有構造だが、パラメータは各解像度設定ごとに独立して学習される。

| モデル名 | セグメント戦略 | 入力トークン数 | 説明 |
|----------|----------------|----------------|------|
| M_50 | Contiguous Cropping | 50 | 長さ 50 の連続部分列 |
| M_25 | Contiguous Cropping | 25 | 長さ 25 の連続部分列 |
| M_10 | Contiguous Cropping | 10 | 長さ 10 の連続部分列 |
| M_50←25 | Skip Sampling | 25 | 50 トークンスパンから r=2 でダウンサンプリング |
| M_50←10 | Skip Sampling | 10 | 50 トークンスパンから r=5 でダウンサンプリング |

### 3.2 アーキテクチャ（Conformer ベース）

| パラメータ | 値 |
|------------|-----|
| d_model（モデル次元） | 256 |
| attention heads（注意ヘッド数） | 8 |
| Transformer layers（層数） | 4 |
| feedforward dimension（FFN 次元） | 1024 |
| dropout | 0.1 |

### 3.3 訓練ハイパーパラメータ

| 項目 | 設定値 |
|------|--------|
| 最適化手法 | AdamW |
| 学習率 | 1 × 10⁻⁴ |
| weight decay | 1 × 10⁻⁴ |
| 損失関数 | Binary Cross-Entropy (BCE) |
| dropout | 0.1 |
| GPU | NVIDIA L40S × 1 |

---

## 4. ベース TTS モデル（NeuTTS）

### 4.1 概要

- **名称**: NeuTTS
- **リポジトリ**: https://github.com/neuphonic/neutts
- **種別**: 事前学習済みコーデックベース TTS システム
- **パラメータ**: 全プロセスを通じて**固定**（frozen）

### 4.2 訓練時と推論時の動作の違い

| フェーズ | 条件付け対象 | 説明 |
|----------|--------------|------|
| 訓練時 | ground-truth コーデック列 | 正解のコーデックトークン列に条件付けて次トークンを予測する（teacher forcing） |
| 推論時 | 以前に生成されたトークン | 自己回帰的に、モデル自身が生成したトークンに条件付けて次トークンを予測する |

この訓練と推論の間の条件付け対象の不一致が **exposure bias**（discrepancy）として知られる現象を引き起こす。MSpoof-TTS は、この exposure bias に起因する合成トークンと実トークンの分布ギャップを検出に利用する。

---

## 5. 重要な設計上の注意点

### 5.1 Training-Free Inference Framework

MSpoof-TTS のフレームワークにおいて、TTS モデル自体は**再訓練しない**。これは以下の理由による。

- TTS モデル（NeuTTS）は事前学習済みの状態をそのまま使用する
- 新たに訓練するのはスプーフ検出器のみである
- 検出器は TTS バックボーンから独立して最適化される

### 5.2 検出器と TTS の分離

スプーフ検出器の訓練は TTS モデルのパラメータに一切影響を与えない。この設計により、以下の利点が得られる。

- **モジュール性**: 検出器のみを独立して改良・差し替えできる
- **汎用性**: 異なる TTS システムに対しても、検出器を再訓練するだけで適用可能
- **効率性**: TTS モデル全体の再訓練が不要であり、計算コストが大幅に削減される

### 5.3 訓練パイプラインの全体像

```
LibriTTS (ground-truth 発話)
    │
    ├─→ NeuTTS (frozen) で合成音声を生成（各発話に対して 3 つの合成対）
    │
    ├─→ Golden 音声 ──→ NeuCodec ──→ 離散トークン列 (Real ラベル)
    │
    └─→ 合成音声 ────→ NeuCodec ──→ 離散トークン列 (Fake ラベル)
                                          │
                                          ▼
                              セグメント構築（Contiguous Cropping / Skip Sampling）
                                          │
                                          ▼
                              5 つの Conformer 検出器を個別に訓練
                              (M_50, M_25, M_10, M_50←25, M_50←10)
```

---

## 6. 別PCでの訓練ワークフロー

スプーフ検出器の訓練はGPU（NVIDIA L40S等）を搭載した別のPCで行い、訓練済みチェックポイントを開発マシンに転送して推論に使用する。

### 6.1 全体の流れ

```
[開発PC (macOS)]                         [訓練PC (Linux + GPU)]
     │                                         │
     ├── コード開発・テスト                      │
     │                                         │
     ├──── リポジトリを転送 ──────────────────→ │
     │                                         ├── 環境構築 (uv sync)
     │                                         ├── Stage 1: LibriTTSダウンロード
     │                                         ├── Stage 2: 合成データ生成
     │                                         ├── Stage 3: 検出器訓練 (×5)
     │                                         │
     │ ←──── チェックポイント (.pt) を転送 ────┤
     │                                         │
     ├── 推論 (EAS / hierarchical)              │
     └── 評価                                   │
```

### 6.2 訓練PCへの転送ファイル

リポジトリ全体をクローンまたはコピーする。最低限必要なファイル:

```bash
# リポジトリごとクローン（推奨）
git clone <repository-url>
cd NeuTTS
git checkout feature/mspoof-tts

# または必要ファイルだけ転送する場合:
# 訓練に必要なファイル一覧
train.py                          # 訓練エントリポイント
pyproject.toml                    # 依存関係定義
uv.lock                          # ロックファイル
configs/detector_train.yaml       # 訓練設定
scripts/prepare_data.sh           # データ準備スクリプト
scripts/train_detectors.sh        # 訓練スクリプト
mspoof_tts/                       # パッケージ全体
```

### 6.3 訓練PCでの環境構築

```bash
# uv のインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync

# GPU が認識されていることを確認
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 6.4 訓練の実行手順

```bash
# Step 1: データ準備（LibriTTSダウンロード + 合成データ生成）
bash scripts/prepare_data.sh --device gpu

# Step 2: 5つのスプーフ検出器を順次訓練
bash scripts/train_detectors.sh --epochs 100 --batch-size 64 --gpu 0

# 特定の検出器だけ訓練する場合
bash scripts/train_detectors.sh --detectors M_50,M_25 --gpu 0
```

訓練完了後、チェックポイントは以下に保存される:

```
checkpoints/detectors/
├── M_50/
│   └── M_50.pt           # best checkpoint
├── M_25/
│   └── M_25.pt
├── M_10/
│   └── M_10.pt
├── M_50_25/
│   └── M_50_25.pt
└── M_50_10/
    └── M_50_10.pt
```

### 6.5 チェックポイントの転送

訓練完了後、5つの `.pt` ファイルを開発PCに転送する:

```bash
# 訓練PCから開発PCへ転送（scp の例）
scp -r user@training-pc:~/NeuTTS/checkpoints/detectors/ ./checkpoints/detectors/

# または rsync で
rsync -avz user@training-pc:~/NeuTTS/checkpoints/detectors/ ./checkpoints/detectors/
```

### 6.6 開発PCでの推論

転送したチェックポイントを使って推論を実行する:

```bash
# EASモード（検出器不要）
bash scripts/inference.sh --input "Hello world" --reference samples/dave.wav \
    --output output_eas.wav --mode eas

# 階層的デコーディングモード（検出器チェックポイントが必要）
bash scripts/inference.sh --input "Hello world" --reference samples/dave.wav \
    --output output_hier.wav --mode hierarchical \
    --checkpoint-dir checkpoints/detectors
```

チェックポイントのパスは `configs/inference.yaml` の `detector_checkpoints` セクションでも設定可能。

### 6.7 訓練PC の推奨スペック

| 項目 | 推奨 | 最低限 |
|------|------|--------|
| GPU | NVIDIA L40S (48GB) | NVIDIA GPU (16GB VRAM以上) |
| RAM | 64GB以上 | 32GB |
| ストレージ | 200GB以上（LibriTTS + 合成データ） | 100GB |
| OS | Ubuntu 22.04+ | Linux (CUDA対応) |

---

## 実装の対応関係

### コード構成

| 論文の概念 | 実装ファイル | クラス/関数 |
|-----------|------------|-----------|
| 訓練パイプライン | `train.py` | メインスクリプト |
| データ準備 | `mspoof_tts/data/prepare.py` | `prepare_dataset()` |
| 音声→トークン変換 | `mspoof_tts/data/prepare.py` | `encode_audio_to_tokens()` |
| 合成音声生成 | `mspoof_tts/data/prepare.py` | `synthesize_utterance()` |
| 話者ごとの発話収集 | `mspoof_tts/data/prepare.py` | `_collect_speaker_utterances()` |
| データ分割 | `mspoof_tts/data/prepare.py` | `_assign_splits()` (90/5/5) |
| セグメント構築 | `mspoof_tts/data/segment.py` | `SegmentExtractor` |
| 訓練データセット | `mspoof_tts/data/dataset.py` | `SpoofDetectionDataset` |
| DataLoader作成 | `mspoof_tts/data/dataset.py` | `SpoofDetectionDataset.create_dataloader()` |
| 検出器モデル | `mspoof_tts/models/spoof_detector.py` | `SpoofDetector` |
| 訓練設定 | `configs/detector_train.yaml` | YAML設定ファイル |
| 設定ローダー | `mspoof_tts/config.py` | `load_training_config()` |

### スクリプト

| スクリプト | 説明 |
|-----------|------|
| `scripts/prepare_data.sh` | LibriTTSダウンロード + 合成データ生成の2段階パイプライン |
| `scripts/train_detectors.sh` | 5検出器を順次訓練（各検出器のセグメントパラメータを自動設定） |

### train.py の訓練ループ

```
シード設定 → SegmentExtractor → SpoofDetectionDataset → SpoofDetector(vocab_size=65536)
→ AdamW + BCELoss + CosineAnnealingLR → train/val loop (AUC/F1/Accuracy)
→ チェックポイント保存 (best + periodic + final as {name}.pt)
```

### データ形式

- 訓練データ: JSONL形式（`prepare.py` が生成）
- 各行: `{"utterance_id": ..., "speaker_id": ..., "tokens": [...], "label": "real"/"fake", "split": "train"/"val"/"test"}`
- 分割比率: 話者単位で 90% train / 5% val / 5% test
