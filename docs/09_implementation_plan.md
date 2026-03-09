# 再現実装ロードマップとコード構成計画

## 1. 推奨プロジェクト構成

```
MSpoof-TTS/
├── CLAUDE.md
├── docs/                       # 論文ドキュメント
├── configs/                    # 設定ファイル
│   ├── detector_train.yaml     # スプーフ検出器訓練設定
│   └── inference.yaml          # 推論設定
├── mspoof_tts/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conformer.py        # Conformerブロック実装
│   │   ├── spoof_detector.py   # スプーフ検出器（Embedding + Conformer + AdaptivePool + Classifier）
│   │   └── multi_resolution.py # 5つのマルチ解像度検出器の管理
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── eas.py              # Entropy-Aware Sampling (Algorithm 1)
│   │   ├── hierarchical.py     # Hierarchical Sampling with Progressive Pruning (Algorithm 2)
│   │   └── utils.py            # nucleus sampling, top-k等のユーティリティ
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # 訓練データセット（real/fakeセグメントペア）
│   │   ├── segment.py          # セグメント構築（cropping, skip sampling）
│   │   └── prepare.py          # LibriTTSからの合成データ生成スクリプト
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # WER, SIM, NISQA, MOSNET計算
│       └── evaluate.py         # 評価パイプライン
├── scripts/
│   ├── prepare_data.sh         # データ準備
│   ├── train_detectors.sh      # 5つの検出器の訓練
│   ├── inference.sh            # 推論実行
│   └── evaluate.sh             # 評価実行
├── train.py                    # 検出器訓練エントリポイント
├── inference.py                # 推論エントリポイント
├── requirements.txt
└── setup.py
```

### 1.1 ディレクトリの役割

| ディレクトリ | 役割 |
|---|---|
| `configs/` | YAML形式の設定ファイル。訓練・推論のハイパーパラメータを一元管理する |
| `mspoof_tts/models/` | スプーフ検出器のモデル定義。Conformerブロック、単体検出器、5検出器の統合管理を含む |
| `mspoof_tts/sampling/` | EASおよび階層的デコーディングのサンプリングロジック |
| `mspoof_tts/data/` | データセット構築、セグメント生成、合成データ準備のためのモジュール |
| `mspoof_tts/evaluation/` | 評価指標の計算と評価パイプライン |
| `scripts/` | データ準備から評価までの実行シェルスクリプト |

---

## 2. 実装フェーズ

### Phase 1: 環境構築とベースTTS準備

**目的**: 再現実装の基盤となる環境を整備し、ベースTTSの動作を確認する。

1. **NeuTTSのセットアップ**
   - リポジトリ: https://github.com/neuphonic/neutts
   - README に従いインストールし、サンプル音声の合成が可能な状態にする
2. **NeuCodecトークナイザーの動作確認**
   - 音声波形から離散トークン列へのエンコード、および逆変換（デコード）の動作を検証する
   - トークンの語彙サイズ、フレームレートなどの仕様を把握する
3. **LibriTTSデータセットのダウンロードと前処理**
   - `train-clean-100`, `train-clean-360`, `train-other-500` の取得
   - テキスト・音声ペアの整理と正規化
4. **依存関係の整理**
   - `requirements.txt` の作成
   - PyTorch >= 2.0, torchaudio, transformers 等の互換性確認

**成果物**: NeuTTSで任意のテキストから音声合成が可能な環境、NeuCodecでトークン化・復元が可能な状態。

---

### Phase 2: 合成データ生成

**目的**: スプーフ検出器の訓練に必要な real/fake ペアデータセットを構築する。

1. **合成音声の生成**
   - LibriTTS training split の全発話に対し、NeuTTS で合成を実行
   - 各発話につき3つの合成対を生成（異なるサンプリングパラメータまたはシード）
2. **離散トークン列への変換**
   - 実音声（ground truth）を NeuCodec でエンコードし、離散トークン列を取得
   - 合成音声も同様に NeuCodec でエンコード
3. **データセット構築**
   - 各トークン列に real/fake ラベルを付与
   - 訓練・検証・テスト分割の設定
   - データセットのメタ情報（発話ID、話者ID、テキスト等）の管理

**成果物**: real/fake ラベル付きの離散トークン列データセット（`mspoof_tts/data/prepare.py` で再現可能）。

---

### Phase 3: スプーフ検出器の実装と訓練

**目的**: 5つのマルチ解像度スプーフ検出器を実装し、個別に訓練する。

#### 3.1 Conformerブロックの実装

- 既存ライブラリ（`torchaudio.models` 等）の活用を検討
- 必要に応じて自前実装: Multi-Head Self-Attention + Convolution Module + Feed-Forward の構成
- 実装先: `mspoof_tts/models/conformer.py`

#### 3.2 スプーフ検出器アーキテクチャ

```
Embedding Layer → Conformer Block × 4 → Adaptive Pooling → Classifier Head
```

- **Embedding Layer**: 離散トークンIDを埋め込みベクトルに変換
- **Conformer × 4**: 時系列特徴の抽出
- **Adaptive Pooling**: 可変長入力を固定長表現に変換
- **Classifier Head**: 二値分類（real/fake）
- 実装先: `mspoof_tts/models/spoof_detector.py`

#### 3.3 セグメント構築ロジック

5つの検出器それぞれに対応するセグメント構築方法:

| 検出器 | 方式 | パラメータ |
|---|---|---|
| M_10 | Contiguous cropping | L = 10 |
| M_25 | Contiguous cropping | L = 25 |
| M_50 | Contiguous cropping | L = 50 |
| M_50←25 | Skip sampling | 50トークンスパンから r = 2（25トークン選択） |
| M_50←10 | Skip sampling | 50トークンスパンから r = 5（10トークン選択） |

- 実装先: `mspoof_tts/data/segment.py`

#### 3.4 訓練設定

- **オプティマイザ**: AdamW
- **学習率**: 1e-4
- **損失関数**: Binary Cross-Entropy (BCE)
- **各検出器を個別に訓練**（5回の独立した訓練ループ）

#### 3.5 検出性能の検証

- 評価指標: AUC, Accuracy, Macro-F1
- 検証セットでの性能を確認し、各検出器が十分な識別能力を持つことを担保する

**成果物**: 訓練済みの5つのスプーフ検出器チェックポイント。

---

### Phase 4: Entropy-Aware Sampling (EAS)

**目的**: Algorithm 1 に基づく EAS を実装し、ベースTTSのデコーディングに統合する。

#### 4.1 メモリバッファの実装

生成済みトークンの情報を保持するバッファ:
- **トークンID**: 生成されたトークンの識別子
- **ランク**: 確率分布におけるランク（高確率ほど低ランク）
- **経過時間**: 生成からの経過ステップ数

#### 4.2 ペナルティ計算

```
penalty(t) = α × inverse_rank_weight(t) × exp(-β × elapsed_time(t))
```

- 直近のウィンドウ内で同一トークンの繰り返しにペナルティを付与
- 高ランク（低確率）トークンほどペナルティが大きい
- 時間減衰により、古い出現の影響を減少させる

#### 4.3 ハイパーパラメータ

| パラメータ | 値 | 説明 |
|---|---|---|
| α | 0.2 | ペナルティ強度 |
| β | 0.7 | 時間減衰率 |
| γ | 0.8 | Nucleus sampling の累積確率閾値 |
| k_e | 3 | エントロピー閾値パラメータ |
| W | 15 | メモリバッファのウィンドウサイズ |

#### 4.4 Nucleus Sampling との統合

- 確率分布にペナルティを適用した後、nucleus sampling（top-p）を実行
- エントロピーが高い場合のみペナルティを適用（低エントロピー時はモデルの確信度が高いためペナルティ不要）

**成果物**: `mspoof_tts/sampling/eas.py` — NeuTTSのデコーディングループに統合可能なEASモジュール。

---

### Phase 5: Hierarchical Spoof-Guided Sampling

**目的**: Algorithm 2 に基づく階層的サンプリングを実装し、マルチ解像度検出器を用いた品質誘導デコーディングを実現する。

#### 5.1 ウォームアップフェーズ

- 最初の L_w = 20 トークンは EAS のみで生成
- 検出器に十分なコンテキストを与えるための初期フェーズ

#### 5.2 3ステージビームサーチ

**Stage 1: 短距離候補生成と枝刈り**
- B0 = 8 個の候補シーケンスを EAS で生成（各10トークン延長）
- M_10（10トークン検出器）で各候補をスコアリング
- 上位 B1 = 5 個を選択

**Stage 2: 中距離延長と枝刈り**
- 残った B1 = 5 個の候補をさらに延長（各15トークン追加、合計25トークン）
- M_25（25トークン検出器）でスコアリング
- 上位 B2 = 3 個を選択

**Stage 3: 長距離延長**
- 残った B2 = 3 個の候補をさらに延長（L3 = 50 トークンまで）

#### 5.3 Multi-Resolution Rank Aggregation

最終候補の選択に3つの検出器を使用:

1. **M_50**: 50トークンの連続セグメントでスコアリング
2. **M_50←25**: 50トークンスパンから skip sampling (r=2) でスコアリング
3. **M_50←10**: 50トークンスパンから skip sampling (r=5) でスコアリング

各検出器のスコアをランクに変換し、重み付き集約:

```
final_score = w1 × rank(M_50) + w2 × rank(M_50←25) + w3 × rank(M_50←10)
```

最も良いスコアを持つ候補を選択し、デコーディングを継続する。

#### 5.4 反復デコーディングループ

- 50トークンごとに上記のビームサーチ・枝刈り・集約を繰り返す
- EOS トークンが生成されるまで、または最大長に達するまで継続

**成果物**: `mspoof_tts/sampling/hierarchical.py` — 階層的デコーディングの完全な実装。

---

### Phase 6: 評価

**目的**: 再現実装の出力品質を定量的に評価し、論文の報告値と比較する。

#### 6.1 評価指標

| 指標 | ツール | 説明 |
|---|---|---|
| WER | Whisper-large-v3 | 音声認識精度（低いほど良い） |
| SIM | WavLM-base-plus-sv | 話者類似度（高いほど良い） |
| NISQA | NISQA モデル | 知覚品質スコア |
| MOSNET | MOSNet モデル | 平均オピニオンスコア推定 |

#### 6.2 評価データセット

- **LibriSpeech**: 標準的な読み上げ音声
- **LibriTTS**: TTSの評価で広く使用
- **TwistList**: 追加評価セット

#### 6.3 評価手順

1. 評価セットの各テキストに対し、ベースライン（NeuTTS標準デコーディング）と提案手法（MSpoof-TTS）で音声を合成
2. 各指標を計算し、比較表を作成
3. 論文の Table 2, Table 3 等の再現を試みる

**成果物**: 評価スクリプト（`scripts/evaluate.sh`）および結果の集計。

---

## 3. 主要な依存ライブラリ

| ライブラリ | バージョン要件 | 用途 |
|---|---|---|
| PyTorch | >= 2.0 | モデル構築・訓練の基盤 |
| torchaudio | PyTorchと互換 | 音声処理、Conformer実装の参考 |
| transformers | 最新安定版 | Whisper (WER), WavLM (SIM) |
| NeuTTS / NeuCodec | リポジトリ最新 | ベースTTS、離散トークン化 |
| numpy | >= 1.24 | 数値計算 |
| scipy | >= 1.10 | 信号処理、統計計算 |
| librosa | >= 0.10 | 音声の読み込み・前処理 |
| scikit-learn | >= 1.2 | 評価指標（AUC, F1等） |
| PyYAML | >= 6.0 | 設定ファイル読み込み |
| tensorboard | 最新安定版 | 訓練の可視化（任意） |

---

## 4. 実装上の注意点

### 4.1 NeuTTSのデコーディングへの介入

- NeuTTS の推論インターフェースを精査し、**top-k / top-p / temperature の制御方法**を把握する必要がある
- NeuTTS の内部デコーディングループにアクセスし、**トークンレベルでのサンプリング介入**が可能か確認する
- 介入が困難な場合、NeuTTS のデコーディング部分をフォークして改変する必要がある

### 4.2 Conformer実装の選択

- `torchaudio.models.Conformer` が利用可能であれば活用を検討
- 論文固有のアーキテクチャ詳細（隠れ層サイズ、ヘッド数、カーネルサイズ等）に合わせてカスタマイズが必要な場合は自前実装

### 4.3 GPUメモリ管理

- 推論時に**5つの検出器を同時にロード**する必要がある
- 各検出器のパラメータ数を見積もり、合計メモリ使用量を計算する
- 必要に応じて:
  - 検出器を使用するステージごとに動的ロード/アンロード
  - 半精度（FP16）での推論
  - 複数GPUへの分散配置

### 4.4 ビームサーチの計算コスト

- 50トークンごとに最大 B0 = 8 候補の生成と評価が必要
- バッチ推論を活用して候補生成を並列化する
- 長い発話では反復回数が多くなるため、全体の推論時間を見積もる

### 4.5 データ前処理の再現性

- 合成データ生成時のランダムシードを固定し、再現性を確保する
- NeuTTS のバージョンとチェックポイントを明記し、同一条件での比較を可能にする

---

## 5. 実装優先度とタイムライン（目安）

| フェーズ | 推定期間 | 優先度 | 前提条件 |
|---|---|---|---|
| Phase 1: 環境構築 | 1 週間 | 最高 | なし |
| Phase 2: 合成データ生成 | 1-2 週間 | 最高 | Phase 1 完了 |
| Phase 3: 検出器の実装と訓練 | 2-3 週間 | 高 | Phase 2 完了 |
| Phase 4: EAS | 1 週間 | 高 | Phase 1 完了（Phase 3 と並行可能） |
| Phase 5: 階層的サンプリング | 2 週間 | 高 | Phase 3, 4 完了 |
| Phase 6: 評価 | 1 週間 | 中 | Phase 5 完了 |

**合計推定期間**: 約 8-10 週間

---

## 6. テスト戦略

### 6.1 ユニットテスト

- **Conformerブロック**: 入出力形状の検証、勾配の流れの確認
- **セグメント構築**: contiguous cropping と skip sampling の正確性
- **EASペナルティ計算**: 既知入力に対する期待出力との一致
- **検出器の推論**: ダミー入力での forward pass の成功

### 6.2 統合テスト

- **データパイプライン**: 音声ファイルからトークン列、セグメント、バッチまでの一貫したフロー
- **訓練ループ**: 小規模データでの1エポック訓練が正常に完了すること
- **階層的デコーディング**: 短いテキストでの end-to-end 推論が完了すること

### 6.3 回帰テスト

- 各フェーズの完了時に、以前のフェーズの機能が壊れていないことを確認
- 検出器の精度が訓練後に一定の閾値を超えていることを自動チェック
