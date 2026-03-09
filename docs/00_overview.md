# MSpoof-TTS: 論文概要

## 基本情報

- **タイトル**: Hierarchical Decoding for Discrete Speech Synthesis with Multi-Resolution Spoof Detection
- **著者**: Junchuan Zhao*¹, Minh Duc Vu*², Ye Wang¹ （*共同第一著者）
- **所属**:
  - ¹ School of Computing, National University of Singapore, Singapore
  - ² Department of Statistics & Data Science, National University of Singapore, Singapore
- **arXiv**: 2603.05373v1 (2026年3月5日公開)
- **分野**: cs.SD（音声・音響処理）

---

## 問題設定

ニューラルコーデック言語モデル（Neural Codec Language Models）は、音声を離散コーデックトークンの列としてモデリングすることで、ゼロショット音声合成（Zero-Shot TTS）の実用的かつ効果的なアプローチとして注目されている。しかし、その推論過程には根本的な不安定性が存在する。

### 具体的な課題

1. **トークンレベルのアーティファクト**: 自己回帰デコーディング中に、離散トークン空間における小さな不整合が蓄積し、聴覚的なアーティファクト（雑音、不自然な遷移）として顕在化する。
2. **分布ドリフト（Distributional Drift）**: 訓練時にはground-truthのコーデック列を条件として次トークンを予測するが、推論時には自身が生成したトークンに基づいて次を予測するため、訓練と推論の間に不一致（exposure bias）が生じる。生成が進むにつれ、合成トークン列が自然な音声コーデックの分布から逸脱していく。
3. **次トークン予測目的関数の限界**: 標準的な次トークン予測の目的関数は、こうした振る舞いを明示的に制約しないため、デコーディング中に問題を検出・修正することが困難である。

### 既存手法の限界

既存の対処法は大きく2つに分類される：

- **追加の学習・最適化を要する手法**: SpeechAlign（選好学習による最適化）、人間フィードバックの活用、微分可能な報酬信号の統合など。効果的だが、再訓練・反復最適化・データキュレーションが必要でコストが高い。
- **デコーディング時の調整**: 繰り返し制御、アライメント制約、サンプリング戦略の修正（VALL-E 2のRAS、ELLA-Vのリオーダリング、VALL-E Rのデコーディング改良など）。適用は容易だが、生成されたトークン列が大域的に整合的で局所的に自然かどうかを明示的に評価するものではない。

---

## 提案手法: MSpoof-TTS

MSpoof-TTSは、**訓練不要の推論フレームワーク（training-free inference framework）** であり、マルチ解像度のスプーフ検出をデコーディングに統合することで、ベースとなるコーデック言語モデルのパラメータを一切変更せずに、音声合成の品質と安定性を向上させる。

ベースTTSシステムとしてNeuTTS（事前訓練済みコーデック言語モデル）を使用し、そのパラメータは完全に固定したまま運用する。

### コンポーネント1: Multi-Resolution Token-Level Spoof Detection

複数の時間解像度でコーデックトークン列を評価し、局所的に不整合または不自然なパターンを検出するフレームワーク。

#### トークンセグメントの構築

2つの相補的な戦略でトークンセグメントを構築し、時間的・構造的な不一致を多角的に捕捉する：

- **連続クロッピング（Contiguous Cropping）**: 長さ L = {10, 25, 50} の連続部分列を抽出。短いセグメントは局所的な遷移ダイナミクスを、長いセグメントは広域の文脈的一貫性と構造的依存関係を捕捉する。
- **スキップサンプリング（Resolution-Based Skip Sampling）**: ダウンサンプリング率 r = {2, 5} で50トークンスパンからサンプリング。元の解像度では見えない粗い粒度の構造的不整合を検出する。

#### スプーフ検出モデル

- **アーキテクチャ**: Conformerベース（d_model=256, 8アテンションヘッド, 4 Transformerレイヤー, FFN次元1024）
- 5つのモデルを訓練: M_50, M_25, M_10（連続クロッピング用）, M_{50←25}, M_{50←10}（スキップサンプリング用）
- アーキテクチャは共有だが、パラメータは各解像度設定ごとに個別に学習
- 二値交差エントロピー（BCE）損失で訓練し、各セグメントが本物（real）か合成（synthetic）かを判定
- Adaptive Poolingで系列表現を集約し、軽量な分類ヘッドで確率を出力

### コンポーネント2: Hierarchical Spoof-Guided Sampling

スプーフ検出器をデコーディングプロセスに統合し、低品質な候補を段階的に枝刈りして再ランキングするデコーディング戦略。2段階で構成される。

#### (a) Entropy-Aware Sampling (EAS)

基本的なトークンサンプリング手法（Algorithm 1）：

- VALL-E 2のRepetition-Aware Sampling（RAS）を改良
- メモリバッファで候補トークンの順位・位置・経過時間を記録
- 逆順位重み付けと指数的時間減衰により、トークンレベルのペナルティを計算
- クリッピング機構で過度なペナルティを防止
- 調整後の分布でNucleus Samplingを実行し、エントロピー正則化と分布の多様性を両立

#### (b) Hierarchical Beam Search with Progressive Discriminator Pruning (Algorithm 2)

階層的ビームサーチ（Algorithm 2）：

1. **ウォームアップ**: 長さ L_w のウォームアップセグメントをEASで生成し、初期デコーディングを安定化
2. **Stage 1**: EASで B_0 個の候補をL_1トークンまで生成 → 短スパン検出器 M_10 で評価 → 上位 B_1 個を保持
3. **Stage 2**: 残った候補をL_2まで延長 → 中スパン検出器 M_25 で評価 → 上位 B_2 個を保持
4. **Stage 3**: 生き残った候補をL_3まで延長 → 完全な候補セグメントを形成
5. **最終選択**: マルチ解像度ランク集約 — 各候補を M_50, M_{50←25}, M_{50←10} で評価し、加重ランク集約で最良候補を選出

この粗→細のマルチ解像度枝刈りにより、ARバックボーンの再訓練なしにデコーディングの安定性と構造的一貫性を向上させる。

---

## 主要な貢献

1. **トークンレベルのスプーフ検出の導入**: 離散コーデック列に特化したマルチ解像度のauthenticity modelingを提案。従来のスプーフ検出は再構成された連続音声波形に対する事後分類が主流であり、離散コーデックトークン列に直接適用しデコーディングをガイドするアプローチは新規である。

2. **推論時デコーディング戦略の開発**: スプーフベースの候補枝刈りと再ランキングにより、ベースモデルの再訓練を一切必要とせずに生成品質を向上。Entropy-Aware Samplingと階層的ビームサーチを組み合わせた推論時フレームワークを構築。

3. **一貫した品質改善の実証**: LibriTTS、LibriSpeech、およびTwistListデータセットで評価を実施し、知覚品質（NISQA, MOSNET）の一貫した改善を示した。知覚品質の向上と同時に、音声明瞭性（WER）および話者類似度（SIM）も競争力のある水準を維持。

---

## 実験結果の概要

### データセット

- **訓練**: LibriTTS training split（約100時間のクリーンな朗読英語音声）
- **評価**: LibriSpeech（テストセット）、LibriTTS（テストセット）、TwistList（早口言葉ベンチマーク）

### 評価指標

- **客観評価**: WER（Whisper-large-v3）、SIM（WavLM話者類似度）、NISQA、MOSNET
- **スプーフ検出評価**: Accuracy, Macro-F1, AUROC
- **主観評価**: MOS-N（自然性）、MOS-Q（品質）、SMOS（話者類似度）、15名の評価者

### 主要結果（Table 1: LibriSpeech / LibriTTS）

| 手法 | LibriSpeech WER↓ | LibriSpeech NISQA↑ | LibriSpeech MOSNET↑ | LibriTTS WER↓ | LibriTTS NISQA↑ | LibriTTS MOSNET↑ |
|---|---|---|---|---|---|---|
| Ground Truth | 0.0337 | 4.620 | 4.5259 | 0.0430 | 4.593 | 4.4766 |
| Original (top-k) | 0.0694 | 4.462 | 4.3418 | 0.0715 | 4.397 | 4.1879 |
| EAS | 0.0576 | 4.571 | 4.3298 | 0.0672 | 4.408 | 4.2565 |
| **HierEAS (MSpoof-TTS)** | **0.0532** | **4.602** | **4.4158** | **0.0633** | **4.562** | **4.3409** |

HierEAS（MSpoof-TTS）は、ほぼ全ての指標でベストまたは2番目のパフォーマンスを達成。特に知覚品質指標（NISQA, MOSNET）で一貫した改善を示した。

### 主観評価

階層的スプーフガイドデコーディング（HierRAS, HierEAS）は、ベースライン手法と比較して自然性（MOS-N）および品質（MOS-Q）で一貫して高いスコアを達成。話者類似度（SMOS）も高水準を維持。

---

## 結論

MSpoof-TTSは、階層的スプーフガイド推論を通じて離散音声合成を改善する訓練不要のフレームワークである。マルチ解像度スプーフ検出器を用いてデコーディングをガイドし、局所的に不整合な音声コーデックトークンパターンを抑制する。事前訓練済みの音声言語モデルを一切修正せずに運用できる点が大きな利点である。

LibriTTSおよびLibriSpeechでの実験により、知覚品質の一貫した改善が確認され、音声明瞭性と話者類似度も競争力のある水準を維持した。さらに、高密度の頭韻や反復的な音素パターンを含むTwistListデータセットでの評価により、困難な音素構造下でも頑健性が実証された。主観的な聴取テストでも、話者のアイデンティティを損なうことなく知覚的自然性の改善が確認され、スプーフガイド推論がニューラルコーデックベースの音声生成を強化する効果的なアプローチであることが示された。

---

## 実装

本プロジェクトでは、上記の論文を以下のコード構成で再現実装している。

### プロジェクト構成

```
mspoof_tts/
├── __init__.py
├── config.py                    # YAML設定ローダー（7つのdataclass）
├── models/
│   ├── conformer.py             # Conformerブロック（FFN, ConvModule, ConformerBlock）
│   ├── spoof_detector.py        # SpoofDetector（Embedding + Conformer + Classifier）
│   └── multi_resolution.py      # MultiResolutionDetector（5検出器の統合管理）
├── sampling/
│   ├── utils.py                 # top-k, nucleus sampling, entropy, rank計算
│   ├── eas.py                   # Entropy-Aware Sampling (Algorithm 1)
│   └── hierarchical.py          # Hierarchical Decoding (Algorithm 2)
├── data/
│   ├── segment.py               # セグメント構築（contiguous cropping, skip sampling）
│   ├── dataset.py               # SpoofDetectionDataset（JSONL, DataLoader）
│   └── prepare.py               # 合成データ生成パイプライン
└── evaluation/
    ├── metrics.py               # WER, SIM, NISQA, MOSNET計算
    └── evaluate.py              # EvaluationPipeline（評価パイプライン）
```

### エントリポイント

| ファイル | 役割 |
|---------|------|
| `train.py` | スプーフ検出器の訓練（5つを個別に訓練） |
| `inference.py` | EAS/階層的デコーディングによる音声合成 |
| `scripts/prepare_data.sh` | データ準備（LibriTTSダウンロード + 合成データ生成） |
| `scripts/train_detectors.sh` | 5検出器の訓練スクリプト |
| `scripts/inference.sh` | 推論実行スクリプト（EAS/hierarchicalモード） |
| `scripts/evaluate.sh` | 評価実行スクリプト |

### テスト

| ファイル | テスト数 | 対象 |
|---------|---------|------|
| `tests/test_mspoof_models.py` | 26 | Conformer, SpoofDetector, MultiResolutionDetector, Segment |
| `tests/test_eas.py` | 28 | EAS, MemoryBuffer, sampling utils |
| `tests/test_hierarchical.py` | 14 | HierarchicalDecoder（各ステージ、E2E） |
| `tests/test_evaluation.py` | 41 | WER, SIM, NISQA, MOSNET, EvaluationPipeline |
