# 評価: データセット・指標・実験結果

本ドキュメントでは、MSpoof-TTS論文（Section 3, Section 4）における評価データセット、評価指標、比較手法、および実験結果を整理する。

---

## 1. 評価データセット（Section 3.1）

| データセット | 出典 | 特徴 | 評価目的 |
|------------|------|------|---------|
| **LibriSpeech** [36] | test split | 多様な話者、標準的な音声条件 | 一般的なTTS品質評価 |
| **LibriTTS** [35] | test split | クリーンなTTSスタイル音声 | クリーン条件下での品質評価 |
| **TwistList** [37] | 全体 | 言語学的に構築されたtongue twister（早口言葉）。密なアリテレーション（頭韻）と反復的音素パターンが特徴 | デコーディングの堅牢性に対するストレステスト |

TwistListは、通常のTTS評価では見過ごされがちな反復的・高密度な音素パターンに対する合成能力を検証するために導入されている。

---

## 2. 評価指標

### 2.1 客観的評価指標（Section 3.3）

| 指標 | 方向 | 算出方法 | 評価対象 |
|------|------|---------|---------|
| **WER (Word Error Rate)** | ↓ | Whisper-large-v3 [38] で合成音声を書き起こし、参照テキストとの誤り率を算出 | 音声明瞭性（intelligibility） |
| **SIM (Speaker Similarity)** | ↑ | WavLM-base-plus-sv [39] でL2正規化embedding間のcosine similarityを算出。同一話者の別の実発話との類似度を測定 | 話者同一性の保持 |
| **NISQA** | ↑ | ニューラルMOS推定器 [40] | 知覚品質（perceptual quality） |
| **MOSNET** | ↑ | ニューラルMOS推定器 [41] | 全体的音声品質（overall speech quality） |

使用モデル:
- WER: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- SIM: [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)

### 2.2 スプーフ検出評価指標

| 指標 | 閾値 | 説明 |
|------|------|------|
| **Accuracy** | 0.5 | 二値分類の正解率 |
| **Macro-F1** | 0.5 | クラス平均F1スコア |
| **AUROC** | - | 候補トークン列のランキング性能を測定する主要指標 |

AUROCはthreshold非依存であり、discriminatorが本物と合成のトークン列をどの程度正しくランク付けできるかを評価するため、主要な評価指標として採用されている。

### 2.3 主観評価（Section 4.3）

| 指標 | スケール | 評価対象 |
|------|---------|---------|
| **MOS-N** | 1-5 | 自然性（naturalness） |
| **MOS-Q** | 1-5 | 全体品質（overall quality） |
| **SMOS** | 1-5 | 話者類似度（speaker similarity） |

15名の参加者が、異なる推論方法で生成されたサンプルを評価した。

---

## 3. 比較手法

| 手法 | 略称 | 説明 |
|------|------|------|
| **Original** | - | バニラ top-k sampling（k=50, temperature=1.0） |
| **RAS** | Repetition-Aware Sampling | VALL-E 2由来のサンプリング手法（top-p=0.8, W=25, τ_r=0.1） |
| **EAS** | Entropy-Aware Sampling | 提案手法のベースとなるサンプリング手法 |
| **HierRAS** | Hierarchical RAS | RASベースの階層的デコーディング |
| **HierEAS (MSpoof-TTS)** | Hierarchical EAS | EASベースの階層的デコーディング（提案手法） |

EASとRASはそれぞれ単独のサンプリング手法であり、HierRASとHierEASはそれらを階層的スプーフガイドデコーディング（Algorithm 2）と組み合わせた手法である。

---

## 4. 実験結果

### 4.1 Table 1: LibriSpeech / LibriTTS 結果

| Method | LibriSpeech WER↓ | SIM↑ | NISQA↑ | MOSNET↑ | LibriTTS WER↓ | SIM↑ | NISQA↑ | MOSNET↑ |
|--------|-------------------|------|--------|---------|---------------|------|--------|---------|
| Ground Truth | 0.0337 | 0.915 | 4.620 | 4.5259 | 0.0430 | 0.904 | 4.593 | 4.4766 |
| Original | 0.0694 | 0.894 | 4.462 | 4.3418 | 0.0715 | 0.872 | 4.397 | 4.1879 |
| RAS | 0.0641 | **0.905** | 4.553 | 4.2772 | 0.0657 | 0.880 | 4.425 | 4.2565 |
| EAS | 0.0576 | 0.902 | 4.571 | 4.3298 | 0.0672 | 0.883 | 4.408 | 4.1937 |
| HierRAS | 0.0591 | 0.902 | 4.596 | 4.3486 | **0.0628** | **0.886** | 4.520 | 4.3277 |
| HierEAS (MSpoof-TTS) | **0.0532** | 0.901 | **4.602** | **4.4158** | 0.0633 | 0.883 | **4.562** | **4.3409** |

#### LibriSpeech における分析

- **WER**: HierEAS が最良（0.0532）。Original（0.0694）から23.3%の相対的改善。
- **SIM**: RAS が最良（0.905）。HierEAS（0.901）はわずかに劣るが、Original（0.894）より改善。
- **NISQA**: HierEAS が最良（4.602）。Ground Truth（4.620）に接近。
- **MOSNET**: HierEAS が最良（4.4158）。Ground Truth（4.5259）との差は0.11ポイント。

#### LibriTTS における分析

- **WER**: HierRAS が最良（0.0628）。HierEAS（0.0633）はほぼ同等。
- **SIM**: HierRAS が最良（0.886）。
- **NISQA**: HierEAS が最良（4.562）。Ground Truth（4.593）に接近。
- **MOSNET**: HierEAS が最良（4.3409）。

### 4.2 Table 2: TwistList 結果

| 指標 | 注目すべき結果 |
|------|--------------|
| **NISQA** | HierEAS が最良（4.513） |
| **MOSNET** | HierEAS が最良（3.9802） |
| **WER** | EAS が最良（0.1433） |

TwistListではWERが全体的に高くなる（密なアリテレーションと反復的音素パターンの影響）。EASが最低WERを達成しており、エントロピー正則化がtongue twisterの反復パターンに対して特に効果的であることを示唆している。一方、知覚品質（NISQA, MOSNET）ではHierEASが最良であり、階層的discriminatorガイドが音声品質の向上に寄与している。

### 4.3 主観評価

階層的スプーフガイドデコーディング（HierRAS, HierEAS）は、ベースライン手法（Original, RAS, EAS）と比較して以下の傾向を示した:

- **MOS-N（自然性）**: 階層的手法が一貫して高スコア
- **MOS-Q（全体品質）**: 階層的手法が一貫して高スコア
- **SMOS（話者類似度）**: 全手法で高水準を維持

主観評価の結果は客観指標（NISQA, MOSNET）の傾向と整合しており、discriminatorガイドによる品質向上が人間の知覚においても確認されたことを示す。

---

## 5. 主要な知見

### 5.1 HierEAS（MSpoof-TTS）の総合性能

HierEAS はほぼ全ての指標でベストまたはセカンドベストの性能を達成した。特に知覚品質指標（NISQA, MOSNET）での改善が顕著であり、Ground Truthに近い値を示している。

### 5.2 WERとSIMの改善幅

WERとSIMの改善は控えめである。これは、ベースラインの精度が既に高い水準にあるためである。Originalの時点でWERは0.07前後、SIMは0.87-0.90の範囲にあり、大幅な改善余地が限られている。

### 5.3 知覚品質での顕著な改善

NISQA と MOSNET における改善は一貫して大きい。これは、discriminatorガイドが生成トークン列の構造的一貫性を向上させ、聴覚的に自然な音声の生成を促進していることを示唆する。

### 5.4 Intelligibilityとspeaker identityの保持

Discriminatorガイドにより生成品質を向上させつつ、音声明瞭性（WER）と話者同一性（SIM）を競争力のある水準で保持している。これは、スプーフ検出がトークンの自然性を評価する際に、内容情報や話者特性を損なわないことを意味する。

### 5.5 EASとHierEASの相補性

- **EAS**: エントロピー正則化によりトークンレベルのサンプリング品質を改善。特に反復パターンに対して効果的（TwistListでの最良WER）。
- **HierEAS**: EASの上に階層的discriminator枝刈りを追加することで、知覚品質をさらに向上。両者の組み合わせにより、単独では達成できない総合的な品質向上を実現している。

---

## 実装の対応関係

### コード構成

| 論文の概念 | 実装ファイル | クラス/関数 |
|-----------|------------|-----------|
| WER (Whisper-large-v3) | `mspoof_tts/evaluation/metrics.py` | `compute_wer()` |
| SIM (WavLM-base-plus-sv) | `mspoof_tts/evaluation/metrics.py` | `compute_similarity()` |
| NISQA | `mspoof_tts/evaluation/metrics.py` | `compute_nisqa()` |
| MOSNET | `mspoof_tts/evaluation/metrics.py` | `compute_mosnet()` |
| モデルロード | `mspoof_tts/evaluation/metrics.py` | `load_whisper_model()`, `load_wavlm_model()` |
| 評価パイプライン | `mspoof_tts/evaluation/evaluate.py` | `EvaluationPipeline` |
| 単一発話評価 | `mspoof_tts/evaluation/evaluate.py` | `EvaluationPipeline.evaluate_single()` |
| データセット一括評価 | `mspoof_tts/evaluation/evaluate.py` | `EvaluationPipeline.evaluate_dataset()` |

### 実装の特徴

- **遅延ロード**: Whisper/WavLMモデルは初回使用時に遅延ロードされる（`lazy_load=True`）
- **フォールバック**: NISQA/MOSNetライブラリが利用不可の場合、スペクトル特徴量ベースの簡易推定にフォールバック
- **デュアルCLI**: `--generated_dir`/`--reference_dir` 形式と `--metric`/`--synth-dir`/`--results-dir` 形式の両方をサポート
- **結果マージ**: 既存のJSON結果ファイルがある場合、新しい結果をマージして保存

### スクリプト

| スクリプト | 説明 |
|-----------|------|
| `scripts/evaluate.sh` | 4指標を順次計算（`--metrics wer,sim,nisqa,mosnet`） |

### テスト

- `tests/test_evaluation.py`: 41テスト（WER, SIM, NISQA, MOSNET, EvaluationPipeline, WAVファイル作成, 関数シグネチャ, パイプライン初期化）
