# Entropy-Aware Sampling (EAS)

## 概要

Entropy-Aware Sampling (EAS) は、VALL-E 2 の Repetition-Aware Sampling (RAS) を改良したベースデコーディング戦略である。RAS がヒューリスティックな繰り返しカウントに依存し、高確率トークンを過度に抑制する可能性があるのに対し、EAS はメモリバッファを使用して候補トークンのランク位置と時間的経過を記録する。これにより、分布の多様性を保ちながらエントロピー正則化を導入する。

## Algorithm 1: Entropy-Aware Sampling

### 入力

| 記号 | 説明 |
|------|------|
| x | テキスト |
| θ\_AR | 事前学習済み AR モデル |
| c\_{<t} | プレフィックス（既生成トークン列） |
| v | top-p しきい値 |
| k\_e | クラスタサイズ |
| W | メモリウィンドウサイズ |
| α | ペナルティスケール |
| β | 時間減衰率 |
| γ | クリッピング上限 |

### 出力

生成されたトークン列 c\_{1:T}

### 手順

各タイムステップ t = 1, ..., T について以下を実行する。

#### ステップ 1: 確率分布の取得

```
s_t ← p(· | x, c_{<t}; θ_AR)
```

AR モデルから次トークンの確率分布を取得する。

#### ステップ 2: メモリベースペナルティの計算

メモリ M 内の各エントリ (i, r, a) について、ペナルティ π\_t を計算する。

```
π_t(j) ← min(γ, Σ_{(i,r,a)∈M, i=j} α · (1/(1+r)) · β^a)
```

ペナルティは以下の 3 つの要素から構成される。

- **Inverse rank weighting** `1/(1+r)`: ランク r が高い（上位の）トークンほど大きなペナルティを受ける。ランクが低いトークンに対してはペナルティが緩和される。
- **指数的時間減衰** `β^a`: メモリに記録されてからの経過ステップ a に応じて、ペナルティが指数的に減衰する。古いエントリほど影響が小さくなる。
- **クリッピング** `min(γ, ...)`: ペナルティの上限を γ で制限し、過度な抑制を防止する。

#### ステップ 3: 確率分布の調整

```
s'_t ← s_t - π_t
```

計算されたペナルティを元の確率分布から減算し、調整済み分布を得る。

#### ステップ 4: Nucleus Sampling によるトークン生成

```
c_t ← NucleusSample(s'_t; v)
```

調整された分布に対して nucleus sampling（top-p sampling）を適用し、トークンを生成する。

#### ステップ 5: クラスタの取得

```
K_t ← TopK(s'_t, k_e)
```

調整済み分布からトップ k\_e 個のトークンをクラスタとして取得する。

#### ステップ 6: メモリの更新

```
M ← M ∪ {(K_t[r], r, 0)}  （各クラスタ内トークンについて）
```

クラスタ内の各トークンについて、そのランク情報をメモリに追加する。新規エントリの経過ステップ a は 0 で初期化される。

#### ステップ 7: メモリの aging

ウィンドウサイズ W を超えた古いエントリをメモリから削除する。これにより、メモリの肥大化を防ぎ、直近の文脈のみを考慮したペナルティ計算を実現する。

## ハイパーパラメータ

| パラメータ | 値 | 説明 |
|------------|-----|------|
| top-k | 50 | top-k フィルタリングの k 値 |
| top-p (v) | 0.8 | nucleus sampling のしきい値 |
| temperature | 1.0 | サンプリング温度 |
| cluster\_size (k\_e) | 3 | メモリに記録するトップトークン数 |
| memory\_window (W) | 15 | メモリの保持ステップ数 |
| α | 0.2 | ペナルティスケール |
| β | 0.7 | 時間減衰率 |
| γ | 0.8 | クリッピング上限 |

## RAS との比較

| 項目 | RAS (VALL-E 2) | EAS (MSpoof-TTS) |
|------|----------------|-------------------|
| ペナルティ方式 | ヒューリスティックな繰り返しカウント | ランクベースの重み付け + 指数的時間減衰 |
| window\_size | 25 | 15 (memory\_window) |
| ペナルティパラメータ | repetition\_penalty τ\_r = 0.1 | α = 0.2, β = 0.7, γ = 0.8 |
| 特徴 | 高確率トークンを過度に抑制する可能性がある | クリッピングにより過度な抑制を防止 |
| 正則化 | 繰り返しカウントベース | エントロピー正則化（分布の多様性を保持） |

### RAS の課題と EAS の改善点

RAS はトークンの繰り返し回数を単純にカウントし、一定のペナルティを課す手法である。このアプローチには以下の問題がある。

1. **高確率トークンの過度な抑制**: 文脈上正当な繰り返しであっても一律にペナルティが課される。
2. **時間的文脈の欠如**: 直近の繰り返しと遠い過去の繰り返しを区別しない。

EAS はこれらの課題に対して、以下の改善を導入する。

1. **ランクベースの重み付け**: トークンのランク位置に基づく重み付けにより、きめ細かいペナルティ制御を実現する。
2. **指数的時間減衰**: 時間的経過に応じてペナルティを自然に減衰させることで、直近の文脈をより重視する。
3. **クリッピング機構**: ペナルティ上限の設定により、分布の多様性を維持しつつ過度な抑制を防止する。

## 実装の対応関係

### コード構成

| 論文の概念 | 実装ファイル | クラス/関数 |
|-----------|------------|-----------|
| Algorithm 1 全体 | `mspoof_tts/sampling/eas.py` | `EntropyAwareSampler` |
| メモリバッファ | `mspoof_tts/sampling/eas.py` | `MemoryBuffer` |
| メモリエントリ | `mspoof_tts/sampling/eas.py` | `MemoryEntry` (dataclass) |
| ペナルティ計算 | `mspoof_tts/sampling/eas.py` | `MemoryBuffer.get_penalty()` |
| サンプリングステップ | `mspoof_tts/sampling/eas.py` | `EntropyAwareSampler.sample_step()` |
| 温度スケーリング | `mspoof_tts/sampling/utils.py` | `apply_temperature()` |
| top-kフィルタ | `mspoof_tts/sampling/utils.py` | `top_k_filter()` |
| Nucleus Sampling | `mspoof_tts/sampling/utils.py` | `nucleus_sample()` |
| エントロピー計算 | `mspoof_tts/sampling/utils.py` | `compute_entropy()` |
| ランク計算 | `mspoof_tts/sampling/utils.py` | `rank_tokens()` |

### Algorithm 1 と sample_step() の対応

| アルゴリズムのステップ | 実装箇所 |
|---------------------|---------|
| ステップ 1: 確率分布の取得 | `probs = torch.softmax(logits / self.temperature, dim=-1)` |
| ステップ 2: ペナルティ計算 | `penalty = self.memory.get_penalty(vocab_size, alpha, beta, gamma)` |
| ステップ 3: 条件付き適用 | `if H.item() > entropy_threshold: adjusted_probs = probs - penalty` |
| ステップ 4: Nucleus Sampling | `token_id = nucleus_sample(filtered_logits, self.top_p)` |
| ステップ 5-6: クラスタ取得・メモリ更新 | `top_k_indices` → `MemoryEntry` → `self.memory.add()` |

### ペナルティのベクトル化実装

`MemoryBuffer.get_penalty()` では、ループではなく `scatter_add_` によるベクトル化計算でペナルティを集約している。

### テスト

- `tests/test_eas.py`: 28テスト（temperature, top-k, nucleus, entropy, rank, MemoryBuffer, EntropyAwareSampler）
