# 階層的サンプリングと段階的Discriminator枝刈り（Algorithm 2）

本ドキュメントでは、MSpoof-TTS論文のAlgorithm 2「Hierarchical Sampling with Progressive Discriminator Pruning」について整理する。本アルゴリズムはEAS（Entropy-Aware Sampling）の上に構築された階層的スプーフガイド付きデコーディング戦略であり、マルチ解像度スプーフ検出を統合して段階的に候補を枝刈りする。

---

## 1. 入出力

### 1.1 入力

| 記号 | 説明 |
|------|------|
| x | テキスト入力 |
| θ_AR | ARモデル |
| M_10, M_25, M_50, M_50←25, M_50←10 | 5つのdiscriminator |
| L_w | ウォームアップ長 |
| L1, L2, L3 | 各ステージのトークン長 |
| B0, B1, B2 | 各ステージのビームサイズ |
| L_max | 最大生成トークン長 |
| w_50, w_25, w_10 | ランク重み |

### 1.2 出力

| 記号 | 説明 |
|------|------|
| y | 生成されたトークン列 |

---

## 2. アルゴリズムの手順

### 2.1 初期化

```
y ← x（テキストプレフィックスで初期化）
```

### 2.2 ウォームアップフェーズ

EASを用いて L_w トークンを生成し、初期セグメントを安定化させる。ウォームアップ期間中はdiscriminatorによる選択を行わず、ARモデルの自然な生成に委ねる。これにより、後続の階層的枝刈りが安定した文脈の上で動作することを保証する。

### 2.3 反復的デコーディング（while |y| < L_max）

各反復は3つのステージから成り、段階的にビームを絞り込む。

#### Stage 1: 短スパン枝刈り（L1 = 10トークン）

1. EASにより B0（= 8）個の候補ビームを L1トークン分だけ生成する
2. 短スパンdiscriminator M_10 で各候補をスコアリングする
3. 上位 B1（= 5）本のビームを保持し、残りを枝刈りする

#### Stage 2: 中距離枝刈り（L2 = 25トークン）

1. Stage 1で生き残った B1本のビームを、EASにより L2トークンまで延長する
2. 中距離discriminator M_25 で各候補をスコアリングする
3. 上位 B2（= 3）本のビームを保持し、残りを枝刈りする

#### Stage 3: 長スパン延長（L3 = 50トークン）

1. Stage 2で生き残った B2本のビームを、EASにより L3トークンまで延長する

#### 最終選択: Multi-Resolution Rank Aggregation

Stage 3で得られた候補に対し、3つのdiscriminatorでマルチ解像度スコアリングを行い、ランクを集約して最良候補を選出する。

1. **スコアリング**: 各候補 b_i に対し、3つの解像度でスコアを算出する
   ```
   s_50[i] ← M_50(b_i)
   s_25[i] ← M_50←25(b_i)
   s_10[i] ← M_50←10(b_i)
   ```

2. **ランキング**: 各解像度のスコアに基づいてランクを算出する
   ```
   r_50, r_25, r_10 ← Rank(s_50, s_25, s_10)
   ```

3. **重み付き集約**: 各候補の総合ランクを重み付き和で算出する
   ```
   R[i] ← w_50 · r_50[i] + w_25 · r_25[i] + w_10 · r_10[i]
   ```

4. **最良候補の選出**: 総合ランクが最小の候補を選択する
   ```
   b* ← argmin_i R[i]
   ```

5. **結合**: 選出された候補をトークン列に追加する
   ```
   y ← y ∪ b*
   ```

### 2.4 出力

反復が終了したら、最終的なトークン列 y を返す。

---

## 3. ハイパーパラメータ

| パラメータ | 値 | 説明 |
|------------|-----|------|
| L_w（ウォームアップ長） | 20 | discriminator介入前の安定化期間 |
| L1（Stage 1長） | 10 | 短スパン枝刈りのトークン数 |
| L2（Stage 2長） | 25 | 中距離枝刈りのトークン数 |
| L3（Stage 3長） | 50 | 長スパン延長のトークン数 |
| B0（初期ビームサイズ） | 8 | Stage 1の候補数 |
| B1（Stage 1通過後ビームサイズ） | 5 | Stage 2に進む候補数 |
| B2（Stage 2通過後ビームサイズ） | 3 | Stage 3に進む候補数 |
| w_50, w_25, w_10（ランク重み） | 均等 | 各解像度のランクに対する重み |

---

## 4. Discriminatorの役割

本アルゴリズムでは5つのdiscriminatorが異なるステージで使用される。

| Discriminator | 使用ステージ | 入力スパン | 役割 |
|---------------|-------------|-----------|------|
| M_10 | Stage 1 枝刈り | 10トークン | 短スパンでの粗いフィルタリング |
| M_25 | Stage 2 枝刈り | 25トークン | 中距離での精緻なフィルタリング |
| M_50 | 最終選択 | 50トークン | 長スパンでのスコアリング |
| M_50←25 | 最終選択 | 50トークン（25トークン解像度で学習） | クロス解像度スコアリング |
| M_50←10 | 最終選択 | 50トークン（10トークン解像度で学習） | クロス解像度スコアリング |

Stage 1およびStage 2では、対応する解像度のdiscriminatorを用いて候補を枝刈りする。最終選択段階では、M_50、M_50←25、M_50←10の3つのdiscriminatorによるマルチ解像度ランク集約を行い、単一解像度では捉えきれない品質特性を総合的に評価する。

---

## 5. 設計の意図

### 5.1 Coarse-to-fine枝刈り

短スパン（10トークン）→ 中距離（25トークン）→ 長スパン（50トークン）の順にdiscriminatorを適用することで、段階的に候補を絞り込む。初期段階では計算コストの低い短スパンdiscriminatorで明らかに品質の低い候補を排除し、後段では計算コストの高い長スパンdiscriminatorでより精密な判定を行う。この戦略により、デコーディングの安定性と構造的一貫性が改善される。

### 5.2 Rank Aggregation

最終選択において、単一のdiscriminatorスコアではなく、複数解像度のランクを統合する。これにより、特定の解像度に偏らない堅牢な候補選択が実現される。ランクベースの集約はスコアのスケール差に影響されないため、異なる解像度のdiscriminator間でのスコアの直接比較が不要となる。

### 5.3 ベースモデル不変

本アルゴリズムはARバックボーンの再訓練やパラメータ変更を一切必要としない。推論時のデコーディング戦略としてのみ機能するため、既存のTTSモデルに対して追加的な品質向上を非侵襲的に適用できる。

## 実装の対応関係

### コード構成

| 論文の概念 | 実装ファイル | クラス/メソッド |
|-----------|------------|--------------|
| Algorithm 2 全体 | `mspoof_tts/sampling/hierarchical.py` | `HierarchicalDecoder` |
| ウォームアップフェーズ | `mspoof_tts/sampling/hierarchical.py` | `_warmup_phase()` |
| Stage 1 枝刈り | `mspoof_tts/sampling/hierarchical.py` | `_stage1_prune()` |
| Stage 2 枝刈り | `mspoof_tts/sampling/hierarchical.py` | `_stage2_prune()` |
| Stage 3 延長 | `mspoof_tts/sampling/hierarchical.py` | `_stage3_extend()` |
| 最終選択（ランク集約） | `mspoof_tts/sampling/hierarchical.py` | `_final_select()` |
| 反復的デコーディング | `mspoof_tts/sampling/hierarchical.py` | `generate()` |
| nトークン生成ヘルパー | `mspoof_tts/sampling/hierarchical.py` | `_generate_n_tokens()` |
| マルチ解像度検出器 | `mspoof_tts/models/multi_resolution.py` | `MultiResolutionDetector` |
| 推論エントリポイント | `inference.py` | `generate_with_hierarchical()` |

### ビーム状態管理

各ビーム候補は独立したEASサンプラー状態を持つ。Stage 1で `copy.deepcopy(self.eas_sampler)` により B0 個の独立サンプラーをフォークし、各ビームのメモリバッファが独立に動作する。

### generate() の反復ループ

```python
while sequence.size(0) < prefix_len + self.max_length:
    stage1 = self._stage1_prune(model, sequence)     # B0→B1
    stage2 = self._stage2_prune(model, stage1)        # B1→B2
    stage3 = self._stage3_extend(model, stage2)       # B2個を延長
    best = self._final_select(stage3)                 # ランク集約で1つ選択
    sequence = torch.cat([sequence, best])             # L3トークン追加
```

### テスト

- `tests/test_hierarchical.py`: 14テスト（init, warmup, stage1-3, final_select, end-to-end）
