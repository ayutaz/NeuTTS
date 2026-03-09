# Conformerベース離散トークンスプーフ検出器のアーキテクチャ詳細

本ドキュメントでは、MSpoof-TTS論文のFigure 1(b)に示されるConformerベースの離散トークンスプーフ検出器のアーキテクチャについて整理する。

---

## 1. モデル全体の構成（上から下へ）

モデルは以下の5つの主要コンポーネントから構成される。

### 1.1 Embedding層

離散コーデックトークンをembedding vectorに変換する層。入力となる離散トークン列を連続的なベクトル空間に射影し、後続のConformerブロックで処理可能な表現を得る。

### 1.2 Positional Encoding

Conformerの標準的な手法に従い、embedding vectorに位置情報を付加する。これにより、トークンの順序関係がモデルに伝達される。

### 1.3 Conformer Blocks（4層スタック）

Conformer [34] (Gulati et al., 2020) に基づくブロックを4層積み重ねる。各ブロックは以下のサブモジュールで構成される。

| 順序 | サブモジュール | 説明 |
|------|----------------|------|
| 1 | LayerNorm | 入力の正規化 |
| 2 | Feed Forward (1/2) | 前半のフィードフォワードモジュール |
| 3 | Convolution Module | 畳み込みモジュール（局所的なトークンパターンの捕捉） |
| 4 | Multi-Head Self-Attention (MHSA) | 8ヘッドの自己注意機構（長距離の構造的依存性の捕捉） |
| 5 | Feed Forward (1/2) | 後半のフィードフォワードモジュール |

Conformerの設計思想として、畳み込みモジュールとself-attentionの組み合わせにより、局所的な相関と長距離依存性を同時に捉えることが可能となる。

### 1.4 Adaptive Pooling

可変長のトークンシーケンスを固定長の表現に集約する。具体的には、mean pooling（temporal次元）によってトークンembeddingを集約し、複数の時間解像度での表現を取得する。

### 1.5 Classifier Head

最終的なreal/fake判定を行う分類ヘッド。以下の層で構成される。

```
Linear → ReLU → Dropout(0.1) → Linear → p ∈ [0, 1]
```

出力 `p` はreal（本物）またはfake（偽物）である確率を表す。

---

## 2. ハイパーパラメータ

| パラメータ | 値 |
|------------|-----|
| d_model（モデル次元） | 256 |
| attention heads（注意ヘッド数） | 8 |
| Transformer layers（層数） | 4 |
| feedforward dimension（FFN次元） | 1024 |
| dropout | 0.1 |

---

## 3. 訓練設定

| 項目 | 設定値 |
|------|--------|
| 最適化手法 | AdamW |
| 学習率 | 1 × 10⁻⁴ |
| weight decay | 1 × 10⁻⁴ |
| 損失関数 | Binary Cross-Entropy (BCE) |
| GPU | NVIDIA L40S × 1 |

---

## 4. 設計上の特徴

### 4.1 アーキテクチャの共有とパラメータの個別学習

モデルのアーキテクチャは全5モデルで共有されるが、パラメータは各解像度設定ごとに個別に学習される。これにより、異なる時間解像度に対して同一の構造を用いつつ、各解像度に特化した特徴抽出が可能となる。

### 4.2 Conformer採用の理由

Conformerが採用された理由は、以下の2つの特性を同時に活用できる点にある。

- **畳み込みモジュール**: 局所的なトークンパターン（近接するトークン間の関係性）を効率的に捕捉する
- **Self-Attention**: 長距離の構造的依存性（離れた位置にあるトークン間の関係性）を捕捉する

この組み合わせにより、離散トークン列に含まれるスプーフィングの手がかりを、局所的・大域的の両方の観点から検出することが可能となる。

### 4.3 時間解像度に基づくプーリング

Mean pooling（temporal次元）を用いてトークンembeddingを集約する。これにより、可変長の入力シーケンスから固定長の表現を得るとともに、複数の時間解像度における特徴を統合的に扱うことができる。

## 実装の対応関係

### コード構成

| 論文のコンポーネント | 実装ファイル | クラス |
|-------------------|------------|-------|
| Feed Forward Module | `mspoof_tts/models/conformer.py` | `FeedForwardModule` |
| Convolution Module | `mspoof_tts/models/conformer.py` | `ConvolutionModule` |
| Conformer Block | `mspoof_tts/models/conformer.py` | `ConformerBlock` |
| Embedding層 | `mspoof_tts/models/spoof_detector.py` | `nn.Embedding` in `SpoofDetector` |
| Positional Encoding | `mspoof_tts/models/spoof_detector.py` | `PositionalEncoding` |
| Adaptive Pooling | `mspoof_tts/models/spoof_detector.py` | Mean pooling in `SpoofDetector.forward()` |
| Classifier Head | `mspoof_tts/models/spoof_detector.py` | `SpoofDetector.classifier` |

### Conformer Block の実装構造

```python
# ConformerBlock (Macaron-style)
output = x
output = output + 0.5 * self.ff1(output)      # 前半FFN (1/2)
output = output + self.mhsa(output)            # Multi-Head Self-Attention
output = output + self.conv(output)            # Convolution Module
output = output + 0.5 * self.ff2(output)       # 後半FFN (1/2)
output = self.layer_norm(output)               # 最終LayerNorm
```

### ConvolutionModule の実装構造

```python
# ConvolutionModule
output = self.layer_norm(x)
output = self.pointwise_conv1(output)          # Pointwise Conv (拡張)
output = self.glu(output)                      # GLU活性化
output = self.depthwise_conv(output)           # Depthwise Conv
output = self.batch_norm(output)               # BatchNorm
output = self.activation(output)               # SiLU活性化
output = self.pointwise_conv2(output)          # Pointwise Conv (縮小)
output = self.dropout(output)
```
