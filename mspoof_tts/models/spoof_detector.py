"""スプーフ検出器モデルの実装。

Embedding Layer -> Conformer Block x 4 -> Adaptive Pooling -> Classifier Head
の構成で、離散トークン列からreal/fakeの二値分類を行う。

全5モデルでアーキテクチャを共有し、パラメータは各解像度設定ごとに個別に学習する。

ハイパーパラメータ:
    - d_model: 256
    - n_heads: 8
    - n_layers: 4
    - ffn_dim: 1024
    - dropout: 0.1
"""

import math

import torch
import torch.nn as nn

from mspoof_tts.models.conformer import ConformerBlock


class PositionalEncoding(nn.Module):
    """正弦波位置エンコーディング。

    標準的なTransformerスタイルの正弦波位置エンコーディングを適用する。

    Args:
        d_model: モデルの埋め込み次元数。
        max_len: サポートする最大系列長。
        dropout: ドロップアウト率。
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """位置エンコーディングを加算しドロップアウトを適用する。

        Args:
            x: 入力テンソル (batch, seq_len, d_model)。

        Returns:
            位置エンコーディング加算済みテンソル (batch, seq_len, d_model)。
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SpoofDetector(nn.Module):
    """Conformerベースの離散トークンスプーフ検出器。

    離散コーデックトークン列を受け取り、real（本物）またはfake（偽物）の
    確率を出力する二値分類器。

    Args:
        vocab_size: コーデックトークンの語彙サイズ。
        d_model: モデルの埋め込み次元数。
        n_heads: マルチヘッドアテンションのヘッド数。
        n_layers: Conformerブロックの積み重ね数。
        ffn_dim: フィードフォワード層の内部次元数。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # 埋め込み層: 離散トークンIDを連続ベクトルに変換
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 正弦波位置エンコーディング
        self.positional_encoding = PositionalEncoding(d_model, max_len=2048, dropout=dropout)

        # Conformerブロックのスタック
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # 分類ヘッド: Linear -> ReLU -> Dropout -> Linear -> Sigmoid
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """トークン列からreal/fakeの確率を予測する。

        Args:
            token_ids: 離散トークンID列 (batch, seq_len)。

        Returns:
            real確率 (batch,)。値域は [0, 1]。
        """
        # 埋め込み: (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.embedding(token_ids)

        # 位置エンコーディング加算
        x = self.positional_encoding(x)

        # Conformerブロックを順次適用
        for block in self.conformer_blocks:
            x = block(x)

        # 適応プーリング: 時間次元の平均 -> (batch, d_model)
        x = x.mean(dim=1)

        # 分類ヘッド: (batch, d_model) -> (batch, 1) -> (batch,)
        x = self.classifier(x)
        return x.squeeze(-1)
