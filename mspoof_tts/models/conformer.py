"""Conformerブロックの実装。

Gulati et al. (2020) に基づくConformerブロックを実装する。
各ブロックは以下のサブモジュールで構成される:
  1. LayerNorm
  2. Feed Forward (前半)
  3. Convolution Module
  4. Multi-Head Self-Attention (MHSA)
  5. Feed Forward (後半)

畳み込みモジュールにより局所的なトークンパターンを、
Self-Attentionにより長距離の構造的依存性を同時に捕捉する。
"""

import torch
import torch.nn as nn


class FeedForwardModule(nn.Module):
    """Conformer内のフィードフォワードモジュール。

    Linear -> Activation -> Dropout -> Linear -> Dropout の構成。

    Args:
        d_model: モデル次元数。
        ffn_dim: フィードフォワード層の内部次元数。
        dropout: ドロップアウト率。
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        raise NotImplementedError("FeedForwardModule の実装が必要")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フィードフォワード変換を適用する。

        Args:
            x: 入力テンソル (batch, seq_len, d_model)。

        Returns:
            出力テンソル (batch, seq_len, d_model)。
        """
        raise NotImplementedError


class ConvolutionModule(nn.Module):
    """Conformer内の畳み込みモジュール。

    局所的なトークンパターンを捕捉するための1次元畳み込みモジュール。
    Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> Activation -> Pointwise Conv の構成。

    Args:
        d_model: モデル次元数。
        kernel_size: 畳み込みカーネルサイズ。
        dropout: ドロップアウト率。
    """

    def __init__(
        self, d_model: int, kernel_size: int = 31, dropout: float = 0.1
    ) -> None:
        super().__init__()
        raise NotImplementedError("ConvolutionModule の実装が必要")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """畳み込み変換を適用する。

        Args:
            x: 入力テンソル (batch, seq_len, d_model)。

        Returns:
            出力テンソル (batch, seq_len, d_model)。
        """
        raise NotImplementedError


class ConformerBlock(nn.Module):
    """単一のConformerブロック。

    Feed Forward (1/2) -> MHSA -> Convolution -> Feed Forward (1/2) -> LayerNorm の構成。
    残差接続を各サブモジュールに適用する。

    Args:
        d_model: モデル次元数。
        n_heads: マルチヘッドアテンションのヘッド数。
        ffn_dim: フィードフォワード層の内部次元数。
        kernel_size: 畳み込みカーネルサイズ。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError("ConformerBlock の実装が必要")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conformerブロックを適用する。

        Args:
            x: 入力テンソル (batch, seq_len, d_model)。

        Returns:
            出力テンソル (batch, seq_len, d_model)。
        """
        raise NotImplementedError
