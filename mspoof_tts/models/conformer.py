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
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フィードフォワード変換を適用する。

        Args:
            x: 入力テンソル (batch, seq_len, d_model)。

        Returns:
            出力テンソル (batch, seq_len, d_model)。
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


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
        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise expansion: d_model -> 2*d_model
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        # Depthwise convolution: groups=d_model for channel-wise processing
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        # Pointwise projection: d_model -> d_model
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """畳み込み変換を適用する。

        Args:
            x: 入力テンソル (batch, seq_len, d_model)。

        Returns:
            出力テンソル (batch, seq_len, d_model)。
        """
        x = self.layer_norm(x)
        # (batch, seq_len, d_model) -> (batch, d_model, seq_len) for Conv1d
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)  # (batch, 2*d_model, seq_len)
        x = self.glu(x)  # (batch, d_model, seq_len)
        x = self.depthwise_conv(x)  # (batch, d_model, seq_len)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # (batch, d_model, seq_len)
        x = self.dropout(x)
        # (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        return x


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
        # Macaron-style half-step feed-forward modules
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Convolution module (has internal LayerNorm)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        # Layer norms for each sub-module input
        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        # Final layer norm
        self.norm_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conformerブロックを適用する。

        Macaron構造に従い、以下の順序で処理を行う:
          x = x + 0.5 * FFN1(LayerNorm(x))
          x = x + MHSA(LayerNorm(x))
          x = x + Conv(LayerNorm(x))
          x = x + 0.5 * FFN2(LayerNorm(x))
          x = LayerNorm(x)

        Args:
            x: 入力テンソル (batch, seq_len, d_model)。

        Returns:
            出力テンソル (batch, seq_len, d_model)。
        """
        # First half-step FFN with residual
        x = x + 0.5 * self.ffn1(self.norm_ffn1(x))
        # Multi-head self-attention with residual
        attn_input = self.norm_attn(x)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input)
        x = x + self.dropout(attn_out)
        # Convolution module with residual
        x = x + self.conv(self.norm_conv(x))
        # Second half-step FFN with residual
        x = x + 0.5 * self.ffn2(self.norm_ffn2(x))
        # Final layer norm
        x = self.norm_final(x)
        return x
