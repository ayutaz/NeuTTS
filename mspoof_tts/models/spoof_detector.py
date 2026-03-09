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

import torch
import torch.nn as nn

from mspoof_tts.models.conformer import ConformerBlock


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
        raise NotImplementedError("SpoofDetector の実装が必要")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """トークン列からreal/fakeの確率を予測する。

        Args:
            token_ids: 離散トークンID列 (batch, seq_len)。

        Returns:
            real確率 (batch,)。値域は [0, 1]。
        """
        raise NotImplementedError
